#!/usr/bin/env python
"""Detect-only Drive run: download -> (frame-sample) -> detect -> crop -> save.

Consumes a media manifest CSV (file_id, path, mimeType, size) produced by
scripts/inventory_drive_folder.py and, for each item:

  1. downloads it to a bounded scratch dir (Shared-Drive aware),
  2. for video: samples frames (rate is a parameter); for image: uses the file,
  3. runs deepface DETECTION ONLY (no embed / cluster / share / upload),
  4. crops each detected face, applies the v1.1 size+blur quality gate,
     writes the crop JPEG (the only thing that persists),
  5. records a row per face in asset_faces.csv + a row per file in
     preprocessing_manifest.csv,
  6. in a try/finally, DELETES the downloaded source and its extracted frames
     so peak local disk stays ~(largest item + its frames), not the folder size.

Idempotent / resumable: every finished file_id is appended to progress.jsonl;
a re-run skips file_ids whose recorded status is terminal (re-tries only
transient download/detect errors).

Concurrency (Phase 4): the run is download-bound (wired single-stream
~35-41 MB/s; aggregate saturates ~100 MB/s at ~8 streams). Pass ``--workers N``
(8 recommended for FOLDER 2) to download N items in parallel while detection
stays single-threaded on one GPU. Concurrent downloads are bounded by
``--max-inflight-gb`` so peak scratch can't blow up on the big masters (single
files reach 211 GB). ``--workers 1`` (default) is the original sequential path.

Usage:
  CUDA_VISIBLE_DEVICES=0 run_detect_only.py MANIFEST.csv \
      --output-dir outputs/detect/folder1 \
      --scratch-dir /tmp/detect-scratch/folder1 \
      --frame-interval 2.0 --max-frames 60 \
      [--workers 8] [--max-inflight-gb 256] \
      [--max-crops-per-file N] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from unlabeled_media_tagger.drive.auth import get_drive_service
from unlabeled_media_tagger.drive.files import download_file
from unlabeled_media_tagger.pipeline.detect_faces import detect_faces_in_image
from unlabeled_media_tagger.pipeline.download_pool import ByteBudget, DownloadPool
from unlabeled_media_tagger.pipeline.extract import ExtractStage
from unlabeled_media_tagger.preprocessing.blur import (
    classify_quality,
    laplacian_variance,
)

ASSET_FACES_COLUMNS = [
    "source_file_id", "source_path", "mime_type",
    "frame_index", "timestamp_sec",
    "face_id", "face_index", "crop_file_name", "crop_path",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence",
    "detector_backend", "processed_at", "status", "quality_score", "error",
]

MANIFEST_COLUMNS = [
    "source_file_id", "source_path", "mime_type", "size",
    "media_type", "frames_sampled", "faces_detected", "crops_written",
    "status", "started_at", "completed_at", "download_sec", "detect_sec",
    "error",
]

# Per-file statuses that count as "successfully processed" — re-runs skip these.
# download_error / detect_error are excluded so transient failures get retried.
TERMINAL_STATUSES = {"ok", "no_faces"}

MIN_FACE_SIDE = 10  # bbox shorter side below this -> skipped_small (pre-crop)

GB = 1 << 30


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_done(progress_path: Path) -> set:
    """file_ids whose last recorded status is terminal."""
    done = {}
    if progress_path.exists():
        with progress_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                done[rec.get("source_file_id")] = rec.get("status")
    return {fid for fid, st in done.items() if st in TERMINAL_STATUSES}


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


class CsvAppender:
    """Append rows to a CSV, writing the header once if the file is new."""

    def __init__(self, path: Path, columns):
        self.path = path
        self.columns = columns
        if not path.exists():
            with path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=columns).writeheader()

    def write(self, rows):
        with self.path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self.columns})


def new_mrow(item) -> dict:
    """Fresh preprocessing_manifest.csv row for one item."""
    return {
        "source_file_id": item["file_id"], "source_path": item["path"],
        "mime_type": item.get("mimeType", ""), "size": item.get("size", ""),
        "media_type": "", "frames_sampled": 0, "faces_detected": 0,
        "crops_written": 0, "status": "", "started_at": iso_now(),
        "completed_at": "", "download_sec": "", "detect_sec": "", "error": "",
    }


def download_item(service, item, scratch_root) -> dict:
    """Download one manifest item to its scratch dir (DownloadPool download_fn).

    Returns the keys DownloadPool merges into the result dict. Raises on
    failure (the pool records it as download_error).
    """
    file_id = item["file_id"]
    name = Path(item["path"]).name or file_id
    item_scratch = Path(scratch_root) / file_id
    local_path = item_scratch / name
    t0 = time.time()
    download_file(service, file_id, str(local_path), supports_all_drives=True)
    return {
        "local_path": str(local_path), "name": name,
        "item_scratch": item_scratch, "download_sec": round(time.time() - t0, 2),
    }


def detect_and_crop(item, local_path, name, item_scratch, extract_stage,
                    crops_root, detector_backend, max_crops_per_file, mrow):
    """Frame-sample -> detect -> crop an already-downloaded file.

    Fills the detect-side fields of ``mrow`` and returns ``face_rows``. Raises
    on failure (callers classify as detect_error). Scratch cleanup is the
    caller's responsibility.
    """
    import cv2

    file_id = item["file_id"]
    crops_dir = crops_root / file_id
    face_rows = []

    # Point the extractor's frame output at this item's scratch so cleanup
    # removes source AND frames together.
    extract_stage.config["frame_dir"] = str(item_scratch / "frames")
    extracted = extract_stage.extract(str(local_path))
    mrow["media_type"] = extracted["media_type"]
    frames = extracted["frames"]
    mrow["frames_sampled"] = len(frames)

    t0 = time.time()
    crops_written = 0
    for frame in frames:
        if max_crops_per_file and crops_written >= max_crops_per_file:
            break
        fpath = frame["path"]
        f_index = frame.get("frame_index")
        ts = frame.get("timestamp_sec")
        bgr = cv2.imread(fpath)
        if bgr is None:
            continue
        h_img, w_img = bgr.shape[:2]
        detections = detect_faces_in_image(fpath, detector_backend)
        for face_index, det in enumerate(detections):
            if max_crops_per_file and crops_written >= max_crops_per_file:
                break
            b = det["bbox"]
            x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + bw), min(h_img, y + bh)
            base = {
                "source_file_id": file_id, "source_path": item["path"],
                "mime_type": item.get("mimeType", ""),
                "frame_index": "" if f_index is None else f_index,
                "timestamp_sec": "" if ts is None else round(ts, 3),
                "face_index": face_index,
                "face_id": f"{file_id}__f{f_index or 0}__face{face_index:02d}",
                "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
                "confidence": round(float(det.get("confidence", 0.0)), 5),
                "detector_backend": detector_backend,
                "processed_at": iso_now(),
                "crop_file_name": "", "crop_path": "",
                "quality_score": "", "error": "",
            }
            if (x2 - x1) < MIN_FACE_SIDE or (y2 - y1) < MIN_FACE_SIDE:
                base["status"] = "skipped_small"
                face_rows.append(base)
                continue
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                base["status"] = "skipped_small"
                base["error"] = "empty crop after clipping"
                face_rows.append(base)
                continue
            crops_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(name).stem
            fpart = f"_f{f_index:06d}" if f_index is not None else ""
            crop_file_name = f"{stem}{fpart}__face{face_index:02d}.jpg"
            crop_full = crops_dir / crop_file_name
            cv2.imwrite(str(crop_full),
                        crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            ch, cw = crop.shape[:2]
            qscore = laplacian_variance(crop)
            base["crop_file_name"] = crop_file_name
            base["crop_path"] = str(crop_full)
            base["quality_score"] = f"{qscore:.4f}"
            base["status"] = classify_quality(qscore, cw, ch)
            face_rows.append(base)
            crops_written += 1

    mrow["detect_sec"] = round(time.time() - t0, 2)
    mrow["faces_detected"] = len(face_rows)
    mrow["crops_written"] = crops_written
    return face_rows


def detect_from_result(result, extract_stage, crops_root, detector_backend,
                       max_crops_per_file):
    """Turn a DownloadPool result dict into (mrow, face_rows).

    Runs only on the consumer thread (single GPU). Classifies download_error /
    detect_error / ok / no_faces. Does NOT clean scratch — the pool does that.
    """
    item = result["item"]
    mrow = new_mrow(item)
    face_rows = []
    try:
        if not result["ok"]:
            mrow["status"] = "download_error"
            mrow["error"] = result.get("download_error", "download failed")
            return mrow, face_rows
        mrow["download_sec"] = result["download_sec"]
        face_rows = detect_and_crop(
            item, result["local_path"], result["name"], result["item_scratch"],
            extract_stage, crops_root, detector_backend, max_crops_per_file, mrow,
        )
        mrow["status"] = "ok" if face_rows else "no_faces"
    except Exception as exc:  # noqa: BLE001 - record + continue, don't abort run
        mrow["status"] = "detect_error"
        mrow["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        mrow["completed_at"] = iso_now()
    return mrow, face_rows


def process_item(item, service, extract_stage, crops_root, scratch_root,
                 detector_backend, max_crops_per_file, verbose):
    """Sequential (--workers 1) path: download -> detect one item inline.

    Always cleans up scratch in finally. Mirrors detect_from_result's status
    classification so the two paths produce identical rows.
    """
    item_scratch = scratch_root / item["file_id"]
    mrow = new_mrow(item)
    face_rows = []
    try:
        dl = download_item(service, item, scratch_root)
        mrow["download_sec"] = dl["download_sec"]
        face_rows = detect_and_crop(
            item, dl["local_path"], dl["name"], item_scratch,
            extract_stage, crops_root, detector_backend, max_crops_per_file, mrow,
        )
        mrow["status"] = "ok" if face_rows else "no_faces"
    except Exception as exc:  # noqa: BLE001 - record + continue, don't abort run
        stage = "detect_error" if mrow["download_sec"] != "" else "download_error"
        mrow["status"] = stage
        mrow["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ! {item['file_id']}: {mrow['error']}", flush=True)
    finally:
        # Delete the downloaded source AND its extracted frames. Crops live
        # under crops_root, NOT scratch, so they survive.
        shutil.rmtree(item_scratch, ignore_errors=True)
        mrow["completed_at"] = iso_now()
    return mrow, face_rows


class ResultSink:
    """Writes one finished item's rows to the CSVs + progress log + totals.

    Used by both the sequential and concurrent paths so output is identical.
    On the concurrent path the consumer is single-threaded, so no locking is
    needed here.
    """

    def __init__(self, asset_csv, manifest_csv, progress_path, total, verbose):
        self.asset_csv = asset_csv
        self.manifest_csv = manifest_csv
        self.progress_path = progress_path
        self.total = total
        self.verbose = verbose
        self.n = 0
        self.totals = {"faces": 0, "crops": 0, "ok": 0, "no_faces": 0, "error": 0}

    def record(self, mrow, face_rows):
        self.n += 1
        self.asset_csv.write(face_rows)
        self.manifest_csv.write([mrow])
        append_jsonl(self.progress_path, {
            "source_file_id": mrow["source_file_id"],
            "status": mrow["status"],
            "faces": mrow["faces_detected"],
            "crops": mrow["crops_written"],
            "frames": mrow["frames_sampled"],
        })
        self.totals["faces"] += mrow["faces_detected"]
        self.totals["crops"] += mrow["crops_written"]
        if mrow["status"] in ("ok", "no_faces"):
            self.totals[mrow["status"]] += 1
        else:
            self.totals["error"] += 1
        if self.verbose:
            print(f"[{self.n}/{self.total}] {Path(mrow['source_path']).name} "
                  f"-> {mrow['status']} media={mrow['media_type']} "
                  f"frames={mrow['frames_sampled']} faces={mrow['faces_detected']} "
                  f"crops={mrow['crops_written']} "
                  f"dl={mrow['download_sec']}s det={mrow['detect_sec']}s",
                  flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scratch-dir", required=True)
    ap.add_argument("--frame-interval", type=float, default=2.0,
                    help="Seconds between sampled video frames")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="Max frames sampled per video")
    ap.add_argument("--max-crops-per-file", type=int, default=None,
                    help="Cap crops written per source file (curbs video redundancy)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel download workers (8 recommended for FOLDER 2; "
                         "1 = sequential). Detection stays single-threaded.")
    ap.add_argument("--max-inflight-gb", type=float, default=256.0,
                    help="Cap on concurrently-downloaded bytes (peak scratch "
                         "ceiling). Only used when --workers > 1.")
    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_root = out / "crops"
    scratch_root = Path(args.scratch_dir)
    scratch_root.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    asset_csv = CsvAppender(out / "asset_faces.csv", ASSET_FACES_COLUMNS)
    manifest_csv = CsvAppender(out / "preprocessing_manifest.csv", MANIFEST_COLUMNS)

    with open(args.manifest) as fh:
        items = list(csv.DictReader(fh))

    done = load_done(progress_path)
    todo = [it for it in items if it["file_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    workers = max(1, args.workers)
    print(f"manifest items: {len(items)}  already-done: {len(done)}  "
          f"to-process: {len(todo)}  workers: {workers}", flush=True)

    extract_stage = ExtractStage({
        "frame_interval": args.frame_interval,
        "max_frames": args.max_frames,
    })
    sink = ResultSink(asset_csv, manifest_csv, progress_path, len(todo), verbose)

    if workers == 1:
        service = get_drive_service(verbose=verbose)
        for item in todo:
            mrow, face_rows = process_item(
                item, service, extract_stage, crops_root, scratch_root,
                args.detector_backend, args.max_crops_per_file, verbose,
            )
            sink.record(mrow, face_rows)
    else:
        # One Drive service per worker: googleapiclient's service object is not
        # thread-safe. Each is built from an independently-loaded Credentials,
        # so token refreshes during a long run are per-service and in-memory
        # (no concurrent writes to secrets/token.json).
        get_drive_service(verbose=verbose)  # ensure token is fresh up front
        services = [get_drive_service(verbose=False) for _ in range(workers)]
        budget = ByteBudget(int(args.max_inflight_gb * GB))
        pool = DownloadPool(
            download_fn=download_item, services=services,
            scratch_root=scratch_root, budget=budget,
        )

        def consume(result):
            mrow, face_rows = detect_from_result(
                result, extract_stage, crops_root,
                args.detector_backend, args.max_crops_per_file,
            )
            sink.record(mrow, face_rows)

        pool.run(todo, consume)

    t = sink.totals
    print(f"\n=== done === processed={len(todo)} ok={t['ok']} "
          f"no_faces={t['no_faces']} errors={t['error']} "
          f"faces={t['faces']} crops={t['crops']}", flush=True)
    print(f"crops dir: {crops_root}")
    print(f"asset_faces.csv: {out / 'asset_faces.csv'}")
    print(f"preprocessing_manifest.csv: {out / 'preprocessing_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
