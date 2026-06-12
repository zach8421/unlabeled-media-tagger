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
transient download/detect errors). Dual-GPU = run two of these on disjoint
manifest shards with CUDA_VISIBLE_DEVICES=0 / =1 and separate --output-dir /
--scratch-dir.

Usage:
  CUDA_VISIBLE_DEVICES=0 run_detect_only.py MANIFEST.csv \
      --output-dir outputs/detect/folder1 \
      --scratch-dir /tmp/detect-scratch/folder1 \
      --frame-interval 2.0 --max-frames 60 [--max-crops-per-file N] [--limit N]
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


def process_item(item, service, extract_stage, crops_root, scratch_root,
                 detector_backend, max_crops_per_file, verbose):
    """Download -> sample -> detect -> crop one manifest item.

    Returns (manifest_row, face_rows). Always cleans up scratch in finally.
    """
    import cv2

    file_id = item["file_id"]
    name = Path(item["path"]).name or file_id
    mime = item.get("mimeType", "")
    item_scratch = scratch_root / file_id
    crops_dir = crops_root / file_id

    mrow = {
        "source_file_id": file_id, "source_path": item["path"],
        "mime_type": mime, "size": item.get("size", ""),
        "media_type": "", "frames_sampled": 0, "faces_detected": 0,
        "crops_written": 0, "status": "", "started_at": iso_now(),
        "completed_at": "", "download_sec": "", "detect_sec": "", "error": "",
    }
    face_rows = []

    try:
        local_path = item_scratch / name
        t0 = time.time()
        download_file(service, file_id, str(local_path), supports_all_drives=True)
        mrow["download_sec"] = round(time.time() - t0, 2)

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
                    "mime_type": mime,
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
        mrow["status"] = "ok" if face_rows else "no_faces"

    except Exception as exc:  # noqa: BLE001 - record + continue, don't abort run
        stage = "detect_error" if mrow["download_sec"] != "" else "download_error"
        mrow["status"] = stage
        mrow["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ! {file_id} {name}: {mrow['error']}", flush=True)
    finally:
        # Delete the downloaded source AND its extracted frames. Crops live
        # under crops_root, NOT scratch, so they survive.
        shutil.rmtree(item_scratch, ignore_errors=True)
        mrow["completed_at"] = iso_now()

    return mrow, face_rows


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
    print(f"manifest items: {len(items)}  already-done: {len(done)}  "
          f"to-process: {len(todo)}", flush=True)

    service = get_drive_service(verbose=verbose)
    extract_stage = ExtractStage({
        "frame_interval": args.frame_interval,
        "max_frames": args.max_frames,
    })

    totals = {"faces": 0, "crops": 0, "ok": 0, "no_faces": 0, "error": 0}
    for i, item in enumerate(todo, start=1):
        if verbose:
            print(f"[{i}/{len(todo)}] {Path(item['path']).name} "
                  f"({item.get('mimeType','')})", flush=True)
        mrow, face_rows = process_item(
            item, service, extract_stage, crops_root, scratch_root,
            args.detector_backend, args.max_crops_per_file, verbose,
        )
        asset_csv.write(face_rows)
        manifest_csv.write([mrow])
        append_jsonl(progress_path, {
            "source_file_id": mrow["source_file_id"],
            "status": mrow["status"],
            "faces": mrow["faces_detected"],
            "crops": mrow["crops_written"],
            "frames": mrow["frames_sampled"],
        })
        totals["faces"] += mrow["faces_detected"]
        totals["crops"] += mrow["crops_written"]
        if mrow["status"] in ("ok", "no_faces"):
            totals[mrow["status"]] += 1
        else:
            totals["error"] += 1
        if verbose:
            print(f"    {mrow['status']}  media={mrow['media_type']} "
                  f"frames={mrow['frames_sampled']} faces={mrow['faces_detected']} "
                  f"crops={mrow['crops_written']} "
                  f"dl={mrow['download_sec']}s det={mrow['detect_sec']}s",
                  flush=True)

    print(f"\n=== done === processed={len(todo)} ok={totals['ok']} "
          f"no_faces={totals['no_faces']} errors={totals['error']} "
          f"faces={totals['faces']} crops={totals['crops']}", flush=True)
    print(f"crops dir: {crops_root}")
    print(f"asset_faces.csv: {out / 'asset_faces.csv'}")
    print(f"preprocessing_manifest.csv: {out / 'preprocessing_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
