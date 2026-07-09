#!/usr/bin/env python
"""Batched ArcFace embedding of an embed_manifest.csv worth of face crops.

Reads the manifest produced by scripts/build_embed_manifest.py, embeds every
crop with the SAME math as
`unlabeled_media_tagger.preprocessing.embed.embed_face_crop_bgr` (the parity
oracle), and writes L2-normalized float32 embeddings in resumable shards.

Why not DeepFace.represent per crop: it is one forward pass per image. The
installed deepface's `represent` skip-path is, verbatim:
  load_image (ndarray passes through) -> flip -> flip (cancels) ->
  preprocessing.resize_image(target_size) -> normalize_input("base") (no-op)
  -> model.forward (predict_on_batch when N>1).
So this script imports deepface's own `resize_image` and `build_model` and
feeds fixed-size batches to `model.forward` — the identical pipeline minus the
per-image call overhead. `--self-check N` proves parity on real crops (cosine
against the oracle) and must pass before a full run.

Layout of --output-dir (default /mnt/media1/folder2_embed):
  embeddings/shard_NNNNN.npy   float32 (n, 512), rows L2-normalized
  index/shard_NNNNN.csv        row-aligned metadata incl. raw_norm
  read_failed/shard_NNNNN.csv  crops that failed decode (usually none)
  progress.jsonl               one record per finished shard (resume unit)
  run_meta.json                manifest fingerprint; resume refuses on mismatch
  faces_index.csv              concatenation of index shards (finalize step)
  read_failed.csv              concatenation of read_failed shards (finalize)

Usage:
  CUDA_VISIBLE_DEVICES=0 run_embed_batch.py \
      --manifest /mnt/media1/folder2_embed/embed_manifest.csv \
      --output-dir /mnt/media1/folder2_embed \
      [--self-check 1000] [--shard-size 8192] [--batch-size 128] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_NAME = "ArcFace"
EMBED_DIM = 512

INDEX_COLUMNS = [
    "shard", "row", "source_file_id", "crop_file_name", "media_type",
    "frame_index", "timestamp_sec", "quality_score", "raw_norm",
]
READ_FAILED_COLUMNS = [
    "shard", "source_file_id", "crop_file_name", "error",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_done_shards(progress_path: Path) -> set:
    done = set()
    if progress_path.exists():
        with progress_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("status") == "ok":
                    done.add(rec.get("shard"))
    return done


def build_model():
    """Build the ArcFace model via deepface's own factory (one instance)."""
    from deepface.modules import modeling

    model = modeling.build_model(task="facial_recognition",
                                 model_name=MODEL_NAME)
    assert model.output_shape == EMBED_DIM, model.output_shape
    return model


def make_preprocessor(model):
    """Return crop_path -> (1, H, W, 3) float32, exactly the represent skip path.

    represent() passes target_size=(input_shape[1], input_shape[0]) — swapped —
    so we do too (symmetric for ArcFace's 112x112, but replicated exactly).
    Returns None when the file can't be decoded or is degenerate (<10px side,
    mirroring embed_face_crop_bgr's guard).
    """
    import cv2

    from deepface.modules import preprocessing

    target_size = (model.input_shape[1], model.input_shape[0])

    def preprocess(crop_path: str):
        bgr = cv2.imread(crop_path)
        if bgr is None or bgr.size == 0:
            return None
        if bgr.shape[0] < 10 or bgr.shape[1] < 10:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return preprocessing.resize_image(img=rgb, target_size=target_size)

    return preprocess


def embed_arrays(model, arrays: list, batch_size: int) -> np.ndarray:
    """Forward fixed-size batches; returns float32 (len(arrays), 512), raw."""
    out = np.empty((len(arrays), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(arrays), batch_size):
        chunk = arrays[start:start + batch_size]
        batch = np.concatenate(chunk, axis=0)
        if len(chunk) < batch_size:
            # Fixed batch shape (pad + slice) so TF never sees a new shape.
            pad = np.zeros(
                (batch_size - len(chunk),) + batch.shape[1:], dtype=batch.dtype)
            batch = np.concatenate([batch, pad], axis=0)
        embeddings = np.asarray(model.forward(batch), dtype=np.float32)
        out[start:start + len(chunk)] = embeddings[: len(chunk)]
    return out


def read_manifest(manifest_path: Path, limit=None) -> list:
    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:limit] if limit is not None else rows


def shard_bounds(n_rows: int, shard_size: int):
    return [(i, s, min(s + shard_size, n_rows))
            for i, s in enumerate(range(0, n_rows, shard_size))]


def crop_path_for(crops_root: Path, row: dict) -> str:
    return str(crops_root / row["source_file_id"] / row["crop_file_name"])


def decode_shard(rows, crops_root, preprocess, pool) -> list:
    """Decode+preprocess one shard's crops in parallel; order-preserving."""
    paths = [crop_path_for(crops_root, r) for r in rows]
    return list(pool.map(preprocess, paths))


def write_shard(out_dir: Path, shard_id: int, rows, arrays, model,
                batch_size: int) -> dict:
    """Embed one decoded shard and atomically write npy + index + failures."""
    kept_rows, kept_arrays, failed = [], [], []
    for row, arr in zip(rows, arrays):
        if arr is None:
            failed.append({
                "shard": shard_id,
                "source_file_id": row["source_file_id"],
                "crop_file_name": row["crop_file_name"],
                "error": "decode failed or degenerate crop",
            })
        else:
            kept_rows.append(row)
            kept_arrays.append(arr)

    if kept_arrays:
        raw = embed_arrays(model, kept_arrays, batch_size)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = raw / norms
    else:
        unit = np.empty((0, EMBED_DIM), dtype=np.float32)
        norms = np.empty((0, 1), dtype=np.float32)

    npy_path = out_dir / "embeddings" / f"shard_{shard_id:05d}.npy"
    idx_path = out_dir / "index" / f"shard_{shard_id:05d}.csv"
    fail_path = out_dir / "read_failed" / f"shard_{shard_id:05d}.csv"

    # np.save appends ".npy" unless the name already ends with it, so the
    # tmp file must keep the suffix: shard_00000.tmp.npy -> shard_00000.npy
    tmp_npy = npy_path.with_name(npy_path.stem + ".tmp.npy")
    np.save(tmp_npy, unit)
    os.replace(tmp_npy, npy_path)

    tmp_idx = idx_path.with_suffix(".csv.tmp")
    with tmp_idx.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(kept_rows):
            writer.writerow({
                "shard": shard_id, "row": i,
                "source_file_id": row["source_file_id"],
                "crop_file_name": row["crop_file_name"],
                "media_type": row["media_type"],
                "frame_index": row["frame_index"],
                "timestamp_sec": row["timestamp_sec"],
                "quality_score": row["quality_score"],
                "raw_norm": f"{float(norms[i][0]):.4f}",
            })
    os.replace(tmp_idx, idx_path)

    if failed:
        tmp_fail = fail_path.with_suffix(".csv.tmp")
        with tmp_fail.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=READ_FAILED_COLUMNS)
            writer.writeheader()
            writer.writerows(failed)
        os.replace(tmp_fail, fail_path)

    return {"embedded": len(kept_rows), "read_failed": len(failed)}


def self_check(model, preprocess, manifest_rows, crops_root, n: int) -> bool:
    """Batch path vs embed_face_crop_bgr on n evenly-spaced real crops."""
    import cv2

    from unlabeled_media_tagger.preprocessing.embed import embed_face_crop_bgr

    step = max(1, len(manifest_rows) // n)
    sample = manifest_rows[::step][:n]
    arrays, oracle = [], []
    for row in sample:
        path = crop_path_for(crops_root, row)
        arr = preprocess(path)
        single = embed_face_crop_bgr(cv2.imread(path))
        if arr is None or single is None:
            continue
        arrays.append(arr)
        oracle.append(single)
    batch = embed_arrays(model, arrays, batch_size=128)
    batch /= np.linalg.norm(batch, axis=1, keepdims=True)
    ref = np.asarray(oracle, dtype=np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cosines = np.sum(batch * ref, axis=1)
    max_abs_diff = float(np.max(np.abs(batch - ref)))
    print(f"self-check n={len(arrays)}: cosine min={cosines.min():.6f} "
          f"mean={cosines.mean():.6f} | max |delta| component={max_abs_diff:.2e}",
          flush=True)
    passed = bool(cosines.min() >= 0.9999)
    print(f"self-check {'PASS' if passed else 'FAIL'} (gate: min cosine >= 0.9999)",
          flush=True)
    return passed


def finalize(out_dir: Path, shards, manifest_rows) -> int:
    """Concatenate shard indexes, reconcile counts, sanity-check norms."""
    total_indexed, total_failed = 0, 0
    faces_tmp = out_dir / "faces_index.csv.tmp"
    with faces_tmp.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for shard_id, _, _ in shards:
            idx_path = out_dir / "index" / f"shard_{shard_id:05d}.csv"
            with idx_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    writer.writerow(row)
                    total_indexed += 1
    os.replace(faces_tmp, out_dir / "faces_index.csv")

    failed_tmp = out_dir / "read_failed.csv.tmp"
    with failed_tmp.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=READ_FAILED_COLUMNS)
        writer.writeheader()
        for shard_id, _, _ in shards:
            fail_path = out_dir / "read_failed" / f"shard_{shard_id:05d}.csv"
            if fail_path.exists():
                with fail_path.open(newline="") as fh:
                    for row in csv.DictReader(fh):
                        writer.writerow(row)
                        total_failed += 1
    os.replace(failed_tmp, out_dir / "read_failed.csv")

    # Row-count + norm sanity on a few shards.
    for shard_id, start, end in shards[:3] + shards[-3:]:
        emb = np.load(out_dir / "embeddings" / f"shard_{shard_id:05d}.npy")
        norms = np.linalg.norm(emb, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5), f"shard {shard_id} not unit-norm"

    ok = total_indexed + total_failed == len(manifest_rows)
    print(f"finalize: indexed={total_indexed} read_failed={total_failed} "
          f"manifest={len(manifest_rows)} -> "
          f"{'RECONCILED' if ok else 'MISMATCH'}", flush=True)
    print(f"faces_index.csv: {out_dir / 'faces_index.csv'}", flush=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/mnt/media1/folder2_embed/embed_manifest.csv")
    ap.add_argument("--crops-root",
                    default="/mnt/media1/folder2_detect/crops")
    ap.add_argument("--output-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--shard-size", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--decode-threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N manifest rows (smoke tests)")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="Parity-check N crops against embed_face_crop_bgr "
                         "and exit (no shard writes)")
    args = ap.parse_args(argv)

    import cv2
    cv2.setNumThreads(1)  # parallelism comes from the decode pool

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    out_dir = Path(args.output_dir)
    crops_root = Path(args.crops_root)
    manifest_path = Path(args.manifest)
    for sub in ("embeddings", "index", "read_failed"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(manifest_path, limit=args.limit)
    print(f"manifest rows: {len(manifest_rows)}  shard_size: {args.shard_size} "
          f" batch_size: {args.batch_size}", flush=True)

    print("building ArcFace model...", flush=True)
    model = build_model()
    preprocess = make_preprocessor(model)

    if args.self_check:
        return 0 if self_check(model, preprocess, manifest_rows, crops_root,
                               args.self_check) else 1

    # Resume bookkeeping: shard membership is a pure function of the sorted
    # manifest + shard size; refuse to resume if either changed.
    meta_path = out_dir / "run_meta.json"
    meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(manifest_rows),
        "shard_size": args.shard_size,
        "model_name": MODEL_NAME,
        "limit": args.limit,
    }
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        for key in ("manifest_sha256", "manifest_rows", "shard_size", "limit"):
            if prior.get(key) != meta[key]:
                print(f"REFUSING RESUME: run_meta.json {key} changed "
                      f"({prior.get(key)} -> {meta[key]}). Move the old run "
                      f"aside or delete progress.jsonl + shards.", flush=True)
                return 2
    else:
        meta["started_at"] = iso_now()
        meta_path.write_text(json.dumps(meta, indent=1))

    progress_path = out_dir / "progress.jsonl"
    done = load_done_shards(progress_path)
    shards = shard_bounds(len(manifest_rows), args.shard_size)
    todo = [(sid, s, e) for sid, s, e in shards if sid not in done]
    print(f"shards: {len(shards)} total, {len(done)} done, {len(todo)} to run",
          flush=True)

    totals = {"embedded": 0, "read_failed": 0}
    with ThreadPoolExecutor(max_workers=args.decode_threads) as pool, \
            ThreadPoolExecutor(max_workers=1) as prefetcher:
        # Prefetch: decode shard i+1 on a side thread while the GPU works on i.
        next_decode = None
        for pos, (shard_id, start, end) in enumerate(todo):
            rows = manifest_rows[start:end]
            if next_decode is None:
                arrays = decode_shard(rows, crops_root, preprocess, pool)
            else:
                arrays = next_decode.result()
            if pos + 1 < len(todo):
                _, nstart, nend = todo[pos + 1]
                next_decode = prefetcher.submit(
                    decode_shard, manifest_rows[nstart:nend], crops_root,
                    preprocess, pool)
            else:
                next_decode = None

            t0 = time.time()
            stats = write_shard(out_dir, shard_id, rows, arrays, model,
                                args.batch_size)
            elapsed = round(time.time() - t0, 2)
            totals["embedded"] += stats["embedded"]
            totals["read_failed"] += stats["read_failed"]
            append_jsonl(progress_path, {
                "shard": shard_id, "status": "ok",
                "faces": stats["embedded"], "read_failed": stats["read_failed"],
                "sec": elapsed,
                "faces_per_sec": round(stats["embedded"] / elapsed, 1)
                if elapsed else 0.0,
                "at": iso_now(),
            })
            print(f"[{pos + 1}/{len(todo)}] shard {shard_id:05d}: "
                  f"{stats['embedded']} faces in {elapsed}s "
                  f"({stats['embedded'] / elapsed if elapsed else 0:.0f}/s) "
                  f"read_failed={stats['read_failed']}", flush=True)

    print(f"\n=== embed done === embedded={totals['embedded']} "
          f"read_failed={totals['read_failed']}", flush=True)
    return finalize(out_dir, shards, manifest_rows)


if __name__ == "__main__":
    sys.exit(main())
