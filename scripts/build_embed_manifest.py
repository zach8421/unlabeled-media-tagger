#!/usr/bin/env python
"""Build the embedding work manifest from a detect-only run's asset_faces.csv.

Selects the faces worth embedding (quality-gate ``status == "ok"``), verifies
each crop actually exists on disk, and writes a deterministic, sorted manifest
that scripts/run_embed_batch.py shards and consumes.

Key facts this script encodes (from the FOLDER 2 run):
  - The stable per-face key is (source_file_id, crop_file_name). The
    ``crop_path`` column is NOT reliable (stale for seeded validation rows) —
    the crop locator is always <crops_root>/<source_file_id>/<crop_file_name>.
  - asset_faces.csv has commas inside source_path; only quote-aware csv
    parsing is safe. The file is ~1.9 GB — stream it, never load it whole.
  - preprocessing_manifest.csv has duplicate rows for retried files; the join
    keeps the row that actually recorded a media_type.

Usage:
  build_embed_manifest.py \
      --detect-dir /mnt/media1/folder2_detect \
      --output-dir /mnt/media1/folder2_embed
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

MANIFEST_COLUMNS = [
    "source_file_id", "crop_file_name", "media_type",
    "frame_index", "timestamp_sec",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "confidence", "quality_score",
]

MISSING_COLUMNS = MANIFEST_COLUMNS + ["reason"]


def load_media_types(preprocessing_manifest: Path) -> dict:
    """Map source_file_id -> media_type, tolerating duplicate retry rows.

    A retried file appears more than once; failed attempts have an empty
    media_type. Keep the first non-empty value per file_id.
    """
    media_types: dict[str, str] = {}
    with preprocessing_manifest.open(newline="") as fh:
        for row in csv.DictReader(fh):
            fid = row["source_file_id"]
            mtype = (row.get("media_type") or "").strip()
            if mtype and not media_types.get(fid):
                media_types[fid] = mtype
    return media_types


def build_manifest(asset_faces: Path, crops_root: Path, media_types: dict,
                   limit: int | None = None):
    """Stream asset_faces.csv -> (kept_rows, counters, missing_rows)."""
    counters = {
        "rows_scanned": 0, "ok_rows": 0, "dupes_dropped": 0,
        "no_crop_name": 0, "missing_on_disk": 0, "kept": 0,
    }
    seen: set[tuple[str, str]] = set()
    kept: list[list] = []
    missing: list[list] = []

    with asset_faces.open(newline="") as fh:
        for row in csv.DictReader(fh):
            counters["rows_scanned"] += 1
            if row["status"] != "ok":
                continue
            counters["ok_rows"] += 1

            fid = row["source_file_id"]
            crop_name = row["crop_file_name"]
            if not crop_name:
                counters["no_crop_name"] += 1
                continue
            key = (fid, crop_name)
            if key in seen:
                counters["dupes_dropped"] += 1
                continue
            seen.add(key)

            out = [
                fid, crop_name,
                media_types.get(fid, ""),
                row["frame_index"], row["timestamp_sec"],
                row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"],
                row["confidence"], row["quality_score"],
            ]
            if not (crops_root / fid / crop_name).is_file():
                counters["missing_on_disk"] += 1
                missing.append(out + ["crop file not found"])
                continue

            kept.append(out)
            counters["kept"] += 1
            if limit is not None and counters["kept"] >= limit:
                break

    kept.sort(key=lambda r: (r[0], r[1]))
    missing.sort(key=lambda r: (r[0], r[1]))
    return kept, counters, missing


def write_csv_atomic(path: Path, columns: list, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect-dir", default="/mnt/media1/folder2_detect",
                    help="Detect-run output dir holding asset_faces.csv, "
                         "preprocessing_manifest.csv and crops/")
    ap.add_argument("--output-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after keeping N faces (smoke tests)")
    args = ap.parse_args(argv)

    detect_dir = Path(args.detect_dir)
    asset_faces = detect_dir / "asset_faces.csv"
    preprocessing_manifest = detect_dir / "preprocessing_manifest.csv"
    crops_root = detect_dir / "crops"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media_types = load_media_types(preprocessing_manifest)
    print(f"media_type join: {len(media_types)} file_ids", flush=True)

    kept, counters, missing = build_manifest(
        asset_faces, crops_root, media_types, limit=args.limit)

    manifest_path = out_dir / "embed_manifest.csv"
    missing_path = out_dir / "missing_crops.csv"
    write_csv_atomic(manifest_path, MANIFEST_COLUMNS, kept)
    write_csv_atomic(missing_path, MISSING_COLUMNS, missing)

    no_media_type = sum(1 for r in kept if not r[2])
    print(f"rows scanned:      {counters['rows_scanned']}")
    print(f"status=ok rows:    {counters['ok_rows']}")
    print(f"dupes dropped:     {counters['dupes_dropped']}")
    print(f"no crop_file_name: {counters['no_crop_name']}")
    print(f"missing on disk:   {counters['missing_on_disk']}")
    print(f"manifest rows out: {counters['kept']}")
    print(f"rows w/o media_type: {no_media_type}")
    reconciled = (counters["kept"] + counters["dupes_dropped"]
                  + counters["no_crop_name"] + counters["missing_on_disk"])
    label = "OK" if reconciled == counters["ok_rows"] else "MISMATCH"
    print(f"reconciliation:    kept+dupes+no_name+missing = {reconciled} "
          f"vs ok rows {counters['ok_rows']} -> {label}")
    print(f"wrote {manifest_path}")
    print(f"wrote {missing_path}")
    return 0 if label == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
