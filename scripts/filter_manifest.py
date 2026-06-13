#!/usr/bin/env python
"""Filter a media manifest CSV by dropping rows whose path matches a substring.

The inventory tool (scripts/inventory_drive_folder.py) walks the *whole* tree,
which on FOLDER 1 includes ~146 files that are PRIOR detect-only output — crops
sitting under `**/face_preprocessing/run_*/crops/`. Feeding those back into
run_detect_only would re-detect our own past crops. This produces a clean
manifest by excluding any row whose path contains one of the given substrings
(case-insensitive).

Quote-aware throughout: reads + writes with Python's csv module (paths contain
commas — never parse these manifests with awk -F, / cut -d,; see dev-log
2026-06-12).

Usage:
  filter_manifest.py IN.csv OUT.csv \
      [--exclude /face_preprocessing/ --exclude /crops/]   # default if omitted
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

DEFAULT_EXCLUDES = ["/face_preprocessing/", "/crops/"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--exclude", action="append", default=None,
                    help="Path substring to drop (case-insensitive). Repeatable. "
                         f"Defaults to: {' '.join(DEFAULT_EXCLUDES)}")
    ap.add_argument("--include-ext", default=None,
                    help="Comma-separated extensions to KEEP (e.g. "
                         "'.mp4,.mov,.jpg'). Rows whose path extension isn't "
                         "listed are dropped. Use to keep only decodable media "
                         "and avoid downloading files cv2 can't read.")
    ap.add_argument("--path-column", default="path")
    args = ap.parse_args(argv)
    excludes = [e.lower() for e in (args.exclude or DEFAULT_EXCLUDES)]
    include_ext = None
    if args.include_ext:
        include_ext = {("." + e.lstrip(".")).lower()
                       for e in args.include_ext.split(",") if e.strip()}

    with open(args.infile, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if args.path_column not in (fieldnames or []):
            raise SystemExit(
                f"path column {args.path_column!r} not in {fieldnames}")
        rows = list(reader)

    kept, dropped = [], 0
    for r in rows:
        p = (r.get(args.path_column) or "")
        pl = p.lower()
        if any(sub in pl for sub in excludes):
            dropped += 1
        elif include_ext is not None and os.path.splitext(pl)[1] not in include_ext:
            dropped += 1
        else:
            kept.append(r)

    with open(args.outfile, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"in: {len(rows)} rows  dropped: {dropped}  kept: {len(kept)}")
    print(f"excludes: {excludes}")
    if include_ext is not None:
        print(f"include-ext: {sorted(include_ext)}")
    print(f"wrote: {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
