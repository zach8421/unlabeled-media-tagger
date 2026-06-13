#!/usr/bin/env python
"""Split a media manifest CSV into byte-bounded chunks for staged processing.

FOLDER 2 is ~98 TB / 140k files — far too big for one detect-only run. This
greedily packs rows (in input order, or sorted by size) into chunk CSVs each
holding at most ``--chunk-gb`` of source bytes, so the run can be done a chunk
at a time ("start small, let it run, check"). A single file larger than the
chunk budget becomes its own solo chunk (flagged) rather than being split or
dropped — detect-only downloads whole files, so a 227 GB master is one
indivisible unit of work.

Idempotency across chunks is automatic: run_detect_only.py keeps its own
progress.jsonl, so pointing it at chunk after chunk (or re-running one) never
re-processes a done file.

Quote-aware in+out (paths contain commas — never awk -F, these manifests).

Usage:
  shard_manifest.py IN.csv OUTDIR --chunk-gb 150 \
      [--max-files-per-chunk N] [--limit-chunks K] \
      [--sort none|size_asc|size_desc] [--prefix chunk]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

GB = 1_000_000_000  # decimal GB, matching Drive's reported sizes


def _size(row) -> int:
    try:
        return int(row.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outdir")
    ap.add_argument("--chunk-gb", type=float, required=True,
                    help="Max source bytes per chunk (decimal GB).")
    ap.add_argument("--max-files-per-chunk", type=int, default=None,
                    help="Also cap files per chunk (bounds detection time when "
                         "a chunk is mostly tiny files).")
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="Only write the first K chunks (e.g. 1 to start small).")
    ap.add_argument("--sort", choices=["none", "size_asc", "size_desc"],
                    default="none",
                    help="Order to pack rows. 'none' preserves walk order "
                         "(keeps event folders contiguous).")
    ap.add_argument("--prefix", default="chunk")
    args = ap.parse_args(argv)

    cap = int(args.chunk_gb * GB)
    with open(args.infile, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if "size" not in (fieldnames or []):
        raise SystemExit(f"manifest has no 'size' column: {fieldnames}")

    if args.sort == "size_asc":
        rows.sort(key=_size)
    elif args.sort == "size_desc":
        rows.sort(key=_size, reverse=True)

    # Greedy pack: close the current chunk when the next row would overflow the
    # byte cap (or the file cap). A row bigger than `cap` lands alone.
    chunks, cur, cur_bytes = [], [], 0
    for r in rows:
        s = _size(r)
        over_bytes = cur and (cur_bytes + s > cap)
        over_files = (args.max_files_per_chunk and cur
                      and len(cur) >= args.max_files_per_chunk)
        if over_bytes or over_files:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += s
    if cur:
        chunks.append(cur)

    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(chunks) - 1)))
    print(f"{len(rows)} rows -> {len(chunks)} chunk(s) "
          f"(cap {args.chunk_gb} GB, sort={args.sort})")
    print(f"{'chunk':>20} {'files':>7} {'GB':>9}  solo-giant")
    for i, ch in enumerate(chunks):
        name = f"{args.prefix}_{i:0{width}d}.csv"
        path = outdir / name
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(ch)
        cb = sum(_size(r) for r in ch)
        solo = "  <-- >cap" if (len(ch) == 1 and cb > cap) else ""
        print(f"{name:>20} {len(ch):7d} {cb / GB:9.1f}{solo}")
    print(f"wrote {len(chunks)} chunk(s) to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
