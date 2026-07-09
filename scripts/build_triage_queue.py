#!/usr/bin/env python
"""Build a run's triage.csv: the ~200-300-cluster review budget.

Selection logic (and its rationale) lives in
unlabeled_media_tagger.review.triage; this writes <run-dir>/triage.csv,
which the review server picks up as the "triage queue" index view.
Deterministic for a given summary + seed; safe to re-run (confirms live in
the label store, keyed by face, not by this file).

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/build_triage_queue.py \
      --run-dir /mnt/media1/folder2_cluster/run_002
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from unlabeled_media_tagger.review.triage import build_triage

TRIAGE_COLUMNS = [
    "rank", "section", "reason", "cluster_id", "cluster_label",
    "n_nodes", "n_faces", "n_files", "n_events", "mean_similarity",
    "degenerate_suspect", "sibling_export_suspect", "static_face_suspect",
    "example_events",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--top-faces", type=int, default=50)
    ap.add_argument("--top-events", type=int, default=20)
    ap.add_argument("--random", type=int, default=100,
                    help="uniform sample size (cluster-level purity CI)")
    ap.add_argument("--size-weighted", type=int, default=25,
                    help="faces-weighted sample size (face-level purity)")
    ap.add_argument("--sibling-sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    with (run_dir / "cluster_summary.csv").open(newline="") as fh:
        summary = list(csv.DictReader(fh))

    triage = build_triage(
        summary, top_faces=args.top_faces, top_events=args.top_events,
        n_random=args.random, n_size_weighted=args.size_weighted,
        sibling_sample=args.sibling_sample, seed=args.seed)

    out_path = run_dir / "triage.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAGE_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(triage)

    counts = Counter(row["section"] for row in triage)
    for section in ["suspect_static", "suspect_degenerate", "suspect_sibling",
                    "largest_faces", "largest_events", "random_uniform",
                    "random_size_weighted"]:
        print(f"{section:>22}: {counts.get(section, 0)}")
    print(f"{'total':>22}: {len(triage)} of {len(summary)} clusters")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
