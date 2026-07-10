#!/usr/bin/env python
"""Build <run-dir>/cluster_neighbors.csv: merge-discovery suggestions.

Selection logic lives in unlabeled_media_tagger.review.neighbors; this
computes per-cluster centroid nearest neighbors and writes the file the
review server renders as the "similar clusters" strip on group pages and
the /merges candidate list. Re-run any time — it is derived data only.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/build_cluster_neighbors.py \
      --run-dir /mnt/media1/folder2_cluster/run_002
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from unlabeled_media_tagger.review.neighbors import (
    DEFAULT_K,
    DEFAULT_MIN_COS,
    cluster_neighbors,
)

NEIGHBOR_COLUMNS = ["cluster_id", "rank", "neighbor_cluster_id", "cosine",
                    "shared_events"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--min-cos", type=float, default=DEFAULT_MIN_COS)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / "params.json").read_text())
    embed_dir = Path(params["embed_dir"])

    members = defaultdict(list)
    events = defaultdict(set)
    with (run_dir / "clusters.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["cluster_id"] != "":
                members[row["cluster_id"]].append(int(row["node_id"]))
                events[row["cluster_id"]].add(row["event_path"])
    embeddings = np.load(embed_dir / "nodes" / "node_embeddings.npy")

    rows = cluster_neighbors(members, embeddings, events,
                             k=args.k, min_cos=args.min_cos)

    # tmp + rename: the review server tolerates but skips corrupt rows, so
    # never leave a half-written file for it to find.
    out_path = run_dir / "cluster_neighbors.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=NEIGHBOR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, out_path)

    with_neighbors = {r["cluster_id"] for r in rows}
    pairs = {tuple(sorted((r["cluster_id"], r["neighbor_cluster_id"])))
             for r in rows}
    strong = {tuple(sorted((r["cluster_id"], r["neighbor_cluster_id"])))
              for r in rows if r["cosine"] >= 0.85}
    print(f"{len(rows)} suggestions across {len(with_neighbors)}/"
          f"{len(members)} clusters (cos >= {args.min_cos}, k={args.k})")
    print(f"unique pairs: {len(pairs)} ({len(strong)} at cos >= 0.85)")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
