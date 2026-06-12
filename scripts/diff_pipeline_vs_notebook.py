"""Compare a local-pipeline face_clusters.csv against a notebook one.

Usage:
    ./.venv/bin/python scripts/diff_pipeline_vs_notebook.py \\
        --pipeline outputs/pipeline/face_clusters.csv \\
        --notebook /path/pulled/from/drive/share_package/face_clusters.csv

The two pipelines should produce the same partition of faces over the same
source Drive folder. Cluster *ids* will not match (clustering is order-
dependent and the two paths process files in different orders), but the
*partition* should — i.e. every pair of faces clustered together by the
pipeline should also be clustered together by the notebook.

Reports:
  - row counts on each side
  - row keys ((drive_id, face_index)) found only in one side
  - cluster-count parity
  - partition agreement: percentage of pipeline-cluster members that map to
    a single notebook cluster, and vice versa
  - any "split clusters" (one pipeline cluster spread across multiple
    notebook clusters) and "merged clusters" (vice versa)

Exit code 0 when the partitions match exactly, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_face_clusters(csv_path: Path) -> dict[tuple[str, str], str]:
    """Return {(drive_id, face_index): cluster_label} for one CSV."""
    rows: dict[tuple[str, str], str] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("drive_id", ""), str(row.get("face_index", "")))
            label = row.get("cluster_label", "")
            rows[key] = label
    return rows


def compute_partition_alignment(
    a_label_by_key: dict[tuple, str],
    b_label_by_key: dict[tuple, str],
) -> dict:
    """Build a → b mapping and report on alignment."""
    shared_keys = set(a_label_by_key) & set(b_label_by_key)

    # For each cluster in A, which clusters in B do its members fall into?
    a_to_b: dict[str, Counter] = defaultdict(Counter)
    for key in shared_keys:
        a_label = a_label_by_key[key]
        b_label = b_label_by_key[key]
        a_to_b[a_label][b_label] += 1

    # A cluster "splits" if its members fall into >1 cluster on the other side.
    split_in_b = {
        label: dict(counter)
        for label, counter in a_to_b.items()
        if len(counter) > 1
    }

    # Total agreement: a row "agrees" if it's in the same a→b mapping that
    # the majority of its a-cluster's members chose. So per-row agreement =
    # (max count in counter) / (total in counter), aggregated across rows.
    correct = 0
    total = 0
    for label, counter in a_to_b.items():
        cluster_total = sum(counter.values())
        majority = max(counter.values())
        correct += majority
        total += cluster_total
    agreement_pct = (100.0 * correct / total) if total else 0.0

    return {
        "shared_rows": len(shared_keys),
        "a_clusters": len({a_label_by_key[k] for k in shared_keys}),
        "agreement_pct": agreement_pct,
        "split_clusters": split_in_b,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline", required=True, type=Path,
        help="Local pipeline face_clusters.csv (typically outputs/pipeline/face_clusters.csv)",
    )
    parser.add_argument(
        "--notebook", required=True, type=Path,
        help="Notebook face_clusters.csv (typically pulled from "
             "{PROJECT_ROOT}/share_package/face_clusters.csv on Drive)",
    )
    args = parser.parse_args(argv)

    pipeline = read_face_clusters(args.pipeline)
    notebook = read_face_clusters(args.notebook)

    pipeline_keys = set(pipeline)
    notebook_keys = set(notebook)
    only_pipeline = pipeline_keys - notebook_keys
    only_notebook = notebook_keys - pipeline_keys

    print(f"pipeline rows: {len(pipeline)}")
    print(f"notebook rows: {len(notebook)}")
    print(f"shared row keys (drive_id, face_index): {len(pipeline_keys & notebook_keys)}")
    print(f"only in pipeline: {len(only_pipeline)}")
    print(f"only in notebook: {len(only_notebook)}")

    if only_pipeline:
        sample = list(sorted(only_pipeline))[:5]
        print(f"  sample pipeline-only keys: {sample}")
    if only_notebook:
        sample = list(sorted(only_notebook))[:5]
        print(f"  sample notebook-only keys: {sample}")

    pipeline_cluster_count = len({pipeline[k] for k in pipeline_keys})
    notebook_cluster_count = len({notebook[k] for k in notebook_keys})
    print(f"pipeline clusters: {pipeline_cluster_count}")
    print(f"notebook clusters: {notebook_cluster_count}")

    # Partition agreement runs in both directions: pipeline→notebook and
    # notebook→pipeline. Both numbers tell you something useful.
    pipeline_to_notebook = compute_partition_alignment(pipeline, notebook)
    notebook_to_pipeline = compute_partition_alignment(notebook, pipeline)

    print()
    print(f"pipeline→notebook agreement: {pipeline_to_notebook['agreement_pct']:.2f}%")
    print(f"notebook→pipeline agreement: {notebook_to_pipeline['agreement_pct']:.2f}%")

    pipeline_splits = pipeline_to_notebook["split_clusters"]
    notebook_splits = notebook_to_pipeline["split_clusters"]
    if pipeline_splits:
        print(f"\npipeline clusters split across notebook clusters: {len(pipeline_splits)}")
        for label, counter in sorted(pipeline_splits.items())[:5]:
            print(f"  {label}: {counter}")
    if notebook_splits:
        print(f"\nnotebook clusters split across pipeline clusters: {len(notebook_splits)}")
        for label, counter in sorted(notebook_splits.items())[:5]:
            print(f"  {label}: {counter}")

    exact_match = (
        not only_pipeline
        and not only_notebook
        and not pipeline_splits
        and not notebook_splits
    )
    if exact_match:
        print("\nPartitions match exactly.")
        return 0
    print("\nPartitions differ — see splits / unique-keys above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
