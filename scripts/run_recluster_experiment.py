#!/usr/bin/env python3
"""Run an offline reclustering experiment and build review contact sheets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from build_share_package import build_share_package
from unlabeled_media_tagger.pipeline.recluster import load_face_records, recluster_faces
from unlabeled_media_tagger.pipeline.run import flatten_clusters, write_cluster_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reclustering from cached embeddings and build review artifacts."
    )
    parser.add_argument(
        "--processed-dir",
        default="outputs/pipeline/processed",
        help="Directory containing processed JSON caches",
    )
    parser.add_argument(
        "--experiments-dir",
        default="outputs/recluster_experiments",
        help="Parent directory for experiment output folders",
    )
    parser.add_argument(
        "--name",
        help="Optional experiment name. Defaults to a name based on threshold settings.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.74,
        help="Minimum pair similarity to consider a merge edge",
    )
    parser.add_argument(
        "--merge-min-similarity",
        type=float,
        default=0.62,
        help="Minimum cross-pair similarity allowed when merging clusters",
    )
    parser.add_argument(
        "--merge-mean-similarity",
        type=float,
        default=0.70,
        help="Minimum average cross-cluster similarity required for a merge",
    )
    parser.add_argument(
        "--allow-same-frame-merge",
        action="store_true",
        help="Allow one cluster to contain multiple faces from the same frame",
    )
    parser.add_argument(
        "--max-faces-per-cluster",
        type=int,
        default=48,
        help="Maximum face crops to include in each contact sheet",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Face crop tile size in contact sheets",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of columns in contact sheets",
    )
    return parser.parse_args()


def threshold_token(value: float) -> str:
    """Format a float threshold for use in a folder name."""
    return f"{value:.2f}".replace(".", "p")


def default_experiment_name(args: argparse.Namespace) -> str:
    """Build a readable experiment folder name from settings."""
    same_frame = "sameframe-ok" if args.allow_same_frame_merge else "sameframe-block"
    return (
        f"edge-{threshold_token(args.edge_threshold)}_"
        f"min-{threshold_token(args.merge_min_similarity)}_"
        f"mean-{threshold_token(args.merge_mean_similarity)}_"
        f"{same_frame}"
    )


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    face_count: int,
    cluster_count: int,
) -> dict:
    """Write experiment settings and output metadata."""
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "graph_quality_gated_recluster",
        "inputs": {
            "processed_dir": args.processed_dir,
            "face_records": face_count,
        },
        "thresholds": {
            "edge_threshold": args.edge_threshold,
            "merge_min_similarity": args.merge_min_similarity,
            "merge_mean_similarity": args.merge_mean_similarity,
            "prevent_same_frame_merge": not args.allow_same_frame_merge,
        },
        "review_package": {
            "max_faces_per_cluster": args.max_faces_per_cluster,
            "tile_size": args.tile_size,
            "cols": args.cols,
        },
        "outputs": {
            "clusters": cluster_count,
            "csv": "face_clusters.csv",
            "summary_csv": "face_clusters_summary.csv",
            "index": "index.html",
            "contact_sheets": "contact_sheets/",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    args = parse_args()
    experiment_name = args.name or default_experiment_name(args)
    out_dir = Path(args.experiments_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "face_clusters.csv"
    manifest_path = out_dir / "manifest.json"

    face_records = load_face_records(args.processed_dir)
    clusters = recluster_faces(
        face_records,
        edge_threshold=args.edge_threshold,
        merge_min_similarity=args.merge_min_similarity,
        merge_mean_similarity=args.merge_mean_similarity,
        prevent_same_frame_merge=not args.allow_same_frame_merge,
    )
    rows = flatten_clusters(clusters)
    write_cluster_csv(csv_path, rows)

    manifest = write_manifest(
        manifest_path=manifest_path,
        args=args,
        face_count=len(face_records),
        cluster_count=len(clusters),
    )
    summary = build_share_package(
        csv_path=csv_path,
        out_dir=out_dir,
        max_faces_per_cluster=args.max_faces_per_cluster,
        tile_size=args.tile_size,
        cols=args.cols,
        metadata=manifest,
    )

    print(f"Experiment: {experiment_name}")
    print(f"Output folder: {out_dir}")
    print(f"Face records: {len(face_records)}")
    print(f"Clusters: {len(clusters)}")
    print(f"CSV: {csv_path}")
    print(f"Review index: {out_dir / 'index.html'}")
    print(f"Contact sheets: {summary['contact_dir']}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
