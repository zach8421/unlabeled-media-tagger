#!/usr/bin/env python3
"""Build a shareable review package from pipeline face-clustering output."""

from __future__ import annotations

import argparse
from pathlib import Path

from unlabeled_media_tagger.pipeline.share_package import (
    build_share_package,
    read_metadata,
)

__all__ = ["build_share_package", "read_metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a shareable folder from face_clusters.csv."
    )
    parser.add_argument(
        "--csv",
        default="outputs/pipeline/face_clusters.csv",
        help="Pipeline face_clusters.csv path",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/team_share",
        help="Output folder for the share package",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional JSON file with experiment metadata to include in README/index",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = read_metadata(args.metadata_json)
    summary = build_share_package(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        metadata=metadata,
    )

    print(f"Wrote share package: {summary['out_dir']}")
    print(f"Rows: {summary['rows']}")
    print(f"Clusters: {summary['clusters']}")
    print(f"Contact sheets: {summary['contact_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
