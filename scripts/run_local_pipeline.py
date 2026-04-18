#!/usr/bin/env python3
"""Run the face pipeline on a local folder without Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path

from unlabeled_media_tagger.pipeline.compare import CompareStage
from unlabeled_media_tagger.pipeline.detect import DetectStage
from unlabeled_media_tagger.pipeline.extract import ExtractStage
from unlabeled_media_tagger.pipeline.run import (
    flatten_clusters,
    read_processed_cache,
    write_cluster_csv,
    write_processed_cache,
)
from unlabeled_media_tagger.utils.file_utils import is_image_file, is_video_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local media through detection, embedding, clustering, and CSV output."
    )
    parser.add_argument("--input-dir", required=True, help="Local folder of media files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder for frames, processed caches, and face_clusters.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process media in nested folders",
    )
    parser.add_argument(
        "--detector-backend",
        default="retinaface",
        help="DeepFace detector backend",
    )
    parser.add_argument(
        "--model-name",
        default="ArcFace",
        help="DeepFace embedding model",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.68,
        help="Online centroid clustering threshold",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled video frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames sampled per video",
    )
    parser.add_argument(
        "--no-processing-cache",
        action="store_true",
        help="Reprocess files even if cached face records exist",
    )
    return parser.parse_args()


def find_media_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Find supported local media files."""
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        [
            path
            for path in paths
            if path.is_file() and (is_image_file(str(path)) or is_video_file(str(path)))
        ]
    )


def local_media_record(path: Path, input_dir: Path) -> dict:
    """Build a pipeline-compatible media record for a local file."""
    relative_path = path.relative_to(input_dir)
    local_id = safe_id(relative_path)
    return {
        "drive_id": local_id,
        "name": path.name,
        "mime_type": "image/*" if is_image_file(str(path)) else "video/*",
        "modified_time": None,
        "description": "",
        "local_path": str(path),
        "relative_path": str(relative_path),
    }


def safe_id(path: Path) -> str:
    """Create a stable local ID from a relative path."""
    return str(path).replace("/", "__").replace(" ", "_")


def run_local_pipeline(args: argparse.Namespace) -> dict:
    """Run the local-folder pipeline."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    processed_dir = output_dir / "processed"
    csv_path = output_dir / "face_clusters.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    media_files = find_media_files(input_dir, recursive=args.recursive)
    print(f"Found {len(media_files)} media file(s) in {input_dir}")

    extract_stage = ExtractStage(
        {
            "frame_dir": str(frames_dir),
            "frame_interval": args.frame_interval,
            "max_frames": args.max_frames,
        }
    )
    detect_stage = DetectStage(
        {
            "detector_backend": args.detector_backend,
            "model_name": args.model_name,
        }
    )
    compare_stage = CompareStage(
        {
            "similarity_threshold": args.similarity_threshold,
        }
    )

    face_records = []
    use_cache = not args.no_processing_cache
    for index, path in enumerate(media_files, start=1):
        media_record = local_media_record(path, input_dir)
        cache_path = processed_dir / f"{media_record['drive_id']}.json"
        cached_faces = read_processed_cache(cache_path) if use_cache else None
        if cached_faces is not None:
            print(
                f"[{index}/{len(media_files)}] Using cached processing: "
                f"{media_record['relative_path']} ({len(cached_faces)} face(s))"
            )
            face_records.extend(cached_faces)
            continue

        print(f"[{index}/{len(media_files)}] Extracting: {media_record['relative_path']}")
        extracted = extract_stage.extract(media_record["local_path"])
        print(
            f"[{index}/{len(media_files)}] Processing "
            f"{len(extracted['frames'])} frame/input(s)"
        )
        current_faces = detect_stage.process_extracted_media(extracted, media_record)
        face_records.extend(current_faces)
        if use_cache:
            write_processed_cache(cache_path, current_faces)
        print(
            f"[{index}/{len(media_files)}] Faces in file: {len(current_faces)}; "
            f"total faces: {len(face_records)}"
        )

    print(f"Clustering {len(face_records)} face(s)...")
    clusters = compare_stage.compare_faces(face_records)
    rows = flatten_clusters(clusters)
    write_cluster_csv(csv_path, rows)

    return {
        "media_count": len(media_files),
        "face_count": len(rows),
        "cluster_count": len(clusters),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
    }


def main() -> int:
    args = parse_args()
    summary = run_local_pipeline(args)
    print(f"Processed media files: {summary['media_count']}")
    print(f"Detected embedded faces: {summary['face_count']}")
    print(f"Clusters: {summary['cluster_count']}")
    print(f"CSV: {summary['csv_path']}")
    print(f"Output directory: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
