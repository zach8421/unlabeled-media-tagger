"""End-to-end Google Drive media tagging pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from unlabeled_media_tagger.drive.auth import get_drive_service
from unlabeled_media_tagger.drive.files import get_file
from unlabeled_media_tagger.pipeline.compare import CompareStage
from unlabeled_media_tagger.pipeline.detect import DetectStage
from unlabeled_media_tagger.pipeline.enrich import EnrichStage, build_drive_metadata
from unlabeled_media_tagger.pipeline.extract import ExtractStage
from unlabeled_media_tagger.pipeline.fetch import FetchStage


CSV_FIELDS = [
    "cluster_id",
    "cluster_label",
    "drive_id",
    "media_name",
    "media_path",
    "frame_path",
    "timestamp_sec",
    "frame_index",
    "face_index",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "confidence",
    "similarity_to_cluster",
    "model_name",
    "detector_backend",
]


class MediaTaggingPipeline:
    """Coordinate Drive fetch, face embedding, clustering, CSV output, and writeback."""

    def __init__(self, config: Optional[dict] = None, service=None):
        self.config = config or {}
        self.service = service

    def get_service(self):
        """Return an authenticated Google Drive service."""
        if self.service is not None:
            return self.service

        drive_config = self.config.get("drive", {})
        self.service = get_drive_service(
            credentials_path=drive_config.get(
                "credentials_path", "secrets/credentials.json"
            ),
            token_path=drive_config.get("token_path", "secrets/token.json"),
            verbose=bool(self.config.get("verbose", False)),
        )
        return self.service

    def run(
        self,
        drive_location: str,
        output_dir: str = "outputs/pipeline",
        limit: Optional[int] = None,
        write_drive_descriptions: bool = False,
        recursive: bool = False,
    ) -> dict:
        """
        Run the full media tagging pipeline.

        Args:
            drive_location: Google Drive folder URL or folder ID.
            output_dir: Local directory for downloads, frames, and CSV output.
            limit: Optional maximum number of Drive media files to process.
            write_drive_descriptions: Whether to update Drive file descriptions.
            recursive: Whether to walk nested Drive folders.

        Returns:
            Summary dictionary with output paths and counts.
        """
        output_path = Path(output_dir)
        downloads_dir = output_path / "downloads"
        frames_dir = output_path / "frames"
        cache_dir = output_path / "processed"
        csv_path = output_path / "face_clusters.csv"
        output_path.mkdir(parents=True, exist_ok=True)
        verbose = bool(self.config.get("verbose", False))
        use_processing_cache = bool(self.config.get("use_processing_cache", True))

        if verbose:
            print("Starting unlabeled-media-tagger pipeline.", flush=True)
            print(f"Output directory: {output_path}", flush=True)
            print("Authenticating with Google Drive...", flush=True)
        service = self.get_service()
        fetch_stage = FetchStage(
            {
                **self.config.get("fetch", {}),
                "download_dir": str(downloads_dir),
                "recursive": recursive,
                "verbose": verbose,
            },
            service=service,
        )
        extract_stage = ExtractStage(
            {
                **self.config.get("extract", {}),
                "frame_dir": str(frames_dir),
            }
        )
        detect_stage = DetectStage(self.config.get("detect", {}))
        compare_stage = CompareStage(self.config.get("compare", {}))

        media_records = fetch_stage.fetch(drive_location, limit=limit)
        face_records = []
        for index, media_record in enumerate(media_records, start=1):
            cache_path = cache_dir / f"{media_record['drive_id']}.json"
            cached_faces = read_processed_cache(cache_path) if use_processing_cache else None
            if cached_faces is not None:
                if verbose:
                    print(
                        f"[{index}/{len(media_records)}] Using cached processing: "
                        f"{media_record['name']} ({len(cached_faces)} face(s))",
                        flush=True,
                    )
                face_records.extend(cached_faces)
                continue

            if verbose:
                print(
                    f"[{index}/{len(media_records)}] Extracting: "
                    f"{media_record['name']}",
                    flush=True,
                )
            extracted = extract_stage.extract(media_record["local_path"])
            if verbose:
                print(
                    f"[{index}/{len(media_records)}] Processing "
                    f"{len(extracted['frames'])} frame/input(s): "
                    f"{media_record['name']}",
                    flush=True,
                )
            current_media_faces = detect_stage.process_extracted_media(
                extracted,
                media_record,
            )
            face_records.extend(current_media_faces)
            if use_processing_cache:
                write_processed_cache(cache_path, current_media_faces)
            if verbose:
                print(
                    f"[{index}/{len(media_records)}] Total embedded faces so far: "
                    f"{len(face_records)}",
                    flush=True,
                )

        if verbose:
            print(f"Clustering {len(face_records)} embedded face(s)...", flush=True)
        clusters = compare_stage.compare_faces(face_records)
        face_rows = flatten_clusters(clusters)
        write_cluster_csv(csv_path, face_rows)
        if verbose:
            print(f"Wrote cluster CSV: {csv_path}", flush=True)

        writeback_count = 0
        if write_drive_descriptions:
            if verbose:
                print("Writing managed metadata to Google Drive descriptions...", flush=True)
            enrich_stage = EnrichStage(self.config.get("enrich", {}), service=service)
            rows_by_file = group_rows_by_drive_file(face_rows)
            for index, media_record in enumerate(media_records, start=1):
                drive_id = media_record["drive_id"]
                rows = rows_by_file.get(drive_id, [])
                metadata = build_drive_metadata(rows)
                if verbose:
                    print(
                        f"[{index}/{len(media_records)}] Updating Drive description: "
                        f"{media_record['name']}",
                        flush=True,
                    )
                enrich_stage.enrich_drive(
                    drive_id,
                    metadata,
                    existing_description=media_record.get("description", ""),
                )
                writeback_count += 1

        return {
            "media_count": len(media_records),
            "face_count": len(face_rows),
            "cluster_count": len(clusters),
            "writeback_count": writeback_count,
            "csv_path": str(csv_path),
            "output_dir": str(output_path),
        }

    def writeback_from_csv(self, csv_path: str) -> dict:
        """
        Write managed Drive descriptions using an existing face clustering CSV.

        This updates only Drive files that have at least one row in the CSV.
        """
        verbose = bool(self.config.get("verbose", False))
        if verbose:
            print(f"Loading cluster CSV: {csv_path}", flush=True)

        rows = read_cluster_csv(Path(csv_path))
        rows_by_file = group_rows_by_drive_file(rows)
        service = self.get_service()
        enrich_stage = EnrichStage(self.config.get("enrich", {}), service=service)

        writeback_count = 0
        for index, (drive_id, file_rows) in enumerate(rows_by_file.items(), start=1):
            file_metadata = get_file(service, drive_id)
            metadata = build_drive_metadata(file_rows)
            if verbose:
                print(
                    f"[{index}/{len(rows_by_file)}] Updating Drive description: "
                    f"{file_metadata.get('name', drive_id)}",
                    flush=True,
                )
            enrich_stage.enrich_drive(
                drive_id,
                metadata,
                existing_description=file_metadata.get("description", ""),
            )
            writeback_count += 1

        return {
            "csv_path": str(csv_path),
            "writeback_count": writeback_count,
        }


def flatten_clusters(clusters: dict[int, list[dict]]) -> list[dict]:
    """Flatten clustered face records into CSV-ready row dictionaries."""
    rows = []
    for cluster_id in sorted(clusters):
        for face in clusters[cluster_id]:
            bbox = face.get("bbox_clipped") or face.get("bbox") or {}
            rows.append(
                {
                    "cluster_id": face.get("cluster_id", cluster_id),
                    "cluster_label": face.get("cluster_label", f"person_{cluster_id:03d}"),
                    "drive_id": face.get("drive_id"),
                    "media_name": face.get("media_name"),
                    "media_path": face.get("media_path"),
                    "frame_path": face.get("frame_path"),
                    "timestamp_sec": face.get("timestamp_sec"),
                    "frame_index": face.get("frame_index"),
                    "face_index": face.get("face_index"),
                    "bbox_x": bbox.get("x"),
                    "bbox_y": bbox.get("y"),
                    "bbox_w": bbox.get("w"),
                    "bbox_h": bbox.get("h"),
                    "confidence": face.get("confidence"),
                    "similarity_to_cluster": face.get("similarity_to_cluster"),
                    "model_name": face.get("model_name"),
                    "detector_backend": face.get("detector_backend"),
                }
            )

    return rows


def write_cluster_csv(csv_path: Path, rows: list[dict]) -> str:
    """Write face clustering rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


def read_cluster_csv(csv_path: Path) -> list[dict]:
    """Read face clustering rows from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {csv_path}")

    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_processed_cache(cache_path: Path) -> Optional[list[dict]]:
    """Read cached face records for one Drive file, if available."""
    if not cache_path.exists():
        return None

    with cache_path.open() as cache_file:
        payload = json.load(cache_file)

    return payload.get("face_records", [])


def write_processed_cache(cache_path: Path, face_records: list[dict]) -> str:
    """Write cached face records for one Drive file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as cache_file:
        json.dump({"face_records": face_records}, cache_file, default=json_safe_value)

    return str(cache_path)


def json_safe_value(value):
    """Convert numpy scalar-like values to plain Python values for JSON output."""
    if hasattr(value, "item"):
        return value.item()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def group_rows_by_drive_file(rows: list[dict]) -> dict[str, list[dict]]:
    """Group CSV rows by Google Drive file ID."""
    grouped = defaultdict(list)
    for row in rows:
        drive_id = row.get("drive_id")
        if drive_id:
            grouped[drive_id].append(row)

    return dict(grouped)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download Google Drive media, cluster faces, and write a CSV."
    )
    parser.add_argument(
        "drive_location",
        nargs="?",
        help="Google Drive folder URL or folder ID",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pipeline",
        help="Directory for downloads, extracted frames, and face_clusters.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of media files to process",
    )
    parser.add_argument(
        "--write-drive-descriptions",
        action="store_true",
        help="Update Google Drive file descriptions with compact clustering metadata",
    )
    parser.add_argument(
        "--writeback-from-csv",
        help="Skip processing and update Drive descriptions from an existing CSV",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process media in nested Drive folders",
    )
    parser.add_argument(
        "--detector-backend",
        default="retinaface",
        help="DeepFace detector backend",
    )
    parser.add_argument(
        "--model-name",
        default="ArcFace",
        help="DeepFace embedding model name",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.68,
        help="Cosine similarity threshold for assigning faces to an existing cluster",
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
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-processing-cache",
        action="store_true",
        help="Reprocess media files even if cached face records exist",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    pipeline = MediaTaggingPipeline(
        {
            "detect": {
                "detector_backend": args.detector_backend,
                "model_name": args.model_name,
            },
            "compare": {
                "similarity_threshold": args.similarity_threshold,
            },
            "extract": {
                "frame_interval": args.frame_interval,
                "max_frames": args.max_frames,
            },
            "verbose": not args.quiet,
            "use_processing_cache": not args.no_processing_cache,
        }
    )
    if args.writeback_from_csv:
        summary = pipeline.writeback_from_csv(args.writeback_from_csv)
        print(f"Drive descriptions updated: {summary['writeback_count']}")
        print(f"CSV: {summary['csv_path']}")
        return 0

    if not args.drive_location:
        raise SystemExit("drive_location is required unless --writeback-from-csv is used")

    summary = pipeline.run(
        args.drive_location,
        output_dir=args.output_dir,
        limit=args.limit,
        write_drive_descriptions=args.write_drive_descriptions,
        recursive=args.recursive,
    )

    print(f"Processed media files: {summary['media_count']}")
    print(f"Detected embedded faces: {summary['face_count']}")
    print(f"Clusters: {summary['cluster_count']}")
    print(f"Drive descriptions updated: {summary['writeback_count']}")
    print(f"CSV: {summary['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
