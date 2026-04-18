#!/usr/bin/env python3
"""Build a spreadsheet writeback starter package from pipeline artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a writeback starter package from a local pipeline output."
    )
    parser.add_argument(
        "--pipeline-output-dir",
        required=True,
        help="Directory containing face_clusters.csv and review_package/",
    )
    parser.add_argument(
        "--source-media-dir",
        required=True,
        help="Local folder containing the media files used by the sample pipeline run",
    )
    parser.add_argument(
        "--starter-code",
        default="examples/colab_drive_writeback_starter.py",
        help="Colab starter code to copy into the package",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output package directory",
    )
    parser.add_argument(
        "--copy-source-media",
        action="store_true",
        help="Copy sample source media into the package",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def media_type(row: dict) -> str:
    timestamp = row.get("timestamp_sec", "")
    return "video" if timestamp not in ("", None) else "image"


def start_time(row: dict) -> str:
    timestamp = row.get("timestamp_sec", "")
    if timestamp in ("", None):
        return "N/A"
    try:
        seconds = float(timestamp)
    except ValueError:
        return str(timestamp)
    return seconds_to_timestamp(seconds)


def end_time(row: dict) -> str:
    timestamp = row.get("timestamp_sec", "")
    if timestamp in ("", None):
        return "N/A"
    try:
        seconds = float(timestamp) + 1.0
    except ValueError:
        return str(timestamp)
    return seconds_to_timestamp(seconds)


def seconds_to_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_face_id_map(cluster_rows: list[dict]) -> dict[str, str]:
    cluster_labels = sorted(
        {row["cluster_label"] for row in cluster_rows},
        key=lambda label: int(label.split("_")[-1]) if "_" in label else label,
    )
    return {cluster_label: str(index + 1) for index, cluster_label in enumerate(cluster_labels)}


def build_face_library_rows(
    cluster_rows: list[dict],
    summaries: list[dict],
    face_id_map: dict[str, str],
) -> list[dict]:
    summary_by_label = {row["cluster_label"]: row for row in summaries}
    rows = []
    for cluster_label, face_id in face_id_map.items():
        summary = summary_by_label.get(cluster_label, {})
        rows.append(
            {
                "Face ID": face_id,
                "cluster_label": cluster_label,
                "Confirmed Name": "",
                "Notes / Role": "",
                "Confirm Status": "No",
                "Reference Photo (URL)": f"contact_sheets/{cluster_label}.jpg",
                "Total Count": summary.get("face_count", "0"),
            }
        )
    return rows


def build_occurrence_rows(cluster_rows: list[dict], face_id_map: dict[str, str]) -> list[dict]:
    rows = []
    for row in cluster_rows:
        cluster_label = row["cluster_label"]
        rows.append(
            {
                "Face ID": face_id_map[cluster_label],
                "cluster_label": cluster_label,
                "Media Type": media_type(row),
                "File Name": row.get("media_name", ""),
                "Start Time": start_time(row),
                "End Time": end_time(row),
                "Duration": "Static" if media_type(row) == "image" else "00:00:01",
                "Drive Link": "REPLACE_WITH_UPLOADED_DRIVE_LINK",
                "local_file_path": row.get("media_path", ""),
                "local_file_id": row.get("drive_id", ""),
                "Confidence_is_face": row.get("confidence", ""),
                "Confidence_this_person": row.get("similarity_to_cluster", ""),
            }
        )
    return rows


def build_upload_manifest(cluster_rows: list[dict], source_media_dir: Path) -> list[dict]:
    seen = {}
    for row in cluster_rows:
        media_path = row.get("media_path", "")
        if not media_path or media_path in seen:
            continue
        path = Path(media_path)
        try:
            relative_path = str(path.relative_to(source_media_dir))
        except ValueError:
            relative_path = path.name
        seen[media_path] = {
            "local_file_id": row.get("drive_id", ""),
            "file_name": row.get("media_name", ""),
            "relative_path": relative_path,
            "original_local_path": media_path,
            "uploaded_drive_link": "PASTE_UPLOADED_DRIVE_LINK_HERE",
            "uploaded_drive_id": "PASTE_UPLOADED_DRIVE_ID_HERE",
        }
    return sorted(seen.values(), key=lambda item: item["relative_path"])


def copy_review_package(pipeline_output_dir: Path, out_dir: Path) -> None:
    source = pipeline_output_dir / "review_package"
    destination = out_dir / "review_package"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def copy_source_media(source_media_dir: Path, out_dir: Path, manifest_rows: list[dict]) -> None:
    """Copy only source media referenced by the pipeline output."""
    destination = out_dir / "sample_drive_files"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for row in manifest_rows:
        source = Path(row["original_local_path"])
        if not source.exists():
            continue
        try:
            relative_path = source.relative_to(source_media_dir)
        except ValueError:
            relative_path = Path(row["relative_path"])

        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_readme(out_dir: Path, source_media_copied: bool) -> None:
    media_note = (
        "- `sample_drive_files/`: sample source files to upload to Drive.\n"
        if source_media_copied
        else "- Source media was not copied. Use the files under the original `Usability Test` folder.\n"
    )
    readme = f"""# Writeback Starter Package From Usability Test Pipeline Output

This package is built from the actual local pipeline run over the `Usability Test`
sample files. It is intended for developing the Google Sheets + Colab writeback
workflow.

## Contents

- `colab_drive_writeback_starter.py`: starter script for Colab.
- `pipeline_face_clusters.csv`: direct pipeline output with local paths and cluster labels.
- `face_library_user_sheet.csv`: user-facing Face Library worksheet seed.
- `face_occurrences_source_sheet.csv`: source occurrence worksheet seed used by the writeback script.
- `drive_upload_manifest.csv`: list of sample files to upload and where to paste resulting Drive links/IDs.
- `review_package/`: contact sheets, review index, and summary CSV from the pipeline output.
{media_note}

## Suggested Google Sheet Tabs

Create a Google Sheet with two tabs:

```text
Face Library
Face Occurrences
```

Import:

```text
face_library_user_sheet.csv -> Face Library
face_occurrences_source_sheet.csv -> Face Occurrences
```

## Important Setup Step: Replace Drive Links

The occurrence sheet contains placeholder Drive links:

```text
REPLACE_WITH_UPLOADED_DRIVE_LINK
```

Upload the sample files to Google Drive, then fill in real links in:

```text
drive_upload_manifest.csv
face_occurrences_source_sheet.csv
```

The starter script can parse normal Drive links like:

```text
https://drive.google.com/file/d/FILE_ID/view
```

## Human Review Flow

The user should edit only the Face Library tab:

```text
Confirmed Name
Notes / Role
Confirm Status
```

Rows are written back only when:

```text
Confirmed Name is not blank
Confirm Status is Yes / approved / confirmed
```

## Colab Flow

In Colab:

```python
!pip install gspread google-api-python-client google-auth

from google.colab import auth
auth.authenticate_user()

import google.auth
creds, _ = google.auth.default()
```

Paste or import `colab_drive_writeback_starter.py`, then set:

```python
SPREADSHEET_ID = "your spreadsheet id"
FACE_LIBRARY_WORKSHEET = "Face Library"
FACE_OCCURRENCES_WORKSHEET = "Face Occurrences"
DRY_RUN = True
```

Run:

```python
main(creds)
```

Review the dry-run output. Set `DRY_RUN = False` only after the planned file
updates look correct.

## Description Writeback Format

The script preserves existing human-written Drive descriptions and replaces only:

```text
<unlabeled-media-tagger>{{...json...}}</unlabeled-media-tagger>
```

That block includes confirmed names, Face IDs, notes, and occurrence mentions.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> int:
    args = parse_args()
    pipeline_output_dir = Path(args.pipeline_output_dir)
    source_media_dir = Path(args.source_media_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows = read_csv(pipeline_output_dir / "face_clusters.csv")
    summaries = read_csv(pipeline_output_dir / "review_package" / "face_clusters_summary.csv")
    face_id_map = build_face_id_map(cluster_rows)

    shutil.copy2(args.starter_code, out_dir / "colab_drive_writeback_starter.py")
    shutil.copy2(pipeline_output_dir / "face_clusters.csv", out_dir / "pipeline_face_clusters.csv")
    copy_review_package(pipeline_output_dir, out_dir)

    face_library_rows = build_face_library_rows(cluster_rows, summaries, face_id_map)
    occurrence_rows = build_occurrence_rows(cluster_rows, face_id_map)
    upload_manifest_rows = build_upload_manifest(cluster_rows, source_media_dir)

    write_csv(
        out_dir / "face_library_user_sheet.csv",
        [
            "Face ID",
            "cluster_label",
            "Confirmed Name",
            "Notes / Role",
            "Confirm Status",
            "Reference Photo (URL)",
            "Total Count",
        ],
        face_library_rows,
    )
    write_csv(
        out_dir / "face_occurrences_source_sheet.csv",
        [
            "Face ID",
            "cluster_label",
            "Media Type",
            "File Name",
            "Start Time",
            "End Time",
            "Duration",
            "Drive Link",
            "local_file_path",
            "local_file_id",
            "Confidence_is_face",
            "Confidence_this_person",
        ],
        occurrence_rows,
    )
    write_csv(
        out_dir / "drive_upload_manifest.csv",
        [
            "local_file_id",
            "file_name",
            "relative_path",
            "original_local_path",
            "uploaded_drive_link",
            "uploaded_drive_id",
        ],
        upload_manifest_rows,
    )

    if args.copy_source_media:
        copy_source_media(source_media_dir, out_dir, upload_manifest_rows)

    write_readme(out_dir, source_media_copied=args.copy_source_media)

    print(f"Wrote starter package: {out_dir}")
    print(f"Pipeline rows: {len(cluster_rows)}")
    print(f"Face library rows: {len(face_library_rows)}")
    print(f"Occurrence rows: {len(occurrence_rows)}")
    print(f"Upload manifest rows: {len(upload_manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
