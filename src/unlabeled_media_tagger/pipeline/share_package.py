"""Build a shareable review package from pipeline face-clustering output."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SHARED_COLUMNS = [
    "cluster_id",
    "cluster_label",
    "drive_id",
    "media_name",
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


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_clean_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SHARED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHARED_COLUMNS})


def cluster_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("cluster_label", "unknown")].append(row)
    return dict(sorted(grouped.items()))


def summarize_clusters(grouped: dict[str, list[dict]], contact_sheet_dir: str) -> list[dict]:
    summaries = []
    for cluster_label, rows in grouped.items():
        media_names = sorted({row.get("media_name", "") for row in rows if row.get("media_name")})
        drive_ids = sorted({row.get("drive_id", "") for row in rows if row.get("drive_id")})
        cluster_id = rows[0].get("cluster_id", "") if rows else ""
        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "face_count": len(rows),
                "media_file_count": len(media_names),
                "drive_file_count": len(drive_ids),
                "example_media_names": "; ".join(media_names[:5]),
                "contact_sheet": f"{contact_sheet_dir}/{cluster_label}.jpg",
            }
        )

    return sorted(summaries, key=lambda item: int(item["face_count"]), reverse=True)


def write_summary_csv(summaries: list[dict], out_path: Path) -> None:
    fieldnames = [
        "cluster_id",
        "cluster_label",
        "face_count",
        "media_file_count",
        "drive_file_count",
        "example_media_names",
        "contact_sheet",
    ]
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def pick_representative_rows(rows: list[dict], max_rows: int) -> list[dict]:
    def score(row: dict) -> tuple[float, float]:
        return (
            parse_float(row.get("similarity_to_cluster"), default=0.0),
            parse_float(row.get("confidence"), default=0.0),
        )

    return sorted(rows, key=score, reverse=True)[:max_rows]


def parse_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def crop_face(row: dict, tile_size: int) -> Image.Image | None:
    frame_path = row.get("frame_path", "")
    if not frame_path:
        return None

    path = Path(frame_path)
    if not path.exists():
        return None

    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return None

    x = parse_int(row.get("bbox_x"))
    y = parse_int(row.get("bbox_y"))
    w = parse_int(row.get("bbox_w"))
    h = parse_int(row.get("bbox_h"))
    if w <= 0 or h <= 0:
        return None

    margin = int(max(w, h) * 0.25)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image.width, x + w + margin)
    bottom = min(image.height, y + h + margin)
    if right <= left or bottom <= top:
        return None

    crop = image.crop((left, top, right, bottom))
    crop = ImageOps.pad(crop, (tile_size, tile_size), color=(245, 245, 245))
    return crop


def build_contact_sheet(
    cluster_label: str,
    rows: list[dict],
    out_path: Path,
    max_faces: int,
    tile_size: int,
    cols: int,
) -> bool:
    selected = pick_representative_rows(rows, max_faces)
    tiles = []
    for row in selected:
        crop = crop_face(row, tile_size)
        if crop is not None:
            tiles.append(crop)

    if not tiles:
        return False

    header_height = 48
    gap = 8
    rows_count = math.ceil(len(tiles) / cols)
    width = (cols * tile_size) + ((cols + 1) * gap)
    height = header_height + (rows_count * tile_size) + ((rows_count + 1) * gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (gap, 14),
        f"{cluster_label} | faces: {len(rows)} | examples shown: {len(tiles)}",
        fill=(20, 20, 20),
    )

    for index, tile in enumerate(tiles):
        col = index % cols
        row = index // cols
        x = gap + col * (tile_size + gap)
        y = header_height + gap + row * (tile_size + gap)
        sheet.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return True


def read_metadata(metadata_path: str | None) -> dict:
    """Read optional experiment metadata."""
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        return {}

    with path.open() as metadata_file:
        return json.load(metadata_file)


def format_metadata(metadata: dict) -> str:
    """Format experiment metadata for Markdown."""
    if not metadata:
        return ""

    lines = ["## Experiment Settings", ""]
    for key, value in metadata.items():
        if isinstance(value, dict):
            lines.append(f"- {key}:")
            for child_key, child_value in value.items():
                lines.append(f"  - {child_key}: `{child_value}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_readme(
    out_dir: Path,
    csv_path: Path,
    summaries: list[dict],
    metadata: dict | None = None,
) -> None:
    total_faces = sum(int(summary["face_count"]) for summary in summaries)
    metadata_section = format_metadata(metadata or {})
    readme = f"""# Face Clustering Review Package

Generated: {datetime.now().isoformat(timespec="seconds")}

This folder contains shareable review artifacts from `unlabeled-media-tagger`.

## Contents

- `face_clusters_share.csv`: cleaned face-level results without local media/frame paths.
- `face_clusters_summary.csv`: one row per cluster with counts and contact-sheet paths.
- `contact_sheets/`: representative cropped face contact sheets for each cluster.
- `index.html`: browser-friendly summary view.

## Run Summary

- Source CSV: `{csv_path}`
- Total detected face rows: {total_faces}
- Total clusters: {len(summaries)}

{metadata_section}
## Notes

- Cluster labels such as `person_000` are anonymous machine-generated IDs.
- Clusters are not verified identities.
- Some clusters may contain false positives or split the same person across multiple clusters.
- This package intentionally excludes raw downloaded media, extracted frame folders, OAuth credentials, tokens, and processed embedding JSON files.
"""
    (out_dir / "README_results.md").write_text(readme)


def write_index(out_dir: Path, summaries: list[dict], metadata: dict | None = None) -> None:
    rows_html = []
    for summary in summaries:
        sheet = summary["contact_sheet"]
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(summary['cluster_label'])}</td>"
            f"<td>{summary['face_count']}</td>"
            f"<td>{summary['media_file_count']}</td>"
            f"<td><a href='{html.escape(sheet)}'><img src='{html.escape(sheet)}' alt='{html.escape(summary['cluster_label'])}'></a></td>"
            f"<td>{html.escape(summary['example_media_names'])}</td>"
            "</tr>"
        )

    metadata_html = ""
    if metadata:
        metadata_rows = []
        for key, value in metadata.items():
            metadata_rows.append(
                f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(json.dumps(value, indent=2))}</pre></td></tr>"
            )
        metadata_html = f"""
  <h2>Experiment Settings</h2>
  <table>
    <tbody>{''.join(metadata_rows)}</tbody>
  </table>
"""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Cluster Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; text-align: left; }}
    img {{ max-width: 360px; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Face Cluster Review</h1>
  <p>Machine-generated clusters for review. Cluster labels are anonymous IDs, not verified names.</p>
  {metadata_html}
  <table>
    <thead>
      <tr>
        <th>Cluster</th>
        <th>Faces</th>
        <th>Media Files</th>
        <th>Contact Sheet</th>
        <th>Example Media Names</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page)


def build_share_package(
    csv_path: Path,
    out_dir: Path,
    max_faces_per_cluster: int = 48,
    tile_size: int = 128,
    cols: int = 8,
    metadata: dict | None = None,
) -> dict:
    """Build a shareable review package and return summary counts."""
    contact_dir = out_dir / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    grouped = cluster_rows(rows)
    write_clean_csv(rows, out_dir / "face_clusters_share.csv")

    for cluster_label, cluster_faces in grouped.items():
        build_contact_sheet(
            cluster_label=cluster_label,
            rows=cluster_faces,
            out_path=contact_dir / f"{cluster_label}.jpg",
            max_faces=max_faces_per_cluster,
            tile_size=tile_size,
            cols=cols,
        )

    summaries = summarize_clusters(grouped, "contact_sheets")
    write_summary_csv(summaries, out_dir / "face_clusters_summary.csv")
    write_readme(out_dir, csv_path, summaries, metadata=metadata)
    write_index(out_dir, summaries, metadata=metadata)

    return {
        "out_dir": str(out_dir),
        "rows": len(rows),
        "clusters": len(summaries),
        "contact_dir": str(contact_dir),
    }
