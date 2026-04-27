"""Build a shareable review package from pipeline face-clustering output."""

from __future__ import annotations

import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from unlabeled_media_tagger.drive.files import (
    create_folder,
    delete_folder,
    list_subfolders,
    upload_file,
    upload_file_overwriting_by_name,
)


CONTACT_SHEET_DIR_NAME = "contact_sheets"
DRIVE_URL_TEMPLATE = "https://drive.google.com/uc?export=view&id={file_id}"
AUTO_SUBFOLDER_PATTERN = re.compile(r"^contact_sheets_\d{4}-\d{2}-\d{2}_\d{6}$")
SHARE_CSV_FILENAMES = ("face_clusters_summary.csv", "face_clusters_share.csv")


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


def summarize_clusters(
    grouped: dict[str, list[dict]],
    contact_sheet_dir: str,
    url_by_label: dict[str, str] | None = None,
) -> list[dict]:
    """Build per-cluster summary rows for the share package.

    ``url_by_label`` maps a cluster label to a remote URL (e.g. a Drive URL)
    for that cluster's contact sheet. When a label has an entry, the
    ``contact_sheet`` column carries the URL; otherwise it falls back to the
    relative local path ``<contact_sheet_dir>/<cluster_label>.jpg``.
    """
    url_by_label = url_by_label or {}
    summaries = []
    for cluster_label, rows in grouped.items():
        media_names = sorted({row.get("media_name", "") for row in rows if row.get("media_name")})
        drive_ids = sorted({row.get("drive_id", "") for row in rows if row.get("drive_id")})
        cluster_id = rows[0].get("cluster_id", "") if rows else ""
        contact_sheet_value = url_by_label.get(
            cluster_label, f"{contact_sheet_dir}/{cluster_label}.jpg"
        )
        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "face_count": len(rows),
                "media_file_count": len(media_names),
                "drive_file_count": len(drive_ids),
                "example_media_names": "; ".join(media_names[:5]),
                "contact_sheet": contact_sheet_value,
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


def select_representative_face(cluster_rows: list[dict]) -> dict | None:
    """Pick the single most representative face row for a cluster.

    Ordering:
      1. ``similarity_to_cluster`` descending (closest to cluster centroid).
      2. Face area ``bbox_w * bbox_h`` descending (larger, typically clearer).
      3. ``frame_index`` ascending (deterministic final tiebreaker).

    Returns ``None`` if ``cluster_rows`` is empty.
    """
    if not cluster_rows:
        return None

    def sort_key(row: dict) -> tuple[float, int, int]:
        similarity = parse_float(row.get("similarity_to_cluster"), default=0.0)
        area = parse_int(row.get("bbox_w")) * parse_int(row.get("bbox_h"))
        frame_index = parse_int(row.get("frame_index"), default=0)
        return (-similarity, -area, frame_index)

    return min(cluster_rows, key=sort_key)


def crop_face_from_row(
    row: dict,
    margin: float = 0.25,
    max_dim: int = 512,
) -> Image.Image | None:
    """Open ``frame_path`` and return a single cropped face.

    The crop covers the bbox plus ``margin`` (fraction of the bbox's larger
    side) on each edge, clamped to image bounds. The natural aspect ratio is
    preserved. If the larger output side exceeds ``max_dim`` it is downscaled
    with Lanczos resampling; smaller crops are saved at native resolution
    (no upscaling).

    Returns ``None`` if the frame is missing/unreadable or the bbox is invalid.
    """
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

    pad = int(round(max(w, h) * margin))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad)
    if right <= left or bottom <= top:
        return None

    crop = image.crop((left, top, right, bottom))

    longest = max(crop.width, crop.height)
    if longest > max_dim:
        scale = max_dim / longest
        new_size = (
            max(1, int(round(crop.width * scale))),
            max(1, int(round(crop.height * scale))),
        )
        crop = crop.resize(new_size, Image.LANCZOS)

    return crop


def build_contact_sheet(rows: list[dict], out_path: Path) -> bool:
    """Write a single representative face crop for a cluster to ``out_path``.

    Returns ``True`` if a JPEG was written, ``False`` if no usable face was
    available (no rows, missing/unreadable frame, or invalid bbox).
    """
    representative = select_representative_face(rows)
    if representative is None:
        return False

    crop = crop_face_from_row(representative)
    if crop is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=90)
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


def write_index(
    out_dir: Path,
    summaries: list[dict],
    contact_sheet_dir: str,
    metadata: dict | None = None,
) -> None:
    """Write ``index.html`` next to the local ``contact_sheet_dir``.

    The HTML always references the local relative paths (so it remains a
    standalone offline companion to ``contact_sheets/``), independent of
    whether the CSV's ``contact_sheet`` column carries a Drive URL.
    """
    rows_html = []
    for summary in summaries:
        local_sheet = f"{contact_sheet_dir}/{summary['cluster_label']}.jpg"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(summary['cluster_label'])}</td>"
            f"<td>{summary['face_count']}</td>"
            f"<td>{summary['media_file_count']}</td>"
            f"<td><a href='{html.escape(local_sheet)}'><img src='{html.escape(local_sheet)}' alt='{html.escape(summary['cluster_label'])}'></a></td>"
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


def generate_auto_subfolder_name(now: datetime | None = None) -> str:
    """Return an auto-timestamped subfolder name in UTC.

    The name matches ``AUTO_SUBFOLDER_PATTERN`` exactly; the cleanup logic
    relies on that strict shape.
    """
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("contact_sheets_%Y-%m-%d_%H%M%S")


def cleanup_auto_subfolders(
    service,
    parent_folder_id: str,
    verbose: bool = False,
) -> dict:
    """Delete every auto-timestamped subfolder under ``parent_folder_id``.

    Custom-named subfolders (anything not matching ``AUTO_SUBFOLDER_PATTERN``)
    are preserved. If a delete fails partway, the exception is re-raised after
    surfacing what was deleted vs. what remained, so the user can finish
    cleanup manually before rerunning.
    """
    existing = list_subfolders(service, parent_folder_id)
    deleted: list[dict] = []
    preserved: list[dict] = [
        sub for sub in existing if not AUTO_SUBFOLDER_PATTERN.match(sub["name"])
    ]
    targets = [sub for sub in existing if AUTO_SUBFOLDER_PATTERN.match(sub["name"])]

    if verbose:
        print(
            f"Cleanup: {len(targets)} auto-timestamped subfolder(s) to delete, "
            f"{len(preserved)} custom-named subfolder(s) preserved.",
            flush=True,
        )

    for sub in targets:
        try:
            delete_folder(service, sub["id"])
        except Exception as exc:
            remaining = [item for item in targets if item not in deleted]
            print(
                "Cleanup failed partway through. "
                f"Deleted: {[item['name'] for item in deleted]}. "
                f"Not yet deleted: {[item['name'] for item in remaining]}.",
                flush=True,
            )
            raise RuntimeError(
                f"Failed to delete subfolder {sub['name']!r} "
                f"(id={sub['id']!r}) under parent {parent_folder_id!r}: {exc}"
            ) from exc
        deleted.append(sub)
        if verbose:
            print(f"  deleted: {sub['name']} (id={sub['id']})", flush=True)

    return {
        "deleted": deleted,
        "preserved": preserved,
    }


def upload_contact_sheets(
    service,
    subfolder_id: str,
    contact_dir: Path,
    cluster_labels: list[str],
    verbose: bool = False,
) -> dict[str, str]:
    """Upload each cluster's local contact-sheet JPEG and return label→URL.

    Clusters whose local JPEG is missing or whose upload fails are omitted
    from the returned mapping; the caller falls back to the local relative
    path for those (per the slice's failure-mode policy: log + continue).
    """
    url_by_label: dict[str, str] = {}
    for label in cluster_labels:
        local_path = contact_dir / f"{label}.jpg"
        if not local_path.exists():
            if verbose:
                print(
                    f"  skip upload (no local JPEG): {label}",
                    flush=True,
                )
            continue
        try:
            file_id = upload_file(
                service,
                str(local_path),
                subfolder_id,
                mime_type="image/jpeg",
            )
        except Exception as exc:
            print(
                f"  upload failed for {label} ({local_path}): {exc} -- "
                "falling back to local relative path for this cluster.",
                flush=True,
            )
            continue
        url_by_label[label] = DRIVE_URL_TEMPLATE.format(file_id=file_id)
        if verbose:
            print(f"  uploaded: {label} -> {file_id}", flush=True)
    return url_by_label


def upload_share_csvs(
    service,
    parent_folder_id: str,
    share_dir: Path,
    verbose: bool = False,
) -> dict:
    """Upload the two share-package CSVs to the configured Drive folder.

    Each CSV is uploaded with overwrite-by-name semantics so file IDs stay
    stable across runs. A failure on one CSV does not abort the other --
    partial state is surfaced via the ``failed`` list rather than raised, so
    the pipeline doesn't die at the very last step over a single CSV upload
    error.

    ``face_clusters.csv`` (the raw output with local filesystem paths) is
    never uploaded.
    """
    uploaded: list[dict] = []
    failed: list[dict] = []
    for filename in SHARE_CSV_FILENAMES:
        local_path = share_dir / filename
        try:
            file_id = upload_file_overwriting_by_name(
                service,
                str(local_path),
                parent_folder_id,
                drive_filename=filename,
                mime_type="text/csv",
            )
        except Exception as exc:
            print(
                f"  CSV upload failed for {filename}: {exc} -- "
                "local copy is still on disk; upload manually if needed.",
                flush=True,
            )
            failed.append({"name": filename, "error": str(exc)})
            continue
        uploaded.append({"name": filename, "file_id": file_id})
        if verbose:
            url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"  CSV uploaded: {filename} -> {url}", flush=True)
    return {"uploaded": uploaded, "failed": failed}


def build_share_package(
    csv_path: Path,
    out_dir: Path,
    metadata: dict | None = None,
    drive_service=None,
    drive_folder_id: str | None = None,
    subfolder_name: str | None = None,
    cleanup_old_subfolders: bool = False,
    verbose: bool = False,
) -> dict:
    """Build a shareable review package and return summary counts.

    When ``drive_service`` and ``drive_folder_id`` are both provided, the
    pipeline uploads each contact-sheet JPEG to a subfolder of the configured
    Drive folder and writes Drive URLs into ``face_clusters_summary.csv``.
    Otherwise the CSV carries relative paths exactly like slice 2.
    """
    contact_dir = out_dir / CONTACT_SHEET_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    grouped = cluster_rows(rows)
    write_clean_csv(rows, out_dir / "face_clusters_share.csv")

    for cluster_label, cluster_faces in grouped.items():
        build_contact_sheet(
            rows=cluster_faces,
            out_path=contact_dir / f"{cluster_label}.jpg",
        )

    uploads_enabled = drive_service is not None and drive_folder_id is not None
    url_by_label: dict[str, str] = {}
    upload_summary: dict | None = None

    if uploads_enabled:
        if cleanup_old_subfolders:
            cleanup_result = cleanup_auto_subfolders(
                drive_service, drive_folder_id, verbose=verbose
            )
        else:
            cleanup_result = {"deleted": [], "preserved": []}

        resolved_subfolder_name = subfolder_name or generate_auto_subfolder_name()

        existing = list_subfolders(drive_service, drive_folder_id)
        if any(item["name"] == resolved_subfolder_name for item in existing):
            raise FileExistsError(
                f"Subfolder {resolved_subfolder_name!r} already exists under "
                f"Drive folder {drive_folder_id!r}. Pick a different "
                "--contact-sheets-subfolder-name or delete the existing "
                "subfolder."
            )

        if verbose:
            print(
                f"Creating contact-sheet subfolder: {resolved_subfolder_name}",
                flush=True,
            )
        subfolder_id = create_folder(
            drive_service, resolved_subfolder_name, drive_folder_id
        )

        url_by_label = upload_contact_sheets(
            drive_service,
            subfolder_id,
            contact_dir,
            list(grouped.keys()),
            verbose=verbose,
        )

        upload_summary = {
            "subfolder_name": resolved_subfolder_name,
            "subfolder_id": subfolder_id,
            "uploaded": len(url_by_label),
            "expected": len(grouped),
            "fallback_to_local": [
                label for label in grouped.keys() if label not in url_by_label
            ],
            "cleanup_deleted": [item["name"] for item in cleanup_result["deleted"]],
            "cleanup_preserved": [item["name"] for item in cleanup_result["preserved"]],
        }
    elif drive_folder_id is None and drive_service is None:
        if verbose:
            print(
                "Drive uploads not configured -- contact_sheet column will "
                "carry local relative paths.",
                flush=True,
            )

    summaries = summarize_clusters(
        grouped,
        CONTACT_SHEET_DIR_NAME,
        url_by_label=url_by_label,
    )
    write_summary_csv(summaries, out_dir / "face_clusters_summary.csv")

    if uploads_enabled:
        if verbose:
            print(
                "Uploading share-package CSVs (overwrite-by-name)...",
                flush=True,
            )
        upload_summary["csv_upload"] = upload_share_csvs(
            drive_service,
            drive_folder_id,
            out_dir,
            verbose=verbose,
        )
    elif drive_folder_id is None and drive_service is None:
        if verbose:
            print(
                "Drive uploads not configured -- share-package CSVs stay local.",
                flush=True,
            )

    write_readme(out_dir, csv_path, summaries, metadata=metadata)
    write_index(out_dir, summaries, CONTACT_SHEET_DIR_NAME, metadata=metadata)

    result = {
        "out_dir": str(out_dir),
        "rows": len(rows),
        "clusters": len(summaries),
        "contact_dir": str(contact_dir),
    }
    if upload_summary is not None:
        result["upload"] = upload_summary
    return result
