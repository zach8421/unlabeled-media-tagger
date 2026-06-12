"""Share-package helpers for the end-to-end Colab notebook.

Filesystem-only library mirror of the relevant pieces of
``pipeline/share_package.py`` — flatten clustered face records to
pipeline-schema rows, summarize per cluster, pick a representative face, and
copy that face's crop JPEG to the share-package's ``contact_sheets/`` dir.

The pipeline-side ``share_package.crop_face_from_row`` re-opens the original
source frame and re-crops with a 25% margin. The notebook can't do that: it
processes images in a temp directory and deletes them after detection. What
the notebook has is the already-written tight bbox crop on Drive, so this
module's ``copy_crop_to_contact_sheet`` just resizes that JPEG to
``max_dim``. Documented divergence — contact sheets are tight bbox crops vs.
the pipeline's margined crops.

Drive upload is intentionally NOT in this module. The notebook either writes
the share package directly to its Drive mount (no API needed) or uploads via
API in a dedicated cell that uses the same overwrite-by-name helpers as the
local pipeline (``drive/files.py``).

Designed to be copy-pasted into the notebook share cell verbatim per the
project's self-containment rule.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image


FACE_CLUSTERS_COLUMNS = [
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


SHARE_COLUMNS = [
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


SUMMARY_COLUMNS = [
    "cluster_id",
    "cluster_label",
    "face_count",
    "media_file_count",
    "drive_file_count",
    "example_media_names",
    "contact_sheet",
]


CONTACT_SHEET_DIR_NAME = "contact_sheets"
DRIVE_URL_TEMPLATE = "https://drive.google.com/uc?export=view&id={file_id}"


def flatten_clusters_to_face_rows(
    clusters: dict[int, list[dict]],
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
) -> list[dict]:
    """Convert ``cluster_face_records`` output into pipeline-schema face rows.

    Each input face dict is expected to carry the notebook's per-face metadata
    (``source_file_id``, ``source_file_name``, ``face_index``, ``bbox_*``,
    ``confidence``, ``crop_path``) plus the cluster fields added by
    ``cluster_face_records``. Missing fields become empty strings.

    The output column order matches ``pipeline/run.py::CSV_FIELDS`` exactly so
    ``face_clusters.csv`` produced by the notebook is drop-in compatible with
    the local pipeline's downstream consumers (Apps Script, diff-vs-pipeline
    verification).

    Mapping notes:
      - ``drive_id`` <- ``source_file_id`` (the Drive file the face came from)
      - ``media_name`` <- ``source_file_name``
      - ``frame_path`` <- ``crop_path`` (the saved tight crop JPEG; the notebook
        does not keep the source photo on disk, so the crop is the closest
        analogue to the pipeline's frame_path)
      - ``media_path`` left empty; the notebook never persists the downloaded
        source image's path beyond the per-image temp dir
      - ``timestamp_sec`` / ``frame_index`` empty (photo-only flow)
    """
    rows = []
    for cluster_id in sorted(clusters):
        for face in clusters[cluster_id]:
            rows.append(
                {
                    "cluster_id": face.get("cluster_id", cluster_id),
                    "cluster_label": face.get(
                        "cluster_label", f"person_{cluster_id:03d}"
                    ),
                    "drive_id": face.get("source_file_id", ""),
                    "media_name": face.get("source_file_name", ""),
                    "media_path": "",
                    "frame_path": face.get("crop_path", ""),
                    "timestamp_sec": "",
                    "frame_index": "",
                    "face_index": face.get("face_index", ""),
                    "bbox_x": face.get("bbox_x", ""),
                    "bbox_y": face.get("bbox_y", ""),
                    "bbox_w": face.get("bbox_w", ""),
                    "bbox_h": face.get("bbox_h", ""),
                    "confidence": face.get("confidence", ""),
                    "similarity_to_cluster": face.get(
                        "similarity_to_cluster", ""
                    ),
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                }
            )
    return rows


def _parse_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def select_representative_face(cluster_rows: list[dict]) -> Optional[dict]:
    """Pick the most representative face row for a cluster.

    Ordering matches ``pipeline/share_package.select_representative_face``:
      1. ``similarity_to_cluster`` desc (closest to centroid)
      2. ``bbox_w * bbox_h`` desc (larger, typically clearer)
      3. ``frame_index`` asc (deterministic tiebreaker; 0 for photos)
    """
    if not cluster_rows:
        return None

    def sort_key(row: dict):
        similarity = _parse_float(row.get("similarity_to_cluster"), default=0.0)
        area = _parse_int(row.get("bbox_w")) * _parse_int(row.get("bbox_h"))
        frame_index = _parse_int(row.get("frame_index"), default=0)
        return (-similarity, -area, frame_index)

    return min(cluster_rows, key=sort_key)


def group_rows_by_cluster(rows: list[dict]) -> dict[str, list[dict]]:
    """Group face rows by cluster_label, sorted by label."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("cluster_label", "unknown")].append(row)
    return dict(sorted(grouped.items()))


def summarize_clusters(
    grouped: dict[str, list[dict]],
    contact_sheet_dir: str = CONTACT_SHEET_DIR_NAME,
    url_by_label: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Build one summary row per cluster, sorted by face_count desc.

    ``url_by_label`` maps cluster_label → remote URL for that cluster's
    contact sheet. When set, the ``contact_sheet`` column carries the URL;
    otherwise it carries the relative local path
    ``<contact_sheet_dir>/<cluster_label>.jpg``. Matches the pipeline-side
    behavior in ``share_package.summarize_clusters``.
    """
    url_by_label = url_by_label or {}
    summaries = []
    for cluster_label, rows in grouped.items():
        media_names = sorted(
            {row.get("media_name", "") for row in rows if row.get("media_name")}
        )
        drive_ids = sorted(
            {row.get("drive_id", "") for row in rows if row.get("drive_id")}
        )
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
    return sorted(
        summaries,
        key=lambda item: int(item["face_count"]),
        reverse=True,
    )


def copy_crop_to_contact_sheet(
    src_crop_path: str,
    dest_path: Path,
    max_dim: int = 512,
) -> bool:
    """Copy / resize the representative tight-crop JPEG to ``dest_path``.

    Returns True on success, False when the source crop is missing or
    unreadable. Resizes with Lanczos when the longer side exceeds ``max_dim``;
    smaller crops are saved at native resolution (no upscaling). JPEG quality
    matches ``share_package.build_contact_sheet`` (90).
    """
    if not src_crop_path:
        return False
    path = Path(src_crop_path)
    if not path.exists():
        return False
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return False

    longest = max(image.width, image.height)
    if longest > max_dim:
        scale = max_dim / longest
        new_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(new_size, Image.LANCZOS)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path, "JPEG", quality=90)
    return True
