"""Generator for examples/face_preprocessing_colab.ipynb.

The notebook is the deliverable; this script builds it. Edit the cell-source
constants below and rerun to regenerate. Hand-editing the .ipynb JSON works
in a pinch but loses round-trip safety with this script.

Usage (from repo root):

    python scripts/build_face_preprocessing_notebook.py examples/face_preprocessing_colab.ipynb

Requires `nbformat` (``pip install nbformat``); not in the runtime
requirements file because it's only needed to regenerate the notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

CELL_01_TITLE = """# Face Clustering Pipeline — Colab

This notebook is the end-to-end face-clustering pipeline as a single Colab
notebook. Point it at a Google Drive folder; it detects faces, embeds them
with ArcFace, clusters them, and writes a share package back to Drive.

Outputs per run:

- One JPEG crop per detected face (raw bbox crop, no alignment), per-folder
  `asset_faces.csv` and per-run manifest entries — same v1.1 preprocessing
  artifacts as before.
- `face_clusters.csv` at `{PROJECT_ROOT}/{SHARE_PACKAGE_DIR}/` with one row
  per face and the same column shape as the local pipeline's output.
- `face_clusters_share.csv` and `face_clusters_summary.csv` for downstream
  spreadsheet consumers.
- `contact_sheets/<cluster_label>.jpg` — one representative crop per cluster.
- (optional) Drive upload of contact sheets + summary CSVs to a configured
  Drive folder, with overwrite-by-name semantics for stable file IDs.

**You only need to edit the Configuration cell (Cell 5).** Everything else
runs as-is.

This notebook is Colab-only: it expects `/content/drive` to be mountable."""

CELL_02_HEADER_INSTALL = "## 1. Install Dependencies And Mount Drive"

CELL_03_INSTALL = """!pip install -q deepface tf-keras opencv-python-headless pillow tqdm

from google.colab import drive, auth
import google.auth

drive.mount("/content/drive")
auth.authenticate_user()
creds, _ = google.auth.default()

print("Drive mounted, authenticated.")"""

CELL_04_HEADER_CONFIG = """## 2. Configuration — Edit These Values

### Make your source folder and project root visible to Colab first

When Colab mounts your Drive, it can **only** see content under your
**My Drive**. Folders in a Shared Drive — or folders that have just been
*shared with you* and not added to your My Drive — will not be visible to
the notebook, even though they appear in the Drive web UI's side panel.

If either `SOURCE_FOLDER_URL` or `PROJECT_ROOT` is a Shared Drive folder or
a "Shared with me" folder, do this once per folder:

1. In the Drive web UI, navigate to the folder.
2. Right-click → **Organize** → **Add shortcut**.
3. Pick `My Drive` as the destination and click **Add**.

The shortcut makes the folder visible at `/content/drive/MyDrive/<folder-name>`
without copying any data. One-time setup per folder.

### Paste your URLs

`SOURCE_FOLDER_URL` is the Drive folder you want to process. Paste a URL like
`https://drive.google.com/drive/folders/<FOLDER_ID>`.

`PROJECT_ROOT` is where the central `preprocessing_manifest.csv` and any
fallback crop output live. It must be a path under your mounted Drive."""

CELL_05_CONFIG = '''SOURCE_FOLDER_URL = "..."
PROJECT_ROOT = "/content/drive/MyDrive/unlabeled-media-tagger"
DETECTOR_BACKEND = "retinaface"
RECURSIVE = True

# End-to-end stage configuration. Matches the local pipeline's defaults so
# notebook output is directly comparable to a local-pipeline run on the same
# source folder.
EMBEDDING_MODEL = "ArcFace"
SIMILARITY_THRESHOLD = 0.68

# Where the share package (face_clusters.csv, summary CSV, contact sheets)
# is written on the Drive mount. Single location overwritten each run, so
# downstream consumers (Apps Script) always point at the latest output.
SHARE_PACKAGE_DIR = "share_package"

# Optional. Set to a Drive folder URL/ID to upload contact sheets +
# face_clusters_summary.csv + face_clusters_share.csv there. When set, the
# summary CSV's `contact_sheet` column carries Drive URLs (IMAGE() ready);
# when unset, it carries local relative paths and the share package only
# exists on the Drive mount (still accessible, just no auto-upload step).
CONTACT_SHEETS_DRIVE_FOLDER_URL = ""

# Image extensions in scope for v1. Videos are out of scope.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

# Faces with a clipped bbox side smaller than this are recorded with
# status="skipped_small" and not written as crops. Matches the threshold used
# by the main pipeline's embed_faces stage. Detection-time guard, not the
# quality-stage size filter (see MIN_CROP_DIM below).
MIN_FACE_SIDE = 10

# v1.1 quality gate (applied after the crop is written, in this order):
#   1. MIN_CROP_DIM — crops whose short side is below this get
#      status="filtered_small_crop" regardless of blur score. Defends against
#      backdrop / collage faces whose halftone-print texture would otherwise
#      score as sharp.
#   2. BLUR_THRESHOLD — crops whose Laplacian-variance score is below this
#      get status="filtered_blurry". Only checked once the crop passes the
#      size gate.
# Both defaults were tuned against real Converge debate photography on
# 2026-05-18 (see dev-log). Crops are always written to disk regardless of
# filter status so the decision is auditable and the threshold is tunable.
BLUR_THRESHOLD = 45.0
MIN_CROP_DIM = 100

# v1.1 status enum (asset_faces.csv `status` column):
#   ok                          face detected, crop written, passes quality gates
#   filtered_small_crop         v1.1: crop short side < MIN_CROP_DIM
#   filtered_blurry             v1.1: quality_score < BLUR_THRESHOLD
#   no_faces                    DeepFace returned an empty list
#   skipped_small               detection-time bbox rejected (< MIN_FACE_SIDE)
#   skipped_already_processed   prior run already produced status=ok for this file
#   read_error                  could not read the image bytes
#   detect_error                DeepFace raised
#
# Any code branching on status MUST use an explicit allow-list of acceptable
# statuses, never an equality check against "ok". This lets future quality
# statuses be added additively without breaking existing readers.'''

CELL_06_HEADER_HELPERS = "## 3. Imports And Helpers"

CELL_07_HELPERS = '''import csv
import json
import logging
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import cv2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from PIL import Image
from tqdm.auto import tqdm
from deepface import DeepFace

logging.getLogger("deepface").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_URL_TEMPLATE = "https://drive.google.com/uc?export=view&id={file_id}"
AUTO_SUBFOLDER_PATTERN = re.compile(r"^contact_sheets_\\d{4}-\\d{2}-\\d{2}_\\d{6}$")


_FOLDER_URL_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_OPEN_URL_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def parse_drive_folder_id(location: str) -> str:
    """Parse a Google Drive folder ID from a raw ID or common folder URL.

    Mirrors the helper at src/unlabeled_media_tagger/drive/files.py:13 in the
    main repo. Copied (not imported) to keep this notebook standalone.
    """
    folder_match = _FOLDER_URL_RE.search(location)
    if folder_match:
        return folder_match.group(1)
    open_match = _OPEN_URL_RE.search(location)
    if open_match:
        return open_match.group(1)
    return location.strip()


def iso_utc_now() -> str:
    """Return the current time as an ISO 8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    """Return a run id like 'run_20260508_143022Z' (UTC, second precision)."""
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def is_image_name(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTENSIONS)


def laplacian_variance(bgr) -> float:
    """Variance of the Laplacian on the grayscale crop. Higher = sharper.

    Mirrors src/unlabeled_media_tagger/preprocessing/blur.py:laplacian_variance
    in the main repo. Copied (not imported) to keep this notebook standalone.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_quality(score: float, width: int, height: int,
                     blur_threshold: float = BLUR_THRESHOLD,
                     min_crop_dim: int = MIN_CROP_DIM) -> str:
    """v1.1 filter: returns 'filtered_small_crop' / 'filtered_blurry' / 'ok'.

    Size gate runs first because the Laplacian-variance score is unreliable
    on small crops (halftone backdrop prints score high despite being out of
    focus). Crops too small for the score to be trustworthy get attributed
    to the size filter, not the blur filter.
    """
    if min(width, height) < min_crop_dim:
        return "filtered_small_crop"
    if score < blur_threshold:
        return "filtered_blurry"
    return "ok"


# --- End-to-end pipeline helpers (embedding, clustering, share package) ----
#
# Mirrors src/unlabeled_media_tagger/preprocessing/{embed,cluster,share}.py
# in the main repo. Copied (not imported) to keep this notebook standalone.
# When updating either side, keep them aligned — equivalence with the local
# pipeline depends on the cluster algorithm and embedding params matching.


def embed_face_crop_bgr(bgr_crop, model_name=EMBEDDING_MODEL):
    """ArcFace embedding for one BGR face crop. Returns None on failure.

    Matches pipeline/embed_faces.py::embed_faces_in_image's represent() call
    exactly: detector_backend="skip", enforce_detection=False, RGB input.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    if bgr_crop.shape[0] < 10 or bgr_crop.shape[1] < 10:
        return None
    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    try:
        result = DeepFace.represent(
            img_path=rgb_crop,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False,
        )
    except Exception:
        return None
    if not isinstance(result, list) or len(result) == 0:
        return None
    embedding = result[0].get("embedding")
    if not embedding:
        return None
    return [float(value) for value in embedding]


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same length")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sqrt(sum(float(a) * float(a) for a in left))
    right_norm = sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def update_centroid(current_centroid, new_embedding, existing_count):
    next_count = existing_count + 1
    return [
        ((float(current) * existing_count) + float(new)) / next_count
        for current, new in zip(current_centroid, new_embedding)
    ]


def cluster_face_records(face_records, similarity_threshold=SIMILARITY_THRESHOLD):
    """Online-centroid clustering — pure-function mirror of CompareStage."""
    clustered = defaultdict(list)
    centroids = []
    for face in face_records:
        embedding = face.get("embedding")
        if not embedding:
            continue
        best_cluster = None
        best_similarity = -1.0
        for cluster_id, centroid in enumerate(centroids):
            similarity = cosine_similarity(embedding, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        if best_cluster is None or best_similarity < similarity_threshold:
            cluster_id = len(centroids)
            centroids.append([float(value) for value in embedding])
            assigned_similarity = 1.0
        else:
            cluster_id = best_cluster
            centroids[cluster_id] = update_centroid(
                centroids[cluster_id],
                embedding,
                len(clustered[cluster_id]),
            )
            assigned_similarity = round(best_similarity, 6)
        clustered[cluster_id].append(
            {
                **face,
                "cluster_id": cluster_id,
                "cluster_label": f"person_{cluster_id:03d}",
                "similarity_to_cluster": assigned_similarity,
            }
        )
    return dict(clustered)


FACE_CLUSTERS_COLUMNS = [
    "cluster_id", "cluster_label", "drive_id", "media_name", "media_path",
    "frame_path", "timestamp_sec", "frame_index", "face_index",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "confidence", "similarity_to_cluster", "model_name", "detector_backend",
]
SHARE_COLUMNS = [
    "cluster_id", "cluster_label", "drive_id", "media_name",
    "timestamp_sec", "frame_index", "face_index",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "confidence", "similarity_to_cluster", "model_name", "detector_backend",
]
SUMMARY_COLUMNS = [
    "cluster_id", "cluster_label", "face_count",
    "media_file_count", "drive_file_count",
    "example_media_names", "contact_sheet",
]
CONTACT_SHEET_DIR_NAME = "contact_sheets"


def flatten_clusters_to_face_rows(clusters, model_name=EMBEDDING_MODEL,
                                  detector_backend=DETECTOR_BACKEND):
    """Convert cluster output to pipeline-schema face_clusters.csv rows.

    Mapping notes (notebook → pipeline schema):
      drive_id    <- source_file_id
      media_name  <- source_file_name
      frame_path  <- crop_path (the saved tight crop JPEG; the notebook does
                     not keep the source photo on disk)
      media_path / timestamp_sec / frame_index: empty (photo-only flow)
    """
    rows = []
    for cluster_id in sorted(clusters):
        for face in clusters[cluster_id]:
            rows.append({
                "cluster_id": face.get("cluster_id", cluster_id),
                "cluster_label": face.get("cluster_label", f"person_{cluster_id:03d}"),
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
                "similarity_to_cluster": face.get("similarity_to_cluster", ""),
                "model_name": model_name,
                "detector_backend": detector_backend,
            })
    return rows


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def select_representative_face(cluster_rows):
    if not cluster_rows:
        return None
    def sort_key(row):
        similarity = _parse_float(row.get("similarity_to_cluster"), 0.0)
        area = _parse_int(row.get("bbox_w")) * _parse_int(row.get("bbox_h"))
        frame_index = _parse_int(row.get("frame_index"), 0)
        return (-similarity, -area, frame_index)
    return min(cluster_rows, key=sort_key)


def group_rows_by_cluster(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("cluster_label", "unknown")].append(row)
    return dict(sorted(grouped.items()))


def summarize_clusters(grouped, contact_sheet_dir=CONTACT_SHEET_DIR_NAME,
                       url_by_label=None):
    url_by_label = url_by_label or {}
    summaries = []
    for cluster_label, rows in grouped.items():
        media_names = sorted({r.get("media_name", "") for r in rows if r.get("media_name")})
        drive_ids = sorted({r.get("drive_id", "") for r in rows if r.get("drive_id")})
        cluster_id = rows[0].get("cluster_id", "") if rows else ""
        contact_sheet_value = url_by_label.get(
            cluster_label, f"{contact_sheet_dir}/{cluster_label}.jpg"
        )
        summaries.append({
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "face_count": len(rows),
            "media_file_count": len(media_names),
            "drive_file_count": len(drive_ids),
            "example_media_names": "; ".join(media_names[:5]),
            "contact_sheet": contact_sheet_value,
        })
    return sorted(summaries, key=lambda item: int(item["face_count"]), reverse=True)


def copy_crop_to_contact_sheet(src_crop_path, dest_path, max_dim=512):
    """Resize the representative tight-crop JPEG to dest_path. Returns bool.

    Notebook contact sheets are tight bbox crops (no margin) since source
    photos aren't kept on disk. The pipeline's contact sheets use 25%-margin
    re-crops from the source frame — documented divergence, visual difference
    is small.
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


# --- Drive upload helpers (mirrors src/.../drive/files.py for self-containment) ---


def drive_create_folder(service, name, parent_folder_id):
    created = service.files().create(
        body={"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE,
              "parents": [parent_folder_id]},
        fields="id",
    ).execute()
    return created["id"]


def drive_upload_file(service, local_path, parent_folder_id, mime_type="image/jpeg"):
    source = Path(local_path)
    media = MediaFileUpload(str(source), mimetype=mime_type, resumable=False)
    created = service.files().create(
        body={"name": source.name, "parents": [parent_folder_id]},
        media_body=media,
        fields="id",
    ).execute()
    return created["id"]


def drive_upload_file_overwriting_by_name(service, local_path, parent_folder_id,
                                          drive_filename, mime_type):
    escaped_name = drive_filename.replace("'", "\\\\'")
    query = (f"'{parent_folder_id}' in parents and name = '{escaped_name}' "
             f"and mimeType != '{DRIVE_FOLDER_MIME_TYPE}' and trashed=false")
    matches = []
    page_token = None
    while True:
        results = service.files().list(
            q=query, pageSize=100, pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType)",
        ).execute()
        matches.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    if len(matches) > 1:
        ids = ", ".join(repr(item["id"]) for item in matches)
        raise RuntimeError(
            f"Found {len(matches)} files named {drive_filename!r} under Drive "
            f"folder {parent_folder_id!r} (ids: {ids}). Clean up duplicates."
        )
    media = MediaFileUpload(str(Path(local_path)), mimetype=mime_type, resumable=False)
    if not matches:
        created = service.files().create(
            body={"name": drive_filename, "parents": [parent_folder_id]},
            media_body=media, fields="id",
        ).execute()
        file_id = created["id"]
    else:
        file_id = matches[0]["id"]
        service.files().update(
            fileId=file_id, media_body=media, fields="id",
        ).execute()
    metadata = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    if metadata.get("mimeType") != mime_type:
        raise RuntimeError(
            f"Drive returned mimeType {metadata.get('mimeType')!r} for "
            f"{file_id!r}; expected {mime_type!r}. Drive may have auto-converted."
        )
    return file_id


def drive_list_subfolders(service, parent_folder_id):
    query = (f"'{parent_folder_id}' in parents "
             f"and mimeType='{DRIVE_FOLDER_MIME_TYPE}' and trashed=false")
    folders = []
    page_token = None
    while True:
        results = service.files().list(
            q=query, pageSize=100, pageToken=page_token,
            fields="nextPageToken,files(id,name)",
        ).execute()
        folders.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return folders


def drive_verify_folder_writable(service, folder_id):
    try:
        meta = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,capabilities/canAddChildren",
        ).execute()
    except HttpError as exc:
        raise FileNotFoundError(f"Drive folder {folder_id!r} not accessible: {exc}") from exc
    if meta.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise NotADirectoryError(
            f"Drive ID {folder_id!r} is not a folder "
            f"(name={meta.get('name', '?')!r}, mimeType={meta.get('mimeType', '?')!r})"
        )
    if not meta.get("capabilities", {}).get("canAddChildren"):
        raise PermissionError(
            f"Drive folder {folder_id!r} (name={meta.get('name', '?')!r}) is not writable"
        )


def generate_auto_subfolder_name(now=None):
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("contact_sheets_%Y-%m-%d_%H%M%S")


# Folder name this notebook writes its own outputs into. Used to skip our own
# output tree during recursive discovery, so re-runs don't treat saved crops
# as fresh source images.
NOTEBOOK_OUTPUT_FOLDER_NAME = "face_preprocessing"'''

CELL_08_HEADER_DISCOVER = """## 4. Discover Source Folders And Image Files

Walks the Drive folder you pasted (recursively if `RECURSIVE`), and builds the
list of folders to process plus their image files. Folder paths are
reconstructed from Drive parents so we can later try writing back to the
mounted source folder."""

CELL_09_DISCOVER = '''def build_drive_service(credentials):
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_drive_children(service, folder_id: str) -> list:
    """List all non-trashed children of a Drive folder. Paginated."""
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=200,
            pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType,parents)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def get_drive_folder_metadata(service, folder_id: str) -> dict:
    return service.files().get(
        fileId=folder_id,
        fields="id,name,parents",
        supportsAllDrives=True,
    ).execute()


def reconstruct_folder_path(service, folder_id: str, cache: dict) -> str:
    """Walk parents up to a parent-less ancestor (My Drive root or Shared
    Drive root) and return a slash-joined path that excludes that root.
    Returns "" when the given folder is itself the root.
    """
    if folder_id in cache:
        return cache[folder_id]
    meta = get_drive_folder_metadata(service, folder_id)
    name = meta.get("name", folder_id)
    parents = meta.get("parents") or []
    if not parents:
        cache[folder_id] = ""
        return ""
    parent_path = reconstruct_folder_path(service, parents[0], cache)
    full = f"{parent_path}/{name}" if parent_path else name
    cache[folder_id] = full
    return full


def discover_folders_and_images(service, root_folder_id: str, recursive: bool):
    """Return [{id, path, images: [{id, name, mimeType}, ...]}, ...].

    Subfolders named NOTEBOOK_OUTPUT_FOLDER_NAME are skipped during recursion
    so re-runs don't traverse this notebook's own output tree (which would
    re-detect on saved crops as if they were fresh source images). The root
    folder is always processed; the filter applies only to descendants.
    """
    path_cache = {}
    folders = []
    queue = [root_folder_id]
    while queue:
        folder_id = queue.pop(0)
        folder_path = reconstruct_folder_path(service, folder_id, path_cache)
        children = list_drive_children(service, folder_id)
        images = [
            c for c in children
            if c.get("mimeType", "").startswith("image/")
            and is_image_name(c.get("name", ""))
        ]
        folders.append({"id": folder_id, "path": folder_path, "images": images})
        if recursive:
            for c in children:
                if c.get("mimeType") != "application/vnd.google-apps.folder":
                    continue
                if c.get("name") == NOTEBOOK_OUTPUT_FOLDER_NAME:
                    parent_label = folder_path or "<root>"
                    print(
                        f"WARN: skipping subfolder named '{NOTEBOOK_OUTPUT_FOLDER_NAME}' "
                        f"under {parent_label} — assumed to be a prior run's output tree. "
                        f"Rename it if it's actually source media."
                    )
                    continue
                queue.append(c["id"])
    return folders


drive_service = build_drive_service(creds)
source_folder_id = parse_drive_folder_id(SOURCE_FOLDER_URL)
folders_to_process = discover_folders_and_images(
    drive_service, source_folder_id, RECURSIVE
)

total_images = sum(len(f["images"]) for f in folders_to_process)
print(f"Discovered {len(folders_to_process)} folder(s), {total_images} image file(s) total.")
for f in folders_to_process:
    print(f"  {f['path'] or '<root>'} ({len(f['images'])} images)")'''

CELL_10_HEADER_DETECT = """## 5. Detect Faces, Save Crops, Embed, Write Per-Folder CSV And Per-Run Manifest JSON

Main processing loop. For each source folder:

1. Choose an output location. Try `<source_folder>/face_preprocessing/run_<run_id>/`
   on the mounted Drive first; fall back to
   `{PROJECT_ROOT}/preprocessing_outputs/<source_folder_id>/run_<run_id>/`
   if the source folder is not writable from the mount (Shared Drive, etc.).
   The chosen location is recorded in `output_folder_path`.
2. Scan all existing `face_preprocessing/run_*/asset_faces.csv` files for
   prior `status=ok` rows. Files already processed are skipped this run with
   `status=skipped_already_processed`.
3. Download each image via the Drive API, run `DeepFace.extract_faces`,
   crop each face from the BGR image with `cv2`, write JPEG quality 95.
4. For every crop that passes the v1.1 quality gate (`status="ok"`),
   compute an ArcFace embedding from the in-memory BGR crop. Embeddings live
   in memory for this run and feed the clustering cell; they are not written
   to disk this version (a future revision may add a sidecar JSONL so
   re-clustering can happen without re-embedding).
5. Write the folder's `asset_faces.csv` and a per-run manifest JSON under
   `{PROJECT_ROOT}/preprocessing_manifest/<run_id>__<source_folder_id>.json`.

Per-image errors are caught and recorded in `status` + `error`. One bad image
never aborts the run."""

CELL_11_DETECT = '''ASSET_FACES_COLUMNS = [
    "run_id", "source_folder_url", "source_folder_id", "source_folder_path",
    "source_file_name", "source_file_path", "source_file_id",
    "face_id", "face_index", "crop_file_name", "crop_path",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence",
    "detector_backend", "processed_at", "status", "quality_score", "error",
]

MANIFEST_COLUMNS = [
    "run_id", "source_folder_url", "source_folder_id", "source_folder_path",
    "output_folder_path", "crops_folder_path", "asset_faces_csv_path",
    "detector_backend", "recursive",
    "images_found", "images_processed", "faces_detected",
    "started_at", "completed_at", "status", "notes",
]


def choose_output_location(folder_path: str, folder_id: str, run_id: str):
    """Try to write under the mounted source folder; fall back under PROJECT_ROOT.

    Returns (output_dir, used_fallback, source_folder_local_root).
    `source_folder_local_root` is the mounted source folder path when the
    primary location is used, else None — used to scan prior runs for
    already-processed file ids.
    """
    fallback = Path(PROJECT_ROOT) / "preprocessing_outputs" / folder_id / run_id

    if not folder_path:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, True, None

    primary_root = Path("/content/drive/MyDrive") / folder_path
    if not primary_root.exists():
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, True, None

    try:
        output_dir = primary_root / NOTEBOOK_OUTPUT_FOLDER_NAME / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        return output_dir, False, primary_root
    except Exception:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, True, None


# Statuses that mean "we successfully ran detection on this source file" —
# including v1.1 quality-filtered rows. read_error / detect_error are
# deliberately excluded so transient failures get retried on the next run.
PROCESSED_STATUSES = {
    "ok",
    "filtered_blurry",
    "filtered_small_crop",
    "no_faces",
    "skipped_small",
}


def scan_existing_processed(folder_local_root) -> set:
    """Return source_file_ids that completed detection in a prior run.

    Walks <folder_local_root>/face_preprocessing/run_*/asset_faces.csv and
    collects every source_file_id whose row has any non-error status. Used
    to skip those files on re-runs.
    """
    processed = set()
    if folder_local_root is None or not folder_local_root.exists():
        return processed
    fp_root = folder_local_root / NOTEBOOK_OUTPUT_FOLDER_NAME
    if not fp_root.exists():
        return processed
    for run_dir in fp_root.iterdir():
        csv_path = run_dir / "asset_faces.csv"
        if not csv_path.exists():
            continue
        try:
            with csv_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("status") in PROCESSED_STATUSES:
                        sid = row.get("source_file_id", "")
                        if sid:
                            processed.add(sid)
        except Exception:
            continue
    return processed


def download_drive_file_to(service, file_id: str, dest: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def make_face_id(file_id: str, face_index: int) -> str:
    return f"{file_id}__face{face_index:02d}"


def base_row(run_id, source_folder_id, source_folder_path, image, source_file_path):
    return {
        "run_id": run_id,
        "source_folder_url": SOURCE_FOLDER_URL,
        "source_folder_id": source_folder_id,
        "source_folder_path": source_folder_path,
        "source_file_name": image["name"],
        "source_file_path": source_file_path,
        "source_file_id": image["id"],
        "face_id": "",
        "face_index": "",
        "crop_file_name": "",
        "crop_path": "",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
        "confidence": "",
        "detector_backend": DETECTOR_BACKEND,
        "processed_at": "",
        "status": "",
        "quality_score": "",
        "error": "",
    }


run_id = make_run_id()
print(f"run_id = {run_id}")

manifest_entries = []
skipped_folders = []

# Accumulates face records across every folder processed this run. Only rows
# that pass the v1.1 quality gate (status="ok") AND get a successful ArcFace
# embedding land here — they feed the cluster cell. Embeddings live in memory
# only this run; the per-folder asset_faces.csv schema is unchanged.
all_face_records = []

for folder in folders_to_process:
    folder_id = folder["id"]
    folder_path = folder["path"]
    images = folder["images"]

    # Read-only lookup for prior-run scanning. Doesn't create or probe anything.
    folder_local_root = None
    if folder_path:
        candidate = Path("/content/drive/MyDrive") / folder_path
        if candidate.exists():
            folder_local_root = candidate

    already_processed = scan_existing_processed(folder_local_root)

    # Output dirs are created lazily so that folders where every image was
    # already processed leave no run_<run_id>/ subtree behind, write no CSV,
    # and emit no manifest entry. choose_output_location is deferred until we
    # actually have something new to write.
    output_dir = None
    crops_dir = None
    csv_path = None
    used_fallback = False
    crops_dir_created = False

    rows = []
    images_found = len(images)
    images_processed = 0
    faces_detected = 0
    detect_failures = 0
    folder_started_at = iso_utc_now()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for image in tqdm(images, desc=f"{folder_path or '<root>'}", leave=False):
            file_id = image["id"]
            file_name = image["name"]
            source_file_path = (
                f"{folder_path}/{file_name}" if folder_path else file_name
            )
            row = base_row(run_id, folder_id, folder_path, image, source_file_path)
            row["processed_at"] = iso_utc_now()

            if file_id in already_processed:
                row["status"] = "skipped_already_processed"
                rows.append(row)
                continue

            local_path = tmpdir / file_name
            try:
                download_drive_file_to(drive_service, file_id, local_path)
            except Exception as e:
                err = dict(row)
                err["status"] = "read_error"
                err["error"] = f"download failed: {e}"
                rows.append(err)
                detect_failures += 1
                continue

            try:
                bgr = cv2.imread(str(local_path))
            except Exception as e:
                err = dict(row)
                err["status"] = "read_error"
                err["error"] = f"cv2.imread raised: {e}"
                rows.append(err)
                detect_failures += 1
                continue
            if bgr is None:
                err = dict(row)
                err["status"] = "read_error"
                err["error"] = "cv2.imread returned None"
                rows.append(err)
                detect_failures += 1
                continue

            try:
                detections = DeepFace.extract_faces(
                    img_path=str(local_path),
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=False,
                )
            except Exception as e:
                err = dict(row)
                err["status"] = "detect_error"
                err["error"] = f"{type(e).__name__}: {e}"
                rows.append(err)
                detect_failures += 1
                continue

            if not detections:
                empty = dict(row)
                empty["status"] = "no_faces"
                rows.append(empty)
                images_processed += 1
                continue

            h_img, w_img = bgr.shape[:2]
            for face_index, face in enumerate(detections):
                facial_area = face.get("facial_area", {}) or {}
                x = int(facial_area.get("x", 0))
                y = int(facial_area.get("y", 0))
                bw = int(facial_area.get("w", 0))
                bh = int(facial_area.get("h", 0))
                confidence = float(face.get("confidence", 0.0))

                x1 = max(0, x); y1 = max(0, y)
                x2 = min(w_img, x + bw); y2 = min(h_img, y + bh)

                face_row = dict(row)
                face_row["face_id"] = make_face_id(file_id, face_index)
                face_row["face_index"] = face_index
                face_row["bbox_x"] = x
                face_row["bbox_y"] = y
                face_row["bbox_w"] = bw
                face_row["bbox_h"] = bh
                face_row["confidence"] = confidence

                if (x2 - x1) < MIN_FACE_SIDE or (y2 - y1) < MIN_FACE_SIDE:
                    face_row["status"] = "skipped_small"
                    rows.append(face_row)
                    continue

                crop = bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    face_row["status"] = "skipped_small"
                    face_row["error"] = "empty crop after clipping"
                    rows.append(face_row)
                    continue

                # Lazily create output_dir + crops_dir on first crop write.
                if output_dir is None:
                    output_dir, used_fallback, _ = choose_output_location(
                        folder_path, folder_id, run_id
                    )
                    crops_dir = output_dir / "crops"
                    csv_path = output_dir / "asset_faces.csv"
                if not crops_dir_created:
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    crops_dir_created = True

                stem = Path(file_name).stem
                crop_file_name = f"{stem}__face{face_index:02d}.jpg"
                crop_path_full = crops_dir / crop_file_name
                cv2.imwrite(
                    str(crop_path_full),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                face_row["crop_file_name"] = crop_file_name
                face_row["crop_path"] = str(crop_path_full)

                # v1.1 quality gate: score on the written crop (so a later
                # human re-tuning the threshold can re-score the same JPEGs),
                # then classify. Status is one of:
                # "ok" / "filtered_small_crop" / "filtered_blurry".
                crop_h, crop_w = crop.shape[:2]
                quality_score = laplacian_variance(crop)
                face_row["quality_score"] = f"{quality_score:.4f}"
                face_row["status"] = classify_quality(
                    quality_score, crop_w, crop_h
                )

                # Embed only quality-passing faces. Match pipeline's policy:
                # represent() on the in-memory BGR crop (not the JPEG roundtrip)
                # so the notebook's embeddings are bit-identical to the local
                # pipeline's on the same input.
                if face_row["status"] == "ok":
                    embedding = embed_face_crop_bgr(crop, model_name=EMBEDDING_MODEL)
                    if embedding is not None:
                        face_row["embedding"] = embedding
                        all_face_records.append(face_row)

                rows.append(face_row)
                faces_detected += 1

            images_processed += 1

    # Decide if this folder did any real work. A folder is a no-op only if
    # every row is skipped_already_processed (or there are no rows at all).
    did_work = any(
        r.get("status") not in ("", "skipped_already_processed") for r in rows
    )

    if not did_work:
        skipped_folders.append({
            "folder_path": folder_path,
            "source_folder_id": folder_id,
            "files_already_processed": len(rows),
        })
        continue

    # Did real work but maybe no crops were written (only no_faces / errors).
    # Make sure output_dir exists for the CSV.
    if output_dir is None:
        output_dir, used_fallback, _ = choose_output_location(
            folder_path, folder_id, run_id
        )
        crops_dir = output_dir / "crops"
        csv_path = output_dir / "asset_faces.csv"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ASSET_FACES_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ASSET_FACES_COLUMNS})

    folder_completed_at = iso_utc_now()

    notes_parts = []
    if used_fallback:
        notes_parts.append(
            "used fallback output location (source folder not writable from mount)"
        )
    if detect_failures:
        notes_parts.append(f"{detect_failures} per-image error(s)")

    if detect_failures == 0:
        folder_status = "success"
    elif images_processed == 0:
        folder_status = "failed"
    else:
        folder_status = "partial"

    entry = {
        "run_id": run_id,
        "source_folder_url": SOURCE_FOLDER_URL,
        "source_folder_id": folder_id,
        "source_folder_path": folder_path,
        "output_folder_path": str(output_dir),
        "crops_folder_path": str(crops_dir),
        "asset_faces_csv_path": str(csv_path),
        "detector_backend": DETECTOR_BACKEND,
        "recursive": RECURSIVE,
        "images_found": images_found,
        "images_processed": images_processed,
        "faces_detected": faces_detected,
        "started_at": folder_started_at,
        "completed_at": folder_completed_at,
        "status": folder_status,
        "notes": "; ".join(notes_parts),
    }
    manifest_entries.append(entry)

    # Filename includes source_folder_id so multiple folders within a single
    # recursive run never collide on disk, and concurrent notebook runs
    # (different run_ids) never collide either.
    manifest_dir = Path(PROJECT_ROOT) / "preprocessing_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / f"{run_id}__{folder_id}.json"
    with json_path.open("w") as fh:
        json.dump(entry, fh, indent=2)

print(f"Folders with new work: {len(manifest_entries)}")
if skipped_folders:
    print(f"Folders fully up-to-date (no run subfolder created): {len(skipped_folders)}")
print(f"Face records embedded and ready for clustering: {len(all_face_records)}")'''

CELL_CLUSTER_HEADER = """## 6. Cluster Faces

Runs online-centroid clustering over every face embedded this run. Matches
the local pipeline's `CompareStage` algorithm (same default similarity
threshold, same online-mean centroid update), so notebook clusters and
pipeline clusters on the same input partition the faces equivalently.

Emits `face_clusters.csv` at `{PROJECT_ROOT}/{SHARE_PACKAGE_DIR}/` with the
same column schema as the local pipeline's `face_clusters.csv` — direct
diff against a pipeline run is the verification path.

Re-running this cell after the detect cell rebuilds the clustering from the
in-memory `all_face_records`; re-running it on a fresh kernel does nothing
useful since embeddings are not persisted (yet)."""

CELL_CLUSTER = '''share_package_dir = Path(PROJECT_ROOT) / SHARE_PACKAGE_DIR
share_package_dir.mkdir(parents=True, exist_ok=True)

clusters = cluster_face_records(
    all_face_records, similarity_threshold=SIMILARITY_THRESHOLD
)
face_cluster_rows = flatten_clusters_to_face_rows(
    clusters,
    model_name=EMBEDDING_MODEL,
    detector_backend=DETECTOR_BACKEND,
)

face_clusters_csv_path = share_package_dir / "face_clusters.csv"
with face_clusters_csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FACE_CLUSTERS_COLUMNS)
    writer.writeheader()
    for row in face_cluster_rows:
        writer.writerow({k: row.get(k, "") for k in FACE_CLUSTERS_COLUMNS})

print(f"Faces clustered: {len(face_cluster_rows)}")
print(f"Clusters formed: {len(clusters)}")
print(f"face_clusters.csv: {face_clusters_csv_path}")'''

CELL_SHARE_HEADER = """## 7. Build Share Package

Writes the same share-package artifacts the local pipeline produces:

- `face_clusters_share.csv` — face-level rows without local filesystem paths.
- `face_clusters_summary.csv` — one row per cluster with counts and a
  `contact_sheet` column. The column carries a local relative path here;
  the upload cell (next) overwrites it with Drive URLs when configured.
- `contact_sheets/<cluster_label>.jpg` — the representative tight bbox crop
  for each cluster, resized to ≤512px on the long side.

One known divergence from the pipeline: the pipeline re-crops the
representative face from its source frame with a 25% margin. The notebook
doesn't keep source images after detection, so its contact sheets are tight
bbox crops without that margin. Visual difference is small."""

CELL_SHARE = '''share_columns_present = SHARE_COLUMNS
face_clusters_share_path = share_package_dir / "face_clusters_share.csv"
with face_clusters_share_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=share_columns_present)
    writer.writeheader()
    for row in face_cluster_rows:
        writer.writerow({k: row.get(k, "") for k in share_columns_present})

grouped = group_rows_by_cluster(face_cluster_rows)
contact_sheet_dir_local = share_package_dir / CONTACT_SHEET_DIR_NAME
contact_sheet_dir_local.mkdir(parents=True, exist_ok=True)

representative_paths = {}
contact_sheets_written = 0
contact_sheets_missing = []
for cluster_label, cluster_rows in grouped.items():
    rep = select_representative_face(cluster_rows)
    if rep is None:
        contact_sheets_missing.append(cluster_label)
        continue
    src_crop = rep.get("frame_path", "")
    dest = contact_sheet_dir_local / f"{cluster_label}.jpg"
    ok = copy_crop_to_contact_sheet(src_crop, dest, max_dim=512)
    if ok:
        representative_paths[cluster_label] = dest
        contact_sheets_written += 1
    else:
        contact_sheets_missing.append(cluster_label)

summaries = summarize_clusters(grouped, CONTACT_SHEET_DIR_NAME, url_by_label=None)
face_clusters_summary_path = share_package_dir / "face_clusters_summary.csv"
with face_clusters_summary_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    for row in summaries:
        writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})

print(f"face_clusters_share.csv:   {face_clusters_share_path}")
print(f"face_clusters_summary.csv: {face_clusters_summary_path}")
print(f"contact_sheets/ written:   {contact_sheets_written}/{len(grouped)}")
if contact_sheets_missing:
    print(f"WARN clusters missing contact sheets: {contact_sheets_missing}")'''

CELL_UPLOAD_HEADER = """## 8. (Optional) Upload Share Package To Drive

When `CONTACT_SHEETS_DRIVE_FOLDER_URL` is set in the configuration cell, this
cell:

1. Creates a timestamped subfolder `contact_sheets_YYYY-MM-DD_HHMMSS` under
   the configured Drive folder and uploads each per-cluster contact-sheet
   JPEG into it.
2. Rewrites `face_clusters_summary.csv` so its `contact_sheet` column
   carries Drive URLs (`https://drive.google.com/uc?export=view&id=...`) in
   place of local relative paths — that URL shape is what makes IMAGE()
   formulas in Google Sheets render the image.
3. Uploads `face_clusters_summary.csv` and `face_clusters_share.csv` to the
   configured Drive folder with overwrite-by-name semantics (file IDs stay
   stable across runs, so downstream consumers like the Apps Script don't
   need to re-discover them).

If the URL is left blank the cell is a no-op — the share package still
exists locally under `{PROJECT_ROOT}/{SHARE_PACKAGE_DIR}/`."""

CELL_UPLOAD = '''if CONTACT_SHEETS_DRIVE_FOLDER_URL:
    contact_sheets_folder_id = parse_drive_folder_id(CONTACT_SHEETS_DRIVE_FOLDER_URL)
    drive_verify_folder_writable(drive_service, contact_sheets_folder_id)

    subfolder_name = generate_auto_subfolder_name()
    existing_subs = drive_list_subfolders(drive_service, contact_sheets_folder_id)
    if any(s["name"] == subfolder_name for s in existing_subs):
        raise FileExistsError(
            f"Subfolder {subfolder_name!r} already exists under "
            f"{contact_sheets_folder_id!r}; re-run in a new second."
        )
    subfolder_id = drive_create_folder(drive_service, subfolder_name,
                                       contact_sheets_folder_id)
    print(f"Created Drive subfolder: {subfolder_name} (id={subfolder_id})")

    url_by_label = {}
    uploaded_count = 0
    for cluster_label, local_path in representative_paths.items():
        try:
            file_id = drive_upload_file(
                drive_service, str(local_path), subfolder_id,
                mime_type="image/jpeg",
            )
        except Exception as exc:
            print(f"  upload failed for {cluster_label}: {exc}")
            continue
        url_by_label[cluster_label] = DRIVE_URL_TEMPLATE.format(file_id=file_id)
        uploaded_count += 1
    print(f"Uploaded contact sheets: {uploaded_count}/{len(representative_paths)}")

    # Rewrite face_clusters_summary.csv with Drive URLs in the contact_sheet
    # column. face_clusters_share.csv has no contact_sheet column so it's
    # uploaded as-is.
    summaries_with_urls = summarize_clusters(
        grouped, CONTACT_SHEET_DIR_NAME, url_by_label=url_by_label
    )
    with face_clusters_summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summaries_with_urls:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})

    csv_uploaded = []
    csv_failed = []
    for filename, mime_type in (
        ("face_clusters_summary.csv", "text/csv"),
        ("face_clusters_share.csv", "text/csv"),
    ):
        local_csv = share_package_dir / filename
        try:
            file_id = drive_upload_file_overwriting_by_name(
                drive_service, str(local_csv),
                contact_sheets_folder_id,
                drive_filename=filename,
                mime_type=mime_type,
            )
        except Exception as exc:
            csv_failed.append((filename, exc))
            continue
        csv_uploaded.append((filename, file_id))
        print(f"  CSV uploaded: {filename} -> {file_id}")
    for filename, exc in csv_failed:
        print(f"  CSV upload FAILED: {filename} ({exc})")
    if csv_failed:
        raise RuntimeError(
            f"{len(csv_failed)} share-package CSV upload(s) failed. "
            "Local copies are still on disk; upload manually if needed."
        )
else:
    print("CONTACT_SHEETS_DRIVE_FOLDER_URL is empty -- skipping Drive upload.")
    print(f"Share package available locally at: {share_package_dir}")'''

CELL_12_HEADER_COMPOSE = """## 9. Compose Unified Manifest CSV

Reads every JSON file under `{PROJECT_ROOT}/preprocessing_manifest/` (this run
plus every prior run) and writes the unified CSV at
`{PROJECT_ROOT}/preprocessing_manifest.csv`. Race-free and self-healing — you
can re-run this cell at any time to rebuild the CSV from the JSONs."""

CELL_13_COMPOSE = '''def compose_manifest_csv():
    manifest_dir = Path(PROJECT_ROOT) / "preprocessing_manifest"
    manifest_csv = Path(PROJECT_ROOT) / "preprocessing_manifest.csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for json_file in sorted(manifest_dir.glob("*.json")):
        try:
            with json_file.open() as fh:
                rows.append(json.load(fh))
        except Exception as e:
            print(f"WARN skipping unreadable {json_file.name}: {e}")

    rows.sort(key=lambda r: (r.get("started_at", ""), r.get("source_folder_id", "")))

    with manifest_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})

    return manifest_csv, len(rows)


manifest_csv_path, manifest_row_count = compose_manifest_csv()
print(f"Wrote {manifest_csv_path} with {manifest_row_count} row(s).")'''

CELL_14_HEADER_SUMMARY = "## 10. Summary"

CELL_15_SUMMARY = '''total_images_found = sum(e["images_found"] for e in manifest_entries)
total_images_processed = sum(e["images_processed"] for e in manifest_entries)
total_faces_detected = sum(e["faces_detected"] for e in manifest_entries)

print(f"run_id: {run_id}")
print(f"folders with new work: {len(manifest_entries)}")
print(f"folders fully up-to-date (no run subfolder created): {len(skipped_folders)}")
print(f"images found (across folders with new work): {total_images_found}")
print(f"images processed this run: {total_images_processed}")
print(f"faces detected this run: {total_faces_detected}")
print(f"faces embedded and clustered: {len(face_cluster_rows)}")
print(f"clusters formed: {len(clusters)}")
print()
if manifest_entries:
    print("Per-folder outputs:")
    for e in manifest_entries:
        print(f"  [{e['status']}] {e['source_folder_path'] or '<root>'}")
        print(f"    -> {e['asset_faces_csv_path']}")
    print()
if skipped_folders:
    print("Folders skipped (every file already had a status=ok row in a prior run):")
    for sf in skipped_folders:
        label = sf["folder_path"] or "<root>"
        print(f"  - {label} ({sf['files_already_processed']} files)")
    print()
print(f"Share package:    {share_package_dir}")
print(f"Unified manifest: {manifest_csv_path}")
if CONTACT_SHEETS_DRIVE_FOLDER_URL:
    print(f"Drive upload target: {CONTACT_SHEETS_DRIVE_FOLDER_URL}")
else:
    print("Drive upload: not configured (set CONTACT_SHEETS_DRIVE_FOLDER_URL to enable).")'''

CELL_16_NOTES = """## Notes

- **Re-running this notebook** on the same `SOURCE_FOLDER_URL` is safe and
  cheap. Folders where every image already has a prior-run row with one of
  the "processed" statuses (`ok`, `filtered_blurry`, `filtered_small_crop`,
  `no_faces`, `skipped_small`) are silently skipped — no new
  `face_preprocessing/run_<run_id>/` is created, no CSV is written, and no
  manifest entry is appended. Folders with at least one new file (or with
  prior `read_error` / `detect_error` rows that should be retried) get a
  fresh run subfolder whose `asset_faces.csv` includes
  `skipped_already_processed` rows for the unchanged files alongside fresh
  rows for the new ones. The summary cell prints which folders were fully
  up-to-date.
- **Recursive discovery skips `face_preprocessing/`.** When `RECURSIVE=True`,
  the walker ignores any subfolder named `face_preprocessing` so prior runs'
  saved crops are not re-detected as fresh source images. Don't rename your
  own photo folders to `face_preprocessing` — they will be skipped.
- **Per-image failures** (download errors, decode errors, DeepFace errors)
  are recorded in `asset_faces.csv` with `status=read_error` or
  `status=detect_error` and never abort the run.
- **Output location fallback.** If the source folder is in a Shared Drive
  or shared with you (not in your My Drive), it is not writable from the
  Drive mount. The notebook detects this and writes crops + CSV to
  `{PROJECT_ROOT}/preprocessing_outputs/<source_folder_id>/run_<run_id>/`
  instead. The actual chosen location is always recorded in
  `output_folder_path` in the manifest.
- **`face_id` is deterministic** — `<source_file_id>__face<face_index:02d>` —
  so the same face on the same file gets the same id across re-runs.
- **v1.1 quality gate.** Every written crop is scored with Laplacian
  variance (`quality_score` column) and classified by `classify_quality`:
  short side < `MIN_CROP_DIM` → `filtered_small_crop`; otherwise score <
  `BLUR_THRESHOLD` → `filtered_blurry`; else `ok`. Crops are written to disk
  regardless of status so thresholds can be re-tuned later by re-scoring the
  existing JPEGs — no need to re-run detection. To bypass either gate set
  the corresponding threshold to 0 in the Configuration cell.
- **Status values are an open string.** Don't write downstream code that
  branches on `status == 'ok'`; always use an explicit allow-list. The
  current set is documented in the Configuration cell.
- **End-to-end output is at `{PROJECT_ROOT}/{SHARE_PACKAGE_DIR}/`.** The
  notebook writes the same artifacts as the local pipeline's share package:
  `face_clusters.csv` (raw, with crop paths), `face_clusters_share.csv`
  (cleaned, no local paths), `face_clusters_summary.csv` (one row per
  cluster), and `contact_sheets/<cluster_label>.jpg`. Column shape matches
  `pipeline/run.py::CSV_FIELDS` exactly, so diffing notebook output against
  a local-pipeline run on the same source folder is the verification path.
- **Embeddings live in memory only this run.** The cluster cell consumes the
  in-memory `all_face_records` accumulator built by the detect cell.
  Re-running the cluster cell after a fresh kernel start without re-running
  detection does nothing useful. A future revision may persist embeddings as
  a sidecar JSONL alongside `asset_faces.csv` to enable re-clustering and
  threshold tuning without re-embedding.
- **Contact sheets are tight bbox crops, not margined crops.** The local
  pipeline's contact sheets re-crop from the source frame with a 25% margin.
  The notebook only has the saved tight crop on disk (source photos are
  deleted after detection), so it copies/resizes that JPEG directly.
  Documented divergence; visual difference is small.
- **Drive upload (optional).** Set `CONTACT_SHEETS_DRIVE_FOLDER_URL` in the
  configuration cell to upload contact sheets + summary/share CSVs to a
  separate Drive folder with overwrite-by-name semantics. The summary CSV's
  `contact_sheet` column then carries Drive URLs (IMAGE() ready); when the
  URL is left blank the share package only exists on the Drive mount.
- **Out of scope for this notebook:** video frame extraction, identity
  review / labeling, Drive metadata write-back to source files. Frame
  extraction stays in the local pipeline; the other two are future work."""


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------

def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(CELL_01_TITLE),
        md(CELL_02_HEADER_INSTALL),
        code(CELL_03_INSTALL),
        md(CELL_04_HEADER_CONFIG),
        code(CELL_05_CONFIG),
        md(CELL_06_HEADER_HELPERS),
        code(CELL_07_HELPERS),
        md(CELL_08_HEADER_DISCOVER),
        code(CELL_09_DISCOVER),
        md(CELL_10_HEADER_DETECT),
        code(CELL_11_DETECT),
        md(CELL_CLUSTER_HEADER),
        code(CELL_CLUSTER),
        md(CELL_SHARE_HEADER),
        code(CELL_SHARE),
        md(CELL_UPLOAD_HEADER),
        code(CELL_UPLOAD),
        md(CELL_12_HEADER_COMPOSE),
        code(CELL_13_COMPOSE),
        md(CELL_14_HEADER_SUMMARY),
        code(CELL_15_SUMMARY),
        md(CELL_16_NOTES),
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    out_path = Path(sys.argv[1])
    nb = build()
    nbf.validate(nb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        nbf.write(nb, fh)
    # Round-trip read to confirm clean open.
    with out_path.open() as fh:
        loaded = nbf.read(fh, as_version=4)
    nbf.validate(loaded)
    n_md = sum(1 for c in loaded["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in loaded["cells"] if c["cell_type"] == "code")
    print(f"OK: wrote {out_path} — {len(loaded['cells'])} cells ({n_md} md, {n_code} code).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
