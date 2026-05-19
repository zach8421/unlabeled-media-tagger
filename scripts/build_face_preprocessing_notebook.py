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

CELL_01_TITLE = """# Face Preprocessing — Colab

This notebook detects faces in every image inside a Google Drive folder and
produces reusable preprocessing outputs:

- One JPEG crop per detected face (raw bbox crop, no alignment)
- A per-folder `asset_faces.csv` listing every detected face
- A per-run JSON manifest entry under `{PROJECT_ROOT}/preprocessing_manifest/`
- A unified `preprocessing_manifest.csv` at `{PROJECT_ROOT}/preprocessing_manifest.csv`
  composed from those JSONs

It does **not** compute embeddings, cluster faces, review identities, or write
back to Drive metadata — those steps stay in the main pipeline.

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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm.auto import tqdm
from deepface import DeepFace

logging.getLogger("deepface").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


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

CELL_10_HEADER_DETECT = """## 5. Detect Faces, Save Crops, Write Per-Folder CSV And Per-Run Manifest JSON

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
4. Write the folder's `asset_faces.csv` and a per-run manifest JSON under
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
    print(f"Folders fully up-to-date (no run subfolder created): {len(skipped_folders)}")'''

CELL_12_HEADER_COMPOSE = """## 6. Compose Unified Manifest CSV

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

CELL_14_HEADER_SUMMARY = "## 7. Summary"

CELL_15_SUMMARY = '''total_images_found = sum(e["images_found"] for e in manifest_entries)
total_images_processed = sum(e["images_processed"] for e in manifest_entries)
total_faces_detected = sum(e["faces_detected"] for e in manifest_entries)

print(f"run_id: {run_id}")
print(f"folders with new work: {len(manifest_entries)}")
print(f"folders fully up-to-date (no run subfolder created): {len(skipped_folders)}")
print(f"images found (across folders with new work): {total_images_found}")
print(f"images processed this run: {total_images_processed}")
print(f"faces detected this run: {total_faces_detected}")
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
print(f"Unified manifest: {manifest_csv_path}")'''

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
- **Out of scope for this notebook:** embeddings, clustering, alignment,
  identity review, Drive metadata write-back, video frames. Those happen in
  the local pipeline (or a future end-to-end Colab notebook)."""


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
