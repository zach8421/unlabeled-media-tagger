"""
Colab starter: write confirmed face names from Google Sheets to Drive descriptions.

Intended workflow:
1. Pipeline writes or refreshes a source Google Sheet with face occurrences.
2. A user-facing Google Sheet lets humans confirm names for Face IDs.
3. This script reads both sheets live and writes confirmed names back to the
   matching Google Drive file descriptions.

This is starter code for a Colab notebook/script. It is intentionally verbose
and conservative:
- DRY_RUN defaults to True.
- Existing human-written Drive descriptions are preserved.
- Only the managed <unlabeled-media-tagger>...</unlabeled-media-tagger> block
  is replaced.
- Unconfirmed or blank names are ignored.

Colab setup:
    !pip install gspread google-api-python-client google-auth

Then run:
    from google.colab import auth
    auth.authenticate_user()

    import google.auth
    creds, _ = google.auth.default()

    main(creds)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import gspread
from googleapiclient.discovery import build


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set this to your Google Spreadsheet ID. It is the long ID in the sheet URL:
# https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
SPREADSHEET_ID = "PASTE_SPREADSHEET_ID_HERE"

# Worksheet names. Rename these to match the team's actual Google Sheets tabs.
FACE_LIBRARY_WORKSHEET = "Face Library"
FACE_OCCURRENCES_WORKSHEET = "Face Occurrences"

# Keep True while developing. Set False only after reviewing dry-run output.
DRY_RUN = True

# Face Library columns expected by this script.
FACE_ID_COLUMN = "Face ID"
CONFIRMED_NAME_COLUMN = "Confirmed Name"
CONFIRM_STATUS_COLUMN = "Confirm Status"
NOTES_COLUMN = "Notes / Role"

# Occurrence sheet columns expected by this script.
DRIVE_LINK_COLUMN = "Drive Link"
FILE_NAME_COLUMN = "File Name"
MEDIA_TYPE_COLUMN = "Media Type"
START_TIME_COLUMN = "Start Time"
END_TIME_COLUMN = "End Time"
CONFIDENCE_PERSON_COLUMN = "Confidence_this_person"

# Only these confirmation values will be treated as approved.
APPROVED_VALUES = {"yes", "y", "true", "approved", "confirmed"}

DESCRIPTION_START = "<unlabeled-media-tagger>"
DESCRIPTION_END = "</unlabeled-media-tagger>"


# ---------------------------------------------------------------------------
# Sheet Reading
# ---------------------------------------------------------------------------

def load_worksheets(creds) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read live Google Sheets worksheets into row dictionaries."""
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    face_library = spreadsheet.worksheet(FACE_LIBRARY_WORKSHEET).get_all_records()
    occurrences = spreadsheet.worksheet(FACE_OCCURRENCES_WORKSHEET).get_all_records()

    return normalize_rows(face_library), normalize_rows(occurrences)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert sheet cell values to stripped strings and drop empty rows."""
    normalized = []
    for row in rows:
        clean_row = {
            strip_bom(str(key)).strip(): str(value).strip()
            for key, value in row.items()
            if str(key).strip()
        }
        if any(clean_row.values()):
            normalized.append(clean_row)

    return normalized


def strip_bom(value: str) -> str:
    """Remove UTF-8 BOM markers that sometimes appear in CSV-derived sheets."""
    return value.lstrip("\ufeff")


# ---------------------------------------------------------------------------
# Mapping Confirmed Names To Drive Files
# ---------------------------------------------------------------------------

def approved_face_names(face_library_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """
    Return approved Face ID -> person info.

    A row is approved when Confirmed Name is non-empty and Confirm Status is in
    APPROVED_VALUES.
    """
    approved = {}
    for row in face_library_rows:
        face_id = row.get(FACE_ID_COLUMN, "").strip()
        name = row.get(CONFIRMED_NAME_COLUMN, "").strip()
        status = row.get(CONFIRM_STATUS_COLUMN, "").strip().lower()
        if not face_id or not name or status not in APPROVED_VALUES:
            continue

        approved[face_id] = {
            "face_id": face_id,
            "name": name,
            "notes": row.get(NOTES_COLUMN, "").strip(),
        }

    return approved


def build_file_updates(
    face_library_rows: list[dict[str, str]],
    occurrence_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """
    Join approved Face IDs to occurrence rows and group updates by Drive file.
    """
    approved = approved_face_names(face_library_rows)
    updates: dict[str, dict[str, Any]] = {}

    for row in occurrence_rows:
        face_id = row.get(FACE_ID_COLUMN, "").strip()
        if face_id not in approved:
            continue

        drive_id = parse_drive_file_id(row.get(DRIVE_LINK_COLUMN, ""))
        if not drive_id:
            continue

        file_update = updates.setdefault(
            drive_id,
            {
                "drive_id": drive_id,
                "file_name": row.get(FILE_NAME_COLUMN, "").strip(),
                "media_type": row.get(MEDIA_TYPE_COLUMN, "").strip(),
                "people": {},
            },
        )

        person = approved[face_id]
        person_update = file_update["people"].setdefault(
            face_id,
            {
                "face_id": face_id,
                "name": person["name"],
                "notes": person.get("notes", ""),
                "mentions": [],
            },
        )
        person_update["mentions"].append(
            {
                "start_time": row.get(START_TIME_COLUMN, "").strip(),
                "end_time": row.get(END_TIME_COLUMN, "").strip(),
                "confidence": parse_optional_float(row.get(CONFIDENCE_PERSON_COLUMN, "")),
            }
        )

    return updates


def parse_drive_file_id(value: str) -> str | None:
    """Parse a Drive file ID from common Drive URL forms or a raw ID."""
    value = value.strip()
    if not value:
        return None

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    # If the cell already contains a raw Drive ID, accept it.
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", value):
        return value

    return None


def parse_optional_float(value: str) -> float | None:
    """Parse a float cell value, returning None for blank/unparseable cells."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Drive Description Writeback
# ---------------------------------------------------------------------------

def build_managed_metadata(file_update: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON payload stored in each Drive file description."""
    people = sorted(
        file_update["people"].values(),
        key=lambda item: (item["name"].lower(), item["face_id"]),
    )
    return {
        "schema": "unlabeled-media-tagger",
        "version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "human_confirmed_spreadsheet",
        "people": people,
    }


def build_description(existing_description: str, metadata: dict[str, Any]) -> str:
    """Preserve human text and replace this tool's managed metadata block."""
    base_description = strip_managed_block(existing_description or "").strip()
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    managed_block = f"{DESCRIPTION_START}{payload}{DESCRIPTION_END}"
    if base_description:
        return f"{base_description}\n\n{managed_block}"

    return managed_block


def strip_managed_block(description: str) -> str:
    """Remove an existing managed metadata block, if present."""
    start = description.find(DESCRIPTION_START)
    end = description.find(DESCRIPTION_END)
    if start == -1 or end == -1 or end < start:
        return description

    end += len(DESCRIPTION_END)
    return f"{description[:start]}{description[end:]}"


def get_drive_description(drive_service, file_id: str) -> str:
    """Fetch the current Drive file description."""
    file_metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="id,name,description")
        .execute()
    )
    return file_metadata.get("description", "")


def update_drive_description(drive_service, file_id: str, description: str) -> dict:
    """Update a Drive file description."""
    return (
        drive_service.files()
        .update(
            fileId=file_id,
            body={"description": description},
            fields="id,name,description",
        )
        .execute()
    )


def write_updates_to_drive(creds, file_updates: dict[str, dict[str, Any]]) -> None:
    """Write all grouped file updates to Google Drive descriptions."""
    drive_service = build("drive", "v3", credentials=creds)

    print(f"Prepared updates for {len(file_updates)} Drive file(s).")
    for index, (drive_id, file_update) in enumerate(file_updates.items(), start=1):
        names = [person["name"] for person in file_update["people"].values()]
        print(
            f"[{index}/{len(file_updates)}] "
            f"{file_update.get('file_name') or drive_id}: {', '.join(sorted(names))}"
        )

        if DRY_RUN:
            continue

        existing_description = get_drive_description(drive_service, drive_id)
        metadata = build_managed_metadata(file_update)
        description = build_description(existing_description, metadata)
        update_drive_description(drive_service, drive_id, description)

    if DRY_RUN:
        print("\nDRY_RUN=True, so no Drive files were updated.")
        print("Set DRY_RUN=False after reviewing the planned updates.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(creds) -> None:
    """Run the spreadsheet-to-Drive writeback workflow."""
    face_library_rows, occurrence_rows = load_worksheets(creds)
    file_updates = build_file_updates(face_library_rows, occurrence_rows)
    write_updates_to_drive(creds, file_updates)


if __name__ == "__main__":
    raise SystemExit(
        "Run this starter from Colab after authenticating and passing creds to main(creds)."
    )
