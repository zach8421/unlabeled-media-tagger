"""Google Drive file operations module."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

FOLDER_URL_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
OPEN_URL_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def parse_drive_folder_id(location: str) -> str:
    """
    Parse a Google Drive folder ID from a raw ID or common folder URL.

    Args:
        location: Google Drive folder ID or URL.

    Returns:
        Folder ID string.
    """
    folder_match = FOLDER_URL_RE.search(location)
    if folder_match:
        return folder_match.group(1)

    open_match = OPEN_URL_RE.search(location)
    if open_match:
        return open_match.group(1)

    return location.strip()


def list_files_in_folder(
    service,
    folder_id: str,
    page_size: int = 100,
    mime_prefixes: Optional[list[str]] = None,
) -> list[dict]:
    """
    List files in a Google Drive folder.
    
    Args:
        service: Authenticated Google Drive service
        folder_id: The ID of the folder to list files from
        page_size: Maximum number of files to return (default: 20)
        
    Returns:
        list[dict]: List of file metadata dictionaries with keys:
                    id, name, mimeType, modifiedTime, description
    """
    query_parts = [f"'{folder_id}' in parents", "trashed=false"]
    if mime_prefixes:
        mime_query = " or ".join(
            f"mimeType contains '{mime_prefix}'" for mime_prefix in mime_prefixes
        )
        query_parts.append(f"({mime_query})")

    files = []
    page_token = None
    while True:
        results = service.files().list(
            q=" and ".join(query_parts),
            pageSize=page_size,
            pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType,modifiedTime,description,size)",
        ).execute()

        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return files


def download_file(service, file_id: str, destination_path: str) -> str:
    """
    Download a Google Drive file to a local path.

    Args:
        service: Authenticated Google Drive service.
        file_id: Google Drive file ID.
        destination_path: Local output path.

    Returns:
        Local destination path.
    """
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError(
            "Google Drive dependencies are not installed. Install the package "
            "with runtime dependencies or run: pip install -r requirements.txt"
        ) from exc

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    with destination.open("wb") as file_handle:
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return str(destination)


def update_file_description(service, file_id: str, description: str) -> dict:
    """
    Update a file's description in Google Drive.
    
    Args:
        service: Authenticated Google Drive service
        file_id: The ID of the file to update
        description: The new description text
        
    Returns:
        dict: Updated file metadata
    """
    file_metadata = {
        "description": description
    }
    
    updated_file = service.files().update(
        fileId=file_id,
        body=file_metadata,
        fields="id,name,description"
    ).execute()
    
    return updated_file


def get_file(service, file_id: str) -> dict:
    """
    Get a file's metadata from Google Drive.

    Args:
        service: Authenticated Google Drive service
        file_id: The ID of the file to retrieve

    Returns:
        dict: File metadata with keys: id, name, description
    """
    file = service.files().get(
        fileId=file_id,
        fields="id,name,description"
    ).execute()

    return file


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def upload_file(
    service,
    local_path: str,
    parent_folder_id: str,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Upload a local file to a Google Drive folder.

    Args:
        service: Authenticated Google Drive service.
        local_path: Path to the local file to upload.
        parent_folder_id: Drive folder ID to upload into.
        mime_type: MIME type to associate with the uploaded file.

    Returns:
        Drive file ID of the newly created file.
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise ImportError(
            "Google Drive dependencies are not installed. Install the package "
            "with runtime dependencies or run: pip install -r requirements.txt"
        ) from exc

    source = Path(local_path)
    media = MediaFileUpload(str(source), mimetype=mime_type, resumable=False)
    created = service.files().create(
        body={"name": source.name, "parents": [parent_folder_id]},
        media_body=media,
        fields="id",
    ).execute()
    return created["id"]


def upload_file_overwriting_by_name(
    service,
    local_path: str,
    parent_folder_id: str,
    drive_filename: str,
    mime_type: str,
) -> str:
    """
    Upload a local file to a Drive folder, overwriting any existing non-folder
    file with the same name (preserving its file ID).

    Lookup uses an exact ``name`` match against children of
    ``parent_folder_id``, excluding folder-typed entries so a folder named
    like the file is never clobbered. With 0 matches a new file is created;
    with 1 the existing file's content is replaced via ``files.update`` (file
    ID preserved); with 2+ matches the helper raises rather than guess which
    file to overwrite. After upload, the resulting file's ``mimeType`` is
    verified against ``mime_type`` -- mismatch raises (catches Drive
    auto-conversion such as CSV -> Sheets).

    Args:
        service: Authenticated Google Drive service.
        local_path: Path to the local file to upload.
        parent_folder_id: Drive folder ID to upload into / look up under.
        drive_filename: Exact name to match in Drive (also used when creating).
        mime_type: MIME type for the uploaded media (e.g. ``text/csv``).

    Returns:
        Drive file ID of the uploaded file (stable across update-in-place).
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise ImportError(
            "Google Drive dependencies are not installed. Install the package "
            "with runtime dependencies or run: pip install -r requirements.txt"
        ) from exc

    escaped_name = drive_filename.replace("'", "\\'")
    query = (
        f"'{parent_folder_id}' in parents "
        f"and name = '{escaped_name}' "
        f"and mimeType != '{FOLDER_MIME_TYPE}' "
        "and trashed=false"
    )

    matches: list[dict] = []
    page_token = None
    while True:
        results = service.files().list(
            q=query,
            pageSize=100,
            pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType)",
        ).execute()
        matches.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    if len(matches) > 1:
        ids = ", ".join(repr(item["id"]) for item in matches)
        raise RuntimeError(
            f"Found {len(matches)} files named {drive_filename!r} under "
            f"Drive folder {parent_folder_id!r} (ids: {ids}). Refusing to "
            "guess which to overwrite -- clean up the duplicates manually "
            "before re-running."
        )

    media = MediaFileUpload(
        str(Path(local_path)), mimetype=mime_type, resumable=False
    )

    if not matches:
        created = service.files().create(
            body={"name": drive_filename, "parents": [parent_folder_id]},
            media_body=media,
            fields="id",
        ).execute()
        file_id = created["id"]
    else:
        file_id = matches[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
            fields="id",
        ).execute()

    metadata = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType",
    ).execute()
    actual_mime = metadata.get("mimeType")
    if actual_mime != mime_type:
        raise RuntimeError(
            f"Drive returned unexpected mimeType {actual_mime!r} for file "
            f"{file_id!r} (name={metadata.get('name')!r}). Expected "
            f"{mime_type!r}. Drive may have auto-converted the upload, which "
            "would silently break downstream consumers."
        )

    return file_id


def create_folder(service, name: str, parent_folder_id: str) -> str:
    """
    Create a folder inside a parent Drive folder.

    Args:
        service: Authenticated Google Drive service.
        name: Name for the new subfolder.
        parent_folder_id: Drive folder ID that will contain the new folder.

    Returns:
        Drive folder ID of the newly created subfolder.
    """
    created = service.files().create(
        body={
            "name": name,
            "mimeType": FOLDER_MIME_TYPE,
            "parents": [parent_folder_id],
        },
        fields="id",
    ).execute()
    return created["id"]


def list_subfolders(service, parent_folder_id: str, page_size: int = 100) -> list[dict]:
    """
    List immediate subfolders of a Drive folder.

    Args:
        service: Authenticated Google Drive service.
        parent_folder_id: Drive folder ID whose subfolders to list.
        page_size: Page size for the listing call.

    Returns:
        List of dicts with keys ``id`` and ``name`` for each subfolder.
    """
    query = (
        f"'{parent_folder_id}' in parents "
        f"and mimeType='{FOLDER_MIME_TYPE}' "
        "and trashed=false"
    )

    folders: list[dict] = []
    page_token = None
    while True:
        results = service.files().list(
            q=query,
            pageSize=page_size,
            pageToken=page_token,
            fields="nextPageToken,files(id,name)",
        ).execute()
        folders.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return folders


def delete_folder(service, folder_id: str) -> None:
    """
    Permanently delete a Drive folder and its descendants.

    Drive's ``files.delete`` permanently removes the target file/folder; for a
    folder this cascades to all contained items.

    Args:
        service: Authenticated Google Drive service.
        folder_id: Drive folder ID to delete.
    """
    service.files().delete(fileId=folder_id).execute()


def verify_folder_writable(service, folder_id: str) -> None:
    """
    Verify a Drive folder ID resolves to a folder the user can write into.

    Performs a single ``files.get`` call to confirm:
    - The folder ID resolves (raises ``FileNotFoundError`` if not).
    - The ID is actually a folder, not another file type
      (raises ``NotADirectoryError`` otherwise).
    - The authenticated user can add children to it
      (raises ``PermissionError`` otherwise).

    Args:
        service: Authenticated Google Drive service.
        folder_id: Drive folder ID to verify.
    """
    try:
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        raise ImportError(
            "Google Drive dependencies are not installed. Install the package "
            "with runtime dependencies or run: pip install -r requirements.txt"
        ) from exc

    try:
        meta = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,capabilities/canAddChildren",
        ).execute()
    except HttpError as exc:
        raise FileNotFoundError(
            f"Drive folder {folder_id!r} could not be accessed: {exc}"
        ) from exc

    if meta.get("mimeType") != FOLDER_MIME_TYPE:
        raise NotADirectoryError(
            f"Drive ID {folder_id!r} resolves to a non-folder: "
            f"name={meta.get('name', '?')!r} mimeType={meta.get('mimeType', '?')!r}"
        )

    if not meta.get("capabilities", {}).get("canAddChildren"):
        raise PermissionError(
            f"Drive folder {folder_id!r} (name={meta.get('name', '?')!r}) "
            "is not writable by the authenticated user"
        )
