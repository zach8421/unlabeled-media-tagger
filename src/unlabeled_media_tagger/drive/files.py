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
