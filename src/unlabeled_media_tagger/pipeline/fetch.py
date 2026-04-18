"""Fetch Stage - media retrieval from Google Drive."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from unlabeled_media_tagger.drive.auth import get_drive_service
from unlabeled_media_tagger.drive.files import (
    download_file,
    list_files_in_folder,
    parse_drive_folder_id,
)


MEDIA_MIME_PREFIXES = ["image/", "video/"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class FetchStage:
    """
    Fetch stage for retrieving media files from Google Drive.
    
    This stage is responsible for:
    - Authenticating with Google Drive API
    - Querying for media files based on criteria
    - Downloading media files for processing
    - Managing local cache of downloaded media
    """
    
    def __init__(self, config=None, service=None):
        """
        Initialize the fetch stage.
        
        Args:
            config: Configuration dictionary for Google Drive API settings
        """
        self.config = config or {}
        self.service = service

    def get_service(self):
        """Return the configured Drive service, creating one if needed."""
        if self.service is not None:
            return self.service

        credentials_path = self.config.get("credentials_path", "secrets/credentials.json")
        token_path = self.config.get("token_path", "secrets/token.json")
        self.service = get_drive_service(
            credentials_path=credentials_path,
            token_path=token_path,
            verbose=bool(self.config.get("verbose", False)),
        )
        return self.service
    
    def fetch(
        self,
        location: Optional[str] = None,
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch media files from Google Drive.
        
        Args:
            location: Google Drive folder ID or URL. Falls back to config["folder_id"].
            output_dir: Local download directory. Falls back to config["download_dir"].
            limit: Optional maximum number of media files to download.
            
        Returns:
            List of downloaded media records with Drive metadata and local paths.
        """
        folder_location = location or self.config.get("folder_id")
        if not folder_location:
            raise ValueError("Google Drive folder location is required")

        folder_id = parse_drive_folder_id(folder_location)
        download_dir = Path(output_dir or self.config.get("download_dir", "outputs/cache"))
        download_dir.mkdir(parents=True, exist_ok=True)
        verbose = bool(self.config.get("verbose", False))

        service = self.get_service()
        if verbose:
            scope = "recursively" if self.config.get("recursive", False) else "directly"
            print(f"Listing media files {scope} in Drive folder: {folder_id}", flush=True)
        files = self._list_media_files(service, folder_id)
        if limit is not None:
            files = files[:limit]
        if verbose:
            print(f"Found {len(files)} media file(s) to process.", flush=True)

        records = []
        for index, file_metadata in enumerate(files, start=1):
            local_path = download_dir / file_metadata["id"] / file_metadata["name"]
            if local_path.exists() and local_path.stat().st_size > 0:
                if verbose:
                    print(
                        f"[{index}/{len(files)}] Using cached download: "
                        f"{file_metadata['name']}",
                        flush=True,
                    )
            else:
                if verbose:
                    print(
                        f"[{index}/{len(files)}] Downloading: {file_metadata['name']}",
                        flush=True,
                    )
                download_file(service, file_metadata["id"], str(local_path))
            records.append(
                {
                    "drive_id": file_metadata["id"],
                    "name": file_metadata["name"],
                    "mime_type": file_metadata.get("mimeType", ""),
                    "modified_time": file_metadata.get("modifiedTime"),
                    "description": file_metadata.get("description", ""),
                    "local_path": str(local_path),
                }
            )

        return records

    def _list_media_files(self, service, folder_id: str) -> list[dict]:
        """List media files, optionally walking nested Drive folders."""
        page_size = int(self.config.get("page_size", 100))
        if not self.config.get("recursive", False):
            return list_files_in_folder(
                service,
                folder_id,
                page_size=page_size,
                mime_prefixes=MEDIA_MIME_PREFIXES,
            )

        media_files = []
        folders_to_visit = [folder_id]
        while folders_to_visit:
            current_folder_id = folders_to_visit.pop(0)
            children = list_files_in_folder(
                service,
                current_folder_id,
                page_size=page_size,
            )
            for child in children:
                mime_type = child.get("mimeType", "")
                if mime_type == FOLDER_MIME_TYPE:
                    folders_to_visit.append(child["id"])
                elif any(mime_type.startswith(prefix) for prefix in MEDIA_MIME_PREFIXES):
                    media_files.append(child)

        return media_files
