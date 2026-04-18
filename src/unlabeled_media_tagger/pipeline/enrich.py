"""Enrich Stage - metadata enrichment and Drive writeback."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from unlabeled_media_tagger.drive.files import update_file_description


DESCRIPTION_START = "<unlabeled-media-tagger>"
DESCRIPTION_END = "</unlabeled-media-tagger>"


class EnrichStage:
    """
    Enrich stage for writing metadata back to media files.
    
    This stage is responsible for:
    - Formatting detected metadata for storage
    - Writing metadata to local media files
    - Updating Google Drive metadata/properties
    - Generating tags and descriptions
    """
    
    def __init__(self, config=None, service=None):
        """
        Initialize the enrich stage.
        
        Args:
            config: Configuration dictionary for enrichment settings
        """
        self.config = config or {}
        self.service = service
    
    def enrich_local(self, media_file, metadata):
        """
        Write metadata to local media file.
        
        Args:
            media_file: Path to the media file
            metadata: Dictionary of metadata to write
            
        Returns:
            Success status
            
        Raises:
            NotImplementedError: Local file metadata writeback is not yet supported.
        """
        raise NotImplementedError("Local file metadata writeback is not implemented")
    
    def enrich_drive(self, file_id, metadata, existing_description=""):
        """
        Update Google Drive metadata for a file.
        
        Args:
            file_id: Google Drive file ID
            metadata: Dictionary of metadata to write
            
        Returns:
            Success status
            
        Raises:
            ValueError: If no Drive service was configured.
        """
        if self.service is None:
            raise ValueError("Drive service is required for Drive enrichment")

        description = build_description(existing_description or "", metadata)
        return update_file_description(self.service, file_id, description)


def build_drive_metadata(face_rows: list[dict], schema_version: int = 1) -> dict:
    """Build the compact JSON payload stored in each Drive file description."""
    clusters = sorted({row["cluster_label"] for row in face_rows})
    return {
        "schema": "unlabeled-media-tagger",
        "version": schema_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "face_count": len(face_rows),
        "clusters": clusters,
    }


def build_description(existing_description: str, metadata: dict) -> str:
    """
    Preserve user-written description text and replace this tool's managed JSON block.
    """
    base_description = strip_managed_block(existing_description).strip()
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    managed_block = f"{DESCRIPTION_START}{payload}{DESCRIPTION_END}"
    if base_description:
        return f"{base_description}\n\n{managed_block}"

    return managed_block


def strip_managed_block(description: str) -> str:
    """Remove the existing managed metadata block, if present."""
    start = description.find(DESCRIPTION_START)
    end = description.find(DESCRIPTION_END)
    if start == -1 or end == -1 or end < start:
        return description

    end += len(DESCRIPTION_END)
    return f"{description[:start]}{description[end:]}"
