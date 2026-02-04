#!/usr/bin/env python3
"""
Google Drive smoke test script.

Tests OAuth authentication, file listing, and description updates.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unlabeled_media_tagger.drive.auth import get_drive_service
from unlabeled_media_tagger.drive.files import (
    list_files_in_folder,
    update_file_description,
    get_file,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test Google Drive integration: list files and update description"
    )
    parser.add_argument(
        "--folder-id",
        default="1GIkurU85_oFA2jlDBtuanehretkl_Z3R",
        help="Google Drive folder ID to list files from",
    )
    parser.add_argument(
        "--file-id",
        help="Specific file ID to update (if not provided, uses first non-folder file)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Google Drive Smoke Test")
    print("=" * 60)
    print()

    # Step 1: Authenticate
    print("🔐 Authenticating with Google Drive...")
    try:
        service = get_drive_service()
        print("✅ Authentication successful!\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return 1

    # Step 2: List files in folder
    print(f"📂 Listing files from folder: {args.folder_id}")
    try:
        files = list_files_in_folder(service, args.folder_id, page_size=20)
        
        if not files:
            print("⚠️  No files found in folder")
            return 1
        
        print(f"✅ Found {len(files)} file(s):\n")
        
        # Print table header
        print(f"{'#':<4} {'Name':<40} {'ID':<35} {'Type':<30}")
        print("-" * 110)
        
        # Print files
        for idx, file in enumerate(files, 1):
            name = file.get("name", "N/A")[:39]
            file_id = file.get("id", "N/A")[:34]
            mime_type = file.get("mimeType", "N/A")[:29]
            print(f"{idx:<4} {name:<40} {file_id:<35} {mime_type:<30}")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return 1

    # Step 3: Select file to update
    if args.file_id:
        target_file_id = args.file_id
        print(f"🎯 Using specified file ID: {target_file_id}\n")
    else:
        # Pick first non-folder file
        non_folder_files = [
            f for f in files 
            if f.get("mimeType") != "application/vnd.google-apps.folder"
        ]
        
        if not non_folder_files:
            print("⚠️  No non-folder files found to update")
            return 1
        
        target_file = non_folder_files[0]
        target_file_id = target_file["id"]
        target_file_name = target_file["name"]
        print(f"🎯 Selected first non-folder file: {target_file_name}")
        print(f"   ID: {target_file_id}\n")

    # Step 4: Update file description
    timestamp = datetime.utcnow().isoformat() + "Z"
    new_description = f"UMT: drive writeback test {timestamp}"
    
    print(f"📝 Updating description to: {new_description}")
    try:
        updated_file = update_file_description(service, target_file_id, new_description)
        print("✅ Description updated successfully!\n")
    except Exception as e:
        print(f"❌ Failed to update description: {e}")
        return 1

    # Step 5: Verify update
    print("🔍 Verifying update...")
    try:
        file_info = get_file(service, target_file_id)
        actual_description = file_info.get("description", "")
        
        print(f"   File name: {file_info.get('name')}")
        print(f"   File ID: {file_info.get('id')}")
        print(f"   Description: {actual_description}")
        
        if actual_description == new_description:
            print("\n✅ Verification successful! Description matches.\n")
        else:
            print("\n⚠️  Warning: Description doesn't match expected value\n")
            
    except Exception as e:
        print(f"❌ Failed to verify update: {e}")
        return 1

    print("=" * 60)
    print("✅ Smoke test completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
