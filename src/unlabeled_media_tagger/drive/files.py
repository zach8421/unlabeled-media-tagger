"""Google Drive file operations module."""

from typing import Optional


def list_files_in_folder(service, folder_id: str, page_size: int = 20) -> list[dict]:
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
    query = f"'{folder_id}' in parents and trashed=false"
    
    results = service.files().list(
        q=query,
        pageSize=page_size,
        fields="files(id,name,mimeType,modifiedTime,description)"
    ).execute()
    
    files = results.get("files", [])
    return files


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
