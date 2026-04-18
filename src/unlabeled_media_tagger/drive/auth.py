"""Google Drive OAuth authentication module."""

import os
from pathlib import Path

# Scope that allows reading files and updating file metadata
SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_drive_service(
    credentials_path: str = "secrets/credentials.json",
    token_path: str = "secrets/token.json",
    verbose: bool = False,
):
    """
    Return an authenticated googleapiclient Drive v3 service.
    
    Uses OAuth Installed App flow (local browser auth) and caches token
    to secrets/token.json for subsequent runs.
    
    Args:
        credentials_path: Path to the OAuth client credentials JSON file
        token_path: Path where the OAuth token will be cached
        
    Returns:
        googleapiclient.discovery.Resource: Authenticated Drive v3 service
        
    Raises:
        FileNotFoundError: If credentials.json is missing
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "Google Drive dependencies are not installed. Install the package "
            "with runtime dependencies or run: pip install -r requirements.txt"
        ) from exc

    creds = None
    
    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"OAuth credentials file not found at: {credentials_path}\n\n"
            f"Please follow these steps:\n"
            f"1. Go to Google Cloud Console (https://console.cloud.google.com/)\n"
            f"2. Create or select a project\n"
            f"3. Enable the Google Drive API\n"
            f"4. Go to 'Credentials' and create OAuth 2.0 Client ID (Desktop app)\n"
            f"5. Download the JSON file\n"
            f"6. Save it as: {credentials_path}"
        )
    
    # Load cached token if it exists
    if os.path.exists(token_path):
        if verbose:
            print(f"Loading cached Google OAuth token: {token_path}", flush=True)
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, trigger OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh expired token
            if verbose:
                print("Refreshing expired Google OAuth token...", flush=True)
            creds.refresh(Request())
        else:
            # Run OAuth flow (opens browser)
            if verbose:
                print("Opening browser for Google OAuth consent...", flush=True)
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save token for next run
        token_dir = Path(token_path).parent
        token_dir.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
        if verbose:
            print(f"Saved Google OAuth token: {token_path}", flush=True)
    
    # Build and return Drive service
    if verbose:
        print("Building Google Drive service...", flush=True)
    service = build("drive", "v3", credentials=creds)
    return service
