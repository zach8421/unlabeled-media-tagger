#!/usr/bin/env python
"""Recursively inventory a Google Drive folder (read-only, no downloads).

Reports count + total bytes by mimeType category (video / image / other /
folder), the largest media files, and writes a media manifest CSV
(file_id, path, mimeType, size) for the detect-only run to consume.

Always passes the Shared-Drive flags (supportsAllDrives /
includeItemsFromAllDrives) — without them a Shared-Drive folder returns
404 / empty listings (see dev-log 2026-06-11 detect-only run).

Usage:
  inventory_drive_folder.py FOLDER_ID [--manifest out.csv]
                                      [--max-items N]   # stop the walk early
                                      [--top N]         # largest-files to show
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from unlabeled_media_tagger.drive.auth import get_drive_service

FOLDER_MIME = "application/vnd.google-apps.folder"
# list() accepts both flags; get() accepts only supportsAllDrives.
ALL_DRIVES = dict(supportsAllDrives=True, includeItemsFromAllDrives=True)
ALL_DRIVES_GET = dict(supportsAllDrives=True)


def list_children(service, folder_id):
    """Yield all non-trashed children of a folder, paginated, all-drives aware."""
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=1000,
            pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            **ALL_DRIVES,
        ).execute()
        for f in resp.get("files", []):
            yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def category(mime: str) -> str:
    if mime == FOLDER_MIME:
        return "folder"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("image/"):
        return "image"
    return "other"


def walk(service, root_id, max_items=None):
    """BFS the folder tree. Returns (media_rows, stats, largest, n_folders)."""
    root = service.files().get(
        fileId=root_id, fields="id,name,mimeType,driveId", **ALL_DRIVES_GET
    ).execute()
    print(
        f"root: {root.get('name')!r} mime={root.get('mimeType')} "
        f"driveId={root.get('driveId')}",
        flush=True,
    )

    stats = defaultdict(lambda: {"count": 0, "bytes": 0})
    media_rows = []          # (file_id, path, mimeType, size)
    largest = []             # (size, path, mimeType)
    n_folders = 0
    queue = [(root_id, root.get("name", ""))]
    seen_items = 0

    while queue:
        folder_id, folder_path = queue.pop(0)
        for child in list_children(service, folder_id):
            seen_items += 1
            mime = child.get("mimeType", "")
            cat = category(mime)
            name = child.get("name", "")
            path = f"{folder_path}/{name}"
            size = int(child.get("size", 0) or 0)
            stats[cat]["count"] += 1
            stats[cat]["bytes"] += size
            if cat == "folder":
                n_folders += 1
                queue.append((child["id"], path))
            elif cat in ("video", "image"):
                media_rows.append((child["id"], path, mime, size))
                largest.append((size, path, mime))
            if seen_items % 500 == 0:
                print(f"  ...walked {seen_items} items, queue depth {len(queue)}",
                      flush=True)
            if max_items and seen_items >= max_items:
                print(f"  [stopped early at --max-items={max_items}]", flush=True)
                queue = []
                break

    largest.sort(reverse=True)
    return media_rows, stats, largest, n_folders


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_id")
    ap.add_argument("--manifest")
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    service = get_drive_service(verbose=True)
    media_rows, stats, largest, n_folders = walk(
        service, args.folder_id, max_items=args.max_items
    )

    print("\n=== inventory by category ===")
    total_bytes = 0
    for cat in ("video", "image", "other", "folder"):
        s = stats[cat]
        total_bytes += s["bytes"]
        print(f"  {cat:7s} count={s['count']:6d}  bytes={human(s['bytes'])}")
    print(f"  {'TOTAL':7s} bytes={human(total_bytes)}  subfolders={n_folders}")

    print(f"\n=== largest {args.top} media files ===")
    for size, path, mime in largest[: args.top]:
        print(f"  {human(size):>9s}  {mime:12s}  {path}")

    if args.manifest:
        with open(args.manifest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file_id", "path", "mimeType", "size"])
            w.writerows(media_rows)
        print(f"\nwrote manifest: {args.manifest} ({len(media_rows)} media rows)")


if __name__ == "__main__":
    sys.exit(main())
