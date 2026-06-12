#!/usr/bin/env python
"""Project frame-targeted-streaming savings across the FOLDER 2 video archive.

Model (derived + validated in the spike): seeking to an arbitrary timestamp
costs ~one GOP, so streaming a video pulls

    stream_bytes = frames_sampled * gop_bytes
                 = min(max_frames, duration/interval) * (size/duration)*gop_sec

The bitrate (size/duration) cancels in the ratio, so the per-file savings ratio
depends only on DURATION (and the sampling params):

    ratio = min(1, gop_sec/interval, max_frames*gop_sec/duration)

We only have sizes for all 90k videos, so we size-stratify, sample durations
per bucket (Drive videoMediaMetadata.durationMillis -- metadata only, no
downloads), compute each bucket's size-weighted ratio, and weight buckets by
their true population bytes.

Usage: spike_duration_projection.py MANIFEST [--per-bucket 150]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = "secrets/token.json"
GET_URL = ("https://www.googleapis.com/drive/v3/files/{fid}"
           "?fields=size,videoMediaMetadata/durationMillis&supportsAllDrives=true")
STREAMABLE = {".mp4", ".mov", ".m4v", ".mts", ".3gp"}

GB = 1024 ** 3
TB = 1024 ** 4
# (low_GB, high_GB) bucket edges
EDGES = [0, 0.1, 0.5, 1, 2, 5, 10, 30, 1e9]

# sampling params == the pipeline's defaults
INTERVAL = 2.0
MAX_FRAMES = 60
GOP_CENTRAL = 2.0
GOP_SENS = [1.5, 2.0, 3.0]
# wired throughput anchors (MB/s) measured in the baseline run
RATE_SINGLE = 35 * 1024 * 1024
RATE_GIGABIT = 118 * 1024 * 1024


def get_token():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w") as fh:
            fh.write(creds.to_json())
    return creds.token


def ratio(duration, gop_sec):
    if not duration or duration <= 0:
        return None
    return min(1.0, gop_sec / INTERVAL, MAX_FRAMES * gop_sec / duration)


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def days(total_bytes, rate):
    return total_bytes / rate / 86400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--per-bucket", type=int, default=150)
    args = ap.parse_args()

    vids = []
    with open(args.manifest) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) != 4:
                continue
            fid, path, mime, size = row
            if not mime.startswith("video/"):
                continue
            ext = os.path.splitext(path)[1].lower()
            vids.append((int(size or 0), ext, fid, ext in STREAMABLE))

    total_video = sum(v[0] for v in vids)
    nonstream_bytes = sum(v[0] for v in vids if not v[3])
    streamable = [v for v in vids if v[3]]

    # bucket streamable videos by size
    buckets = {i: [] for i in range(len(EDGES) - 1)}
    for sz, ext, fid, _ in streamable:
        for i in range(len(EDGES) - 1):
            if EDGES[i] * GB <= sz < EDGES[i + 1] * GB:
                buckets[i].append((sz, fid))
                break

    rng = random.Random(42)
    sample = []  # (bucket_idx, size, fid)
    for bi, items in buckets.items():
        pick = items if len(items) <= args.per_bucket else rng.sample(items, args.per_bucket)
        for sz, fid in pick:
            sample.append((bi, sz, fid))

    print(f"videos={len(vids)} total={total_video/TB:.2f}TB  "
          f"streamable={len(streamable)} ({(total_video-nonstream_bytes)/TB:.2f}TB)  "
          f"sampling {len(sample)} durations...", flush=True)

    token = get_token()
    sess = requests.Session()

    def fetch(item):
        bi, sz, fid = item
        try:
            resp = sess.get(GET_URL.format(fid=fid),
                            headers={"Authorization": f"Bearer {token}"}, timeout=30)
            d = resp.json().get("videoMediaMetadata", {}).get("durationMillis")
            return (bi, sz, float(d) / 1000.0 if d else None)
        except Exception:
            return (bi, sz, None)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(fetch, sample))

    # per-bucket size-weighted ratio (central gop), aggregate weighted by pop bytes
    print(f"\n{'bucket':>12} {'pop files':>9} {'pop TB':>7} {'samp':>5} "
          f"{'medDur':>7} {'ratio':>6}")
    no_dur = 0
    proj = {g: 0.0 for g in GOP_SENS}
    for bi in range(len(EDGES) - 1):
        pop = buckets[bi]
        if not pop:
            continue
        pop_bytes = sum(s for s, _ in pop)
        srows = [(sz, dur) for b, sz, dur in results if b == bi]
        durs = [d for _, d in srows if d]
        no_dur += sum(1 for _, d in srows if not d)
        for g in GOP_SENS:
            num = sum(sz * (ratio(d, g) if d else 1.0) for sz, d in srows)
            den = sum(sz for sz, _ in srows) or 1
            proj[g] += (num / den) * pop_bytes
        # display row uses central gop
        numc = sum(sz * (ratio(d, GOP_CENTRAL) if d else 1.0) for sz, d in srows)
        rc = numc / (sum(sz for sz, _ in srows) or 1)
        med = sorted(durs)[len(durs) // 2] if durs else 0
        lab = f"{EDGES[bi]:g}-{EDGES[bi+1]:g}GB" if EDGES[bi+1] < 1e8 else f">{EDGES[bi]:g}GB"
        print(f"{lab:>12} {len(pop):9d} {pop_bytes/TB:7.2f} {len(srows):5d} "
              f"{med:6.0f}s {rc:6.2f}", flush=True)

    print(f"\nsamples missing durationMillis: {no_dur} (counted as full-download)")
    print("\n=== PROJECTION (streamable video; non-streamable counted full) ===")
    for g in GOP_SENS:
        stream_total = proj[g] + nonstream_bytes
        sav = total_video / stream_total
        tag = "  <-- central" if g == GOP_CENTRAL else ""
        print(f" gop={g}s: stream {human_bytes(stream_total)} "
              f"of {total_video/TB:.1f}TB  ->  {100*stream_total/total_video:.1f}%  "
              f"({sav:.1f}x less){tag}")

    stream_central = proj[GOP_CENTRAL] + nonstream_bytes
    print("\n=== WALL-CLOCK (download-bound) ===")
    for label, rate in [("single-stream ~35MB/s", RATE_SINGLE),
                        ("saturated gigabit ~118MB/s", RATE_GIGABIT)]:
        print(f" {label:28s}: full-download {days(total_video, rate):6.1f}d   "
              f"streaming {days(stream_central, rate):5.1f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
