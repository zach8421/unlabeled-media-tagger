#!/usr/bin/env python
"""Measure whether concurrent Drive downloads scale aggregate throughput.

Decides the FOLDER 2 bottleneck: if N parallel streams ~= N x single-stream,
the limit is per-stream (concurrency wins big). If aggregate flatlines, the
limit is the ISP/account pipe (concurrency can't help; full-download is locked).

Pulls a fixed byte chunk (Range request, no full downloads) from DISTINCT files
at parallelism 1/3/6 and reports aggregate + per-stream MB/s. Disjoint files per
level so nothing is server-cached across levels.

Usage: spike_concurrency_test.py MANIFEST [--chunk-mb 200] [--levels 1,3,6]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = "secrets/token.json"
MEDIA = ("https://www.googleapis.com/drive/v3/files/{fid}"
         "?alt=media&supportsAllDrives=true")
MB = 1024 * 1024


def get_token():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w") as fh:
            fh.write(creds.to_json())
    return creds.token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--chunk-mb", type=int, default=200)
    ap.add_argument("--levels", default="1,3,6")
    args = ap.parse_args()
    chunk = args.chunk_mb * MB
    levels = [int(x) for x in args.levels.split(",")]

    # pool of distinct files big enough to serve a full chunk
    pool = []
    with open(args.manifest) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) != 4:
                continue
            fid, path, mime, size = row
            ext = os.path.splitext(path)[1].lower()
            if mime.startswith("video/") and ext in {".mp4", ".mov"} \
                    and int(size or 0) > args.chunk_mb * MB * 1.2:
                pool.append(fid)
    random.Random(7).shuffle(pool)

    token = get_token()
    hdr = {"Authorization": f"Bearer {token}", "Range": f"bytes=0-{chunk-1}"}

    def pull(fid):
        t0 = time.time()
        n = 0
        resp = requests.get(MEDIA.format(fid=fid), headers=hdr, stream=True, timeout=120)
        for c in resp.iter_content(256 * 1024):
            n += len(c)
        resp.close()
        return n, time.time() - t0

    print(f"chunk={args.chunk_mb}MB per stream; pool={len(pool)} distinct files\n")
    print(f"{'parallel':>8} {'streams':>7} {'agg MB/s':>9} "
          f"{'per-stream MB/s':>16} {'wall s':>7}")
    idx = 0
    for p in levels:
        fids = pool[idx:idx + p]
        idx += p
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=p) as ex:
            res = list(ex.map(pull, fids))
        wall = time.time() - t0
        total = sum(n for n, _ in res)
        agg = total / MB / wall
        per = [n / MB / dt for n, dt in res]
        per_avg = sum(per) / len(per)
        print(f"{p:8d} {len(fids):7d} {agg:9.1f} {per_avg:16.1f} {wall:7.1f}",
              flush=True)

    print("\nRead: if 'agg MB/s' rises ~linearly with parallelism -> per-stream "
          "limit, concurrency wins. If it flatlines -> pipe/account ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
