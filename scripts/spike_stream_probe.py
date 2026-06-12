#!/usr/bin/env python
"""Feasibility probe for frame-targeted streaming (spike, uncommitted).

Question this answers: when cv2's built-in ffmpeg seeks to a timestamp in a
Drive video served over HTTP, does it issue *range* requests (pulling only the
bytes it needs) or does it linearly read the whole file?

How: stand up a tiny local proxy that forwards GETs to Drive's get_media
endpoint, injecting the OAuth bearer token and passing the client's Range
header through. The proxy tallies every body byte it relays. Then point
cv2.VideoCapture at the proxy URL, seek to a few timestamps, read one frame at
each, and compare bytes-pulled against the full file size.

Usage: spike_stream_probe.py FILE_ID [--at 10,30,60,120] [--port 0]
"""
from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = "secrets/token.json"
MEDIA_URL = "https://www.googleapis.com/drive/v3/files/{fid}?alt=media&supportsAllDrives=true"

# shared across proxy threads
_bytes = {"n": 0, "reqs": 0}
_lock = threading.Lock()


def get_token() -> str:
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w") as fh:
            fh.write(creds.to_json())
    return creds.token


def make_handler(token: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def _serve(self, body: bool):
            fid = self.path.rsplit("/", 1)[-1]
            headers = {"Authorization": f"Bearer {token}"}
            rng = self.headers.get("Range")
            if rng:
                headers["Range"] = rng
            n = 0  # bytes pulled FROM Drive == the WAN cost we want to measure
            try:
                up = requests.get(
                    MEDIA_URL.format(fid=fid), headers=headers,
                    stream=True, timeout=60,
                )
                try:
                    self.send_response(up.status_code)
                    for h in ("Content-Length", "Content-Range",
                              "Accept-Ranges", "Content-Type"):
                        if h in up.headers:
                            self.send_header(h, up.headers[h])
                    if "Accept-Ranges" not in up.headers:
                        self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    if body:
                        for chunk in up.iter_content(64 * 1024):
                            if not chunk:
                                break
                            n += len(chunk)  # count on fetch, before client write
                            try:
                                self.wfile.write(chunk)
                            except (ConnectionResetError, BrokenPipeError):
                                # ffmpeg abandons a ranged conn when it seeks
                                # again; the bytes were still pulled from Drive.
                                break
                finally:
                    up.close()
            except (ConnectionResetError, BrokenPipeError):
                pass
            with _lock:
                _bytes["n"] += n
                _bytes["reqs"] += 1

        def do_GET(self):
            self._serve(body=True)

        def do_HEAD(self):
            self._serve(body=False)

    return H


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # ffmpeg resetting ranged connections is expected; don't dump tracebacks.
        pass


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_id")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between sampled frames (matches the pipeline)")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--full-size", type=int, default=None,
                    help="full file size in bytes, for the savings ratio")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    import cv2

    token = get_token()
    srv = QuietServer(("127.0.0.1", args.port), make_handler(token))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/f/{args.file_id}"
    print(f"proxy on :{port}  ->  {url}", flush=True)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("FAILED: cv2 could not open the stream", flush=True)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = nframes / fps if fps else 0
    with _lock:
        open_bytes = _bytes["n"]
    print(f"opened: fps={fps:.2f} frame_count={nframes:.0f} "
          f"duration={duration:.1f}s bytes_to_open={human(open_bytes)} "
          f"reqs={_bytes['reqs']}", flush=True)

    # Replicate the pipeline's sample grid: t = 0, interval, 2*interval, ...
    times = [i * args.interval for i in range(args.max_frames)
             if i * args.interval < max(duration, args.interval)]
    decoded = 0
    for t in times:
        before = _bytes["n"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            decoded += 1
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"  t={t:>6.1f}s ok={ok} landed={actual_ms/1000:7.2f}s "
              f"(+{human(_bytes['n'] - before)})", flush=True)

    cap.release()
    with _lock:
        total = _bytes["n"]
        reqs = _bytes["reqs"]
    print(f"\nframes decoded: {decoded}/{len(times)}", flush=True)
    print(f"TOTAL bytes pulled from Drive: {human(total)} "
          f"across {reqs} HTTP requests", flush=True)
    if args.full_size:
        print(f"full file size: {human(args.full_size)}  ->  "
              f"streaming pulled {100*total/args.full_size:.2f}%  "
              f"({args.full_size/total:.1f}x less)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
