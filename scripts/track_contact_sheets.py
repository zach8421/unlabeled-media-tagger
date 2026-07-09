#!/usr/bin/env python
"""HTML QA sheets for the track layer: eyeball tracks before clustering.

The M3 human gate: a track containing two different people poisons every
downstream stage, so before the first clustering run a human scans:
  (a) the largest tracks (most damage potential),
  (b) random multi-face tracks (unbiased sample),
  (c) the lowest-cohesion tracks (most likely to be impure).

Same static-HTML idiom as scripts/blur_filter_tune.py: file:// img tags in a
CSS grid, one section per track, member crops in temporal order. Open the
report in a browser; note any track showing two people (its track_id) and
either tighten the gates and re-run build_tracks.py, or exclude those
face_keys via --exclude-keys.

Usage:
  track_contact_sheets.py --embed-dir /mnt/media1/folder2_embed \
      --crops-root /mnt/media1/folder2_detect/crops \
      --out outputs/track_qa/track_qa.html
"""

from __future__ import annotations

import argparse
import csv
import html
import random
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def file_url(path: Path) -> str:
    """file:// URL with percent-encoding — crop names contain '#' and '%'."""
    return "file://" + quote(str(path))

MAX_CROPS_PER_TRACK = 24  # temporal spread sample; enough to spot an impostor

STYLE = """
body { font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
.track { border: 1px solid #ddd; border-radius: 8px; padding: 10px;
         margin: 10px 0; background: #fff; }
.track.warn { border-color: #e08000; background: #fff8ef; }
.thead { font-size: 13px; color: #333; margin-bottom: 6px; }
.thead code { background: #f0f0f0; padding: 1px 4px; border-radius: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
        gap: 6px; }
.card { text-align: center; }
.card img { width: 96px; height: 96px; object-fit: cover; border-radius: 4px; }
.card .m { font-size: 10px; color: #777; }
"""


def load_tracks(tracks_csv: Path) -> list[dict]:
    with tracks_csv.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_members(members_csv: Path) -> dict:
    members = defaultdict(list)
    with members_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            members[row["track_id"]].append(row["crop_file_name"])
    return members


def render_track(track: dict, member_crops: list, crops_root: Path,
                 warn: bool) -> list:
    file_id = track["source_file_id"]
    crops = member_crops
    if len(crops) > MAX_CROPS_PER_TRACK:
        step = len(crops) / MAX_CROPS_PER_TRACK
        crops = [crops[int(i * step)] for i in range(MAX_CROPS_PER_TRACK)]
    parts = [f"<div class='track{' warn' if warn else ''}'>"]
    parts.append(
        f"<div class='thead'><code>{html.escape(track['track_id'])}</code> "
        f"| {track['n_faces']} faces | mean_sim {track['mean_intra_sim']} "
        f"| min_sim {track['min_intra_sim']} "
        f"| ts {track['first_timestamp_sec']}–{track['last_timestamp_sec']}s"
        f"</div>")
    parts.append("<div class='grid'>")
    for crop_name in crops:
        crop_path = (crops_root / file_id / crop_name).resolve()
        parts.append("<div class='card'>")
        parts.append(f"<img src='{html.escape(file_url(crop_path))}' "
                     f"loading='lazy' alt=''>")
        parts.append(f"<div class='m'>{html.escape(crop_name[-24:])}</div>")
        parts.append("</div>")
    parts.append("</div></div>")
    return parts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--crops-root", default="/mnt/media1/folder2_detect/crops")
    ap.add_argument("--out", default="outputs/track_qa/track_qa.html")
    ap.add_argument("--n-largest", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-lowest-cohesion", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tracks_dir = Path(args.embed_dir) / "tracks"
    crops_root = Path(args.crops_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(tracks_dir / "tracks.csv")
    members = load_members(tracks_dir / "track_members.csv")
    multi = [t for t in tracks if int(t["n_faces"]) > 1]
    print(f"tracks: {len(tracks)} total, {len(multi)} multi-face", flush=True)

    largest = sorted(multi, key=lambda t: -int(t["n_faces"]))[: args.n_largest]
    lowest = sorted(multi, key=lambda t: float(t["min_intra_sim"]))[
        : args.n_lowest_cohesion]
    rng = random.Random(args.seed)
    pool = [t for t in multi
            if t["track_id"] not in {x["track_id"] for x in largest + lowest}]
    randoms = rng.sample(pool, min(args.n_random, len(pool)))

    sections = [
        (f"Lowest cohesion ({len(lowest)}) — most likely impure, read first",
         lowest, True),
        (f"Largest tracks ({len(largest)})", largest, False),
        (f"Random multi-face tracks ({len(randoms)}, seed={args.seed})",
         randoms, False),
    ]

    parts = ["<meta charset='utf-8'><title>Track QA</title>",
             f"<style>{STYLE}</style>",
             "<h1>Track QA — one section per track; flag any track showing "
             "two different people</h1>"]
    for title, rows, warn in sections:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        for track in rows:
            parts.extend(render_track(
                track, members[track["track_id"]], crops_root, warn))

    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path} — open in a browser "
          f"(file://{out_path.resolve()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
