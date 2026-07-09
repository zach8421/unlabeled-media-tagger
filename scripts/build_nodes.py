#!/usr/bin/env python
"""Build the global-clustering input (nodes/) from the track layer.

A "node" is what global clustering operates on: exactly one row per track
(image faces are singleton tracks, so every embedded face is covered). This
step joins on event metadata and derives the same-frame cannot-link pairs:

  - two tracks touching the same (source_file_id, frame_index) contain faces
    that co-occurred in one frame -> different people by construction;
  - two faces in one still image likewise (frame key is the image itself).

Outputs under <embed-dir>/nodes/:
  nodes.csv             node_id (= row), track_id, rep face parts, event_path,
                        cohesion + quality carried from tracks.csv
  node_embeddings.npy   float32 (n_nodes, 512) unit-norm (copy of track
                        embeddings, row-aligned with nodes.csv)
  same_frame_pairs.csv  node_id_a, node_id_b (deduped, a < b)
  nodes_meta.json       counts + provenance

CPU-only, minutes, atomic, rerun-cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

NODES_COLUMNS = [
    "node_id", "track_id", "source_file_id", "media_type", "n_faces",
    "medoid_crop_file_name", "rep_crop_file_names",
    "mean_intra_sim", "min_intra_sim", "max_quality_score",
    "event_path",
]


def load_event_paths(clean_manifest: Path) -> dict:
    """file_id -> containing folder path (one Drive folder = one event)."""
    events = {}
    with clean_manifest.open(newline="") as fh:
        for row in csv.DictReader(fh):
            path = row.get("path") or ""
            events[row["file_id"]] = str(Path(path).parent) if path else ""
    return events


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--clean-manifest",
                    default="outputs/manifests/folder2_clean.csv")
    args = ap.parse_args(argv)

    embed_dir = Path(args.embed_dir)
    tracks_dir = embed_dir / "tracks"
    out_dir = embed_dir / "nodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_event_paths(Path(args.clean_manifest))
    print(f"event join: {len(events)} file_ids", flush=True)

    # tracks.csv row order == track_index.csv row order == embedding row order
    # (build_tracks.py writes them in lockstep), so node_id = position.
    node_id_of_track: dict[str, int] = {}
    nodes_tmp = out_dir / "nodes.csv.tmp"
    n_nodes = 0
    with (tracks_dir / "tracks.csv").open(newline="") as fh, \
            nodes_tmp.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=NODES_COLUMNS)
        writer.writeheader()
        for node_id, row in enumerate(csv.DictReader(fh)):
            node_id_of_track[row["track_id"]] = node_id
            writer.writerow({
                "node_id": node_id,
                "track_id": row["track_id"],
                "source_file_id": row["source_file_id"],
                "media_type": row["media_type"],
                "n_faces": row["n_faces"],
                "medoid_crop_file_name": row["medoid_crop_file_name"],
                "rep_crop_file_names": row["rep_crop_file_names"],
                "mean_intra_sim": row["mean_intra_sim"],
                "min_intra_sim": row["min_intra_sim"],
                "max_quality_score": row["max_quality_score"],
                "event_path": events.get(row["source_file_id"], ""),
            })
            n_nodes += 1

    # Same-frame pairs: join track members back to their frame via
    # faces_index.csv. frame key: (source_file_id, frame_index) — for images
    # frame_index is empty, so the key is the image itself. Correct either way.
    frame_of_face: dict[tuple, str] = {}
    with (embed_dir / "faces_index.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            frame_of_face[(row["source_file_id"], row["crop_file_name"])] = \
                row["frame_index"]

    tracks_in_frame = defaultdict(set)
    with (tracks_dir / "track_members.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            frame = frame_of_face.get(
                (row["source_file_id"], row["crop_file_name"]), "")
            tracks_in_frame[(row["source_file_id"], frame)].add(row["track_id"])

    pairs = set()
    for track_ids in tracks_in_frame.values():
        if len(track_ids) < 2:
            continue
        ids = sorted(node_id_of_track[t] for t in track_ids)
        pairs.update(combinations(ids, 2))

    pairs_tmp = out_dir / "same_frame_pairs.csv.tmp"
    with pairs_tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id_a", "node_id_b"])
        writer.writerows(sorted(pairs))

    import numpy as np
    n_vectors = np.load(tracks_dir / "track_embeddings.npy",
                        mmap_mode="r").shape[0]
    if n_vectors != n_nodes:
        print(f"ABORT: tracks.csv has {n_nodes} rows but track_embeddings.npy "
              f"has {n_vectors} — regenerate the track layer.", flush=True)
        return 1

    shutil.copyfile(tracks_dir / "track_embeddings.npy",
                    out_dir / "node_embeddings.tmp.npy")
    os.replace(out_dir / "node_embeddings.tmp.npy",
               out_dir / "node_embeddings.npy")
    os.replace(nodes_tmp, out_dir / "nodes.csv")
    os.replace(pairs_tmp, out_dir / "same_frame_pairs.csv")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "nodes": n_nodes,
        "same_frame_pairs": len(pairs),
        "tracks_meta": json.loads((tracks_dir / "tracks_meta.json").read_text()),
    }
    (out_dir / "nodes_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"=== nodes done === nodes={n_nodes} "
          f"same_frame_pairs={len(pairs)}", flush=True)
    print(f"output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
