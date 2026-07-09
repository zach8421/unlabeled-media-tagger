#!/usr/bin/env python
"""Collapse the embedded face set into per-file person-tracks.

Consumes faces_index.csv + embeddings/shard_*.npy (from
scripts/run_embed_batch.py) and writes the track layer under
<embed-dir>/tracks/. Videos get embedding-based within-file grouping
(clustering/tracks.py); every image face is its own singleton track.

CPU-only, no deepface import, rerun-cheap (minutes), atomic outputs — no
resume machinery. Re-run freely after changing gates or adding
--exclude-keys (track-impurity flags exported from the label store).

Outputs:
  tracks/tracks.csv           one row per track (medoid, reps, cohesion)
  tracks/track_members.csv    track_id -> every member face (the fan-out map)
  tracks/track_embeddings.npy float32 (n_tracks, 512), unit-norm mean vectors
  tracks/track_index.csv      row-aligned with the .npy
  tracks/tracks_meta.json     gates, input fingerprint, reduction stats
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unlabeled_media_tagger.clustering.tracks import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_MERGE_MEAN,
    DEFAULT_MERGE_MIN,
    build_tracks_for_file,
)

TRACKS_COLUMNS = [
    "track_id", "source_file_id", "media_type", "n_faces",
    "first_frame_index", "last_frame_index",
    "first_timestamp_sec", "last_timestamp_sec",
    "medoid_crop_file_name", "rep_crop_file_names",
    "mean_intra_sim", "min_intra_sim", "max_quality_score",
]

MEMBERS_COLUMNS = ["track_id", "source_file_id", "crop_file_name"]

INDEX_COLUMNS = ["row", "track_id", "source_file_id", "media_type", "n_faces"]


class ShardStore:
    """Random access to embeddings by (shard, row) via cached memmaps."""

    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = embeddings_dir
        self._open: dict[int, np.ndarray] = {}

    def get(self, shard: int, row: int) -> np.ndarray:
        mm = self._open.get(shard)
        if mm is None:
            mm = np.load(self.embeddings_dir / f"shard_{shard:05d}.npy",
                         mmap_mode="r")
            self._open[shard] = mm
        return mm[row]


def iter_file_groups(faces_index: Path):
    """Yield (source_file_id, [row dicts]) — rows are contiguous per file."""
    with faces_index.open(newline="") as fh:
        current_id, group = None, []
        for row in csv.DictReader(fh):
            if row["source_file_id"] != current_id:
                if group:
                    yield current_id, group
                current_id, group = row["source_file_id"], []
            group.append(row)
        if group:
            yield current_id, group


def _num(value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--edge-threshold", type=float,
                    default=DEFAULT_EDGE_THRESHOLD)
    ap.add_argument("--merge-min", type=float, default=DEFAULT_MERGE_MIN)
    ap.add_argument("--merge-mean", type=float, default=DEFAULT_MERGE_MEAN)
    ap.add_argument("--exclude-keys", default=None,
                    help="File of face_keys (source_file_id/crop_file_name, one "
                         "per line) to pin as singleton tracks (label-store "
                         "track-impurity flags)")
    ap.add_argument("--min-confidence", type=float, default=0.95,
                    help="Drop faces below this detector confidence entirely. "
                         "12.4%% of ok faces are conf<0.5 detector false "
                         "positives (whole scenes, logos) that form a junk "
                         "attractor cluster — the v1.1 size+blur gate never "
                         "checked confidence. 0 disables.")
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args(argv)

    embed_dir = Path(args.embed_dir)
    faces_index = embed_dir / "faces_index.csv"
    out_dir = embed_dir / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_keys: set[str] = set()
    if args.exclude_keys:
        with open(args.exclude_keys) as fh:
            exclude_keys = {line.strip() for line in fh if line.strip()}
        print(f"excluding {len(exclude_keys)} face_keys into singletons",
              flush=True)

    # Detector confidence lives in embed_manifest.csv (faces_index doesn't
    # carry it). Only sub-threshold keys are kept in memory.
    low_conf_keys: set[tuple] = set()
    if args.min_confidence > 0:
        with (embed_dir / "embed_manifest.csv").open(newline="") as fh:
            for row in csv.DictReader(fh):
                if float(row["confidence"] or 0) < args.min_confidence:
                    low_conf_keys.add(
                        (row["source_file_id"], row["crop_file_name"]))
        print(f"confidence gate: dropping {len(low_conf_keys)} faces below "
              f"{args.min_confidence}", flush=True)

    store = ShardStore(embed_dir / "embeddings")

    tracks_tmp = out_dir / "tracks.csv.tmp"
    members_tmp = out_dir / "track_members.csv.tmp"
    index_tmp = out_dir / "track_index.csv.tmp"

    stats = defaultdict(int)
    embeddings_out = []

    with tracks_tmp.open("w", newline="") as tfh, \
            members_tmp.open("w", newline="") as mfh, \
            index_tmp.open("w", newline="") as ifh:
        tw = csv.DictWriter(tfh, fieldnames=TRACKS_COLUMNS)
        mw = csv.DictWriter(mfh, fieldnames=MEMBERS_COLUMNS)
        iw = csv.DictWriter(ifh, fieldnames=INDEX_COLUMNS)
        tw.writeheader()
        mw.writeheader()
        iw.writeheader()

        for n_files, (file_id, rows) in enumerate(iter_file_groups(faces_index)):
            if args.limit_files is not None and n_files >= args.limit_files:
                break
            if low_conf_keys:
                kept = [r for r in rows
                        if (file_id, r["crop_file_name"]) not in low_conf_keys]
                stats["faces_dropped_low_conf"] += len(rows) - len(kept)
                rows = kept
                if not rows:
                    stats["files"] += 1
                    continue
            media_type = rows[0]["media_type"]
            emb = np.stack([store.get(int(r["shard"]), int(r["row"]))
                            for r in rows]).astype(np.float32)

            if media_type == "video" and len(rows) > 1:
                excluded = {i for i, r in enumerate(rows)
                            if f"{file_id}/{r['crop_file_name']}" in exclude_keys}
                tracks = build_tracks_for_file(
                    rows, emb,
                    edge_threshold=args.edge_threshold,
                    merge_min=args.merge_min,
                    merge_mean=args.merge_mean,
                    exclude_indices=excluded,
                )
            else:
                # Images (and single-face videos): one singleton per face.
                tracks = build_tracks_for_file(
                    rows, emb, edge_threshold=2.0)  # >1.0: no edges, all singleton

            for t_num, track in enumerate(tracks):
                track_id = f"{file_id}__t{t_num:03d}"
                m_rows = [rows[i] for i in track.member_indices]
                frames = [_num(r["frame_index"], int) for r in m_rows
                          if r["frame_index"] != ""]
                timestamps = [_num(r["timestamp_sec"]) for r in m_rows
                              if r["timestamp_sec"] != ""]
                qualities = [_num(r["quality_score"]) for r in m_rows
                             if r["quality_score"] != ""]
                tw.writerow({
                    "track_id": track_id,
                    "source_file_id": file_id,
                    "media_type": media_type,
                    "n_faces": len(m_rows),
                    "first_frame_index": min(frames) if frames else "",
                    "last_frame_index": max(frames) if frames else "",
                    "first_timestamp_sec": min(timestamps) if timestamps else "",
                    "last_timestamp_sec": max(timestamps) if timestamps else "",
                    "medoid_crop_file_name":
                        rows[track.medoid_index]["crop_file_name"],
                    "rep_crop_file_names": ";".join(
                        rows[i]["crop_file_name"]
                        for i in track.representative_indices),
                    "mean_intra_sim": track.mean_intra_sim,
                    "min_intra_sim": track.min_intra_sim,
                    "max_quality_score": max(qualities) if qualities else "",
                })
                mw.writerows({
                    "track_id": track_id,
                    "source_file_id": file_id,
                    "crop_file_name": rows[i]["crop_file_name"],
                } for i in track.member_indices)
                iw.writerow({
                    "row": len(embeddings_out), "track_id": track_id,
                    "source_file_id": file_id, "media_type": media_type,
                    "n_faces": len(m_rows),
                })
                embeddings_out.append(track.embedding)
                stats["tracks"] += 1
                stats["multi_face_tracks"] += len(m_rows) > 1

            stats["files"] += 1
            stats["faces"] += len(rows)
            if stats["files"] % 20000 == 0:
                print(f"  {stats['files']} files -> {stats['tracks']} tracks",
                      flush=True)

    matrix = (np.stack(embeddings_out) if embeddings_out
              else np.empty((0, 512), np.float32))
    npy_tmp = out_dir / "track_embeddings.tmp.npy"
    np.save(npy_tmp, matrix)
    os.replace(npy_tmp, out_dir / "track_embeddings.npy")
    os.replace(tracks_tmp, out_dir / "tracks.csv")
    os.replace(members_tmp, out_dir / "track_members.csv")
    os.replace(index_tmp, out_dir / "track_index.csv")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "faces_index": str(faces_index),
        "edge_threshold": args.edge_threshold,
        "merge_min": args.merge_min,
        "merge_mean": args.merge_mean,
        "exclude_keys": len(exclude_keys),
        "min_confidence": args.min_confidence,
        "faces_dropped_low_conf": stats["faces_dropped_low_conf"],
        "files": stats["files"],
        "faces_in": stats["faces"],
        "tracks_out": stats["tracks"],
        "multi_face_tracks": stats["multi_face_tracks"],
        "reduction_ratio": round(stats["faces"] / stats["tracks"], 3)
        if stats["tracks"] else None,
    }
    (out_dir / "tracks_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\n=== tracks done === files={stats['files']} "
          f"faces={stats['faces']} tracks={stats['tracks']} "
          f"(x{meta['reduction_ratio']} reduction, "
          f"{stats['multi_face_tracks']} multi-face)", flush=True)
    print(f"output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
