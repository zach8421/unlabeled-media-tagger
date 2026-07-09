"""Resolve cannot-link constraints to node-id pairs for one clustering run.

Two provenances, one mechanism (sparse_cluster's enemy sets):
  - same-frame pairs derived by build_nodes.py (physics: two faces in one
    frame are different people);
  - human judgments from the label store's derived views
    (labels/derived/cannot_links_face_face.csv, face_key pairs). Face keys
    resolve to nodes via the track-membership map. A face-face pair that
    lands INSIDE one node is unenforceable at node granularity: it is
    returned separately as track-impurity face_keys, to be fed back to
    build_tracks.py --exclude-keys on the next track rebuild.

The label store may not exist yet (M5); everything degrades to same-frame
pairs only.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_same_frame_pairs(nodes_dir: Path) -> list:
    pairs = []
    path = Path(nodes_dir) / "same_frame_pairs.csv"
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pairs.append((int(row["node_id_a"]), int(row["node_id_b"])))
    return pairs


def load_node_of_face(tracks_dir: Path, node_id_of_track: dict) -> dict:
    """face_key -> node_id via track membership."""
    node_of_face = {}
    with (Path(tracks_dir) / "track_members.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            node_id = node_id_of_track.get(row["track_id"])
            if node_id is not None:
                key = f"{row['source_file_id']}/{row['crop_file_name']}"
                node_of_face[key] = node_id
    return node_of_face


def load_user_cannot_links(labels_dir, node_of_face: dict):
    """-> (node pairs, impure face_keys). Empty when no label store exists."""
    pairs, impure = [], set()
    if labels_dir is None:
        return pairs, impure
    path = Path(labels_dir) / "derived" / "cannot_links_face_face.csv"
    if not path.exists():
        return pairs, impure
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            node_a = node_of_face.get(row["face_key_a"])
            node_b = node_of_face.get(row["face_key_b"])
            if node_a is None or node_b is None:
                continue  # face not in this embed set (filtered/missing)
            if node_a == node_b:
                # Both faces live in one track: the track itself is impure.
                impure.add(row["face_key_a"])
                impure.add(row["face_key_b"])
            else:
                pairs.append((node_a, node_b))
    return pairs, impure


def resolve_constraints(nodes_dir, tracks_dir, node_id_of_track: dict,
                        labels_dir=None):
    """-> (all cannot-link node pairs, impure face_keys, provenance counts)."""
    same_frame = load_same_frame_pairs(nodes_dir)
    node_of_face = load_node_of_face(tracks_dir, node_id_of_track)
    user_pairs, impure = load_user_cannot_links(labels_dir, node_of_face)
    counts = {"same_frame": len(same_frame), "user": len(user_pairs),
              "track_impure_faces": len(impure)}
    return same_frame + user_pairs, impure, counts
