"""Offline reclustering from cached face embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from unlabeled_media_tagger.pipeline.run import flatten_clusters, write_cluster_csv


class UnionFind:
    """Disjoint-set structure for cluster merges."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root

        keep, drop = sorted((left_root, right_root))
        self.parent[drop] = keep
        return keep


def load_face_records(processed_dir: str) -> list[dict]:
    """Load cached face records from processed JSON files."""
    records = []
    for path in sorted(Path(processed_dir).glob("*.json")):
        with path.open() as json_file:
            payload = json.load(json_file)
        records.extend(payload.get("face_records", []))

    return records


def normalize_embeddings(face_records: list[dict]) -> np.ndarray:
    """Return L2-normalized embedding matrix for face records."""
    embeddings = np.asarray(
        [[float(value) for value in face["embedding"]] for face in face_records],
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix for normalized embeddings."""
    return embeddings @ embeddings.T


def candidate_edges(similarities: np.ndarray, threshold: float) -> list[tuple[float, int, int]]:
    """Return pair edges above a threshold, strongest first."""
    rows, cols = np.where(np.triu(similarities, k=1) >= threshold)
    edges = [(float(similarities[i, j]), int(i), int(j)) for i, j in zip(rows, cols)]
    return sorted(edges, reverse=True)


def recluster_faces(
    face_records: list[dict],
    edge_threshold: float = 0.74,
    merge_min_similarity: float = 0.62,
    merge_mean_similarity: float = 0.70,
    prevent_same_frame_merge: bool = True,
) -> dict[int, list[dict]]:
    """
    Recluster cached face embeddings with graph edges and merge quality gates.

    Unlike the online centroid pass, this uses all embeddings at once. Candidate
    merges are considered strongest-first and accepted only when every cross-pair
    is above ``merge_min_similarity`` and the average cross similarity is above
    ``merge_mean_similarity``. The same-frame constraint prevents assigning two
    simultaneous faces from one frame to the same person cluster.
    """
    if not face_records:
        return {}

    embeddings = normalize_embeddings(face_records)
    similarities = pairwise_similarities(embeddings)
    edges = candidate_edges(similarities, edge_threshold)
    union_find = UnionFind(len(face_records))
    members = {index: {index} for index in range(len(face_records))}
    frame_sets = {
        index: {face_records[index].get("frame_path")}
        for index in range(len(face_records))
        if face_records[index].get("frame_path")
    }

    for _, left, right in edges:
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue

        left_members = members[left_root]
        right_members = members[right_root]
        if prevent_same_frame_merge and frame_sets.get(left_root, set()).intersection(
            frame_sets.get(right_root, set())
        ):
            continue

        cross = similarities[
            np.ix_(list(left_members), list(right_members))
        ]
        if float(cross.min()) < merge_min_similarity:
            continue
        if float(cross.mean()) < merge_mean_similarity:
            continue

        new_root = union_find.union(left_root, right_root)
        old_root = right_root if new_root == left_root else left_root
        members[new_root] = members[left_root] | members[right_root]
        members.pop(old_root, None)
        frame_sets[new_root] = frame_sets.get(left_root, set()) | frame_sets.get(
            right_root, set()
        )
        frame_sets.pop(old_root, None)

    clusters = {}
    for new_cluster_id, root in enumerate(sorted(members)):
        cluster_members = sorted(members[root])
        clusters[new_cluster_id] = [
            {
                **face_records[index],
                "cluster_id": new_cluster_id,
                "cluster_label": f"person_{new_cluster_id:03d}",
                "similarity_to_cluster": round(
                    similarity_to_cluster(index, cluster_members, similarities),
                    6,
                ),
            }
            for index in cluster_members
        ]

    return clusters


def similarity_to_cluster(
    index: int,
    cluster_members: list[int],
    similarities: np.ndarray,
) -> float:
    """Return mean similarity from one face to the rest of its cluster."""
    others = [member for member in cluster_members if member != index]
    if not others:
        return 1.0

    return float(similarities[index, others].mean())


def build_parser() -> argparse.ArgumentParser:
    """Build the reclustering CLI parser."""
    parser = argparse.ArgumentParser(
        description="Recluster cached face embeddings without rerunning detection."
    )
    parser.add_argument(
        "--processed-dir",
        default="outputs/pipeline/processed",
        help="Directory containing per-file processed JSON caches",
    )
    parser.add_argument(
        "--csv-out",
        default="outputs/recluster/face_clusters_reclustered.csv",
        help="Output CSV for reclustered face rows",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.74,
        help="Minimum pair similarity to consider a merge edge",
    )
    parser.add_argument(
        "--merge-min-similarity",
        type=float,
        default=0.62,
        help="Minimum cross-pair similarity allowed when merging clusters",
    )
    parser.add_argument(
        "--merge-mean-similarity",
        type=float,
        default=0.70,
        help="Minimum average cross-cluster similarity required for a merge",
    )
    parser.add_argument(
        "--allow-same-frame-merge",
        action="store_true",
        help="Allow one cluster to contain multiple faces from the same frame",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    face_records = load_face_records(args.processed_dir)
    clusters = recluster_faces(
        face_records,
        edge_threshold=args.edge_threshold,
        merge_min_similarity=args.merge_min_similarity,
        merge_mean_similarity=args.merge_mean_similarity,
        prevent_same_frame_merge=not args.allow_same_frame_merge,
    )
    rows = flatten_clusters(clusters)
    write_cluster_csv(Path(args.csv_out), rows)

    print(f"Loaded face records: {len(face_records)}")
    print(f"Clusters: {len(clusters)}")
    print(f"Wrote CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
