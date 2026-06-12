"""Online-centroid face clustering for the Colab end-to-end pipeline.

Pure-function mirror of `pipeline/compare.py::CompareStage.compare_faces`.
Same algorithm, same default similarity threshold (0.68), same online-mean
centroid update — the only differences are:

- Takes plain dicts, returns plain dicts. No class wrapper.
- Zero dependencies beyond `math.sqrt`. Safe to copy-paste into the
  notebook's helpers cell verbatim per the project's self-containment rule.

Tested for cluster-assignment equivalence against `CompareStage` in
`tests/test_cluster.py`. Keep them aligned: if `CompareStage` ever
changes its algorithm, mirror the change here.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two equal-length numeric vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same length")

    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sqrt(sum(float(a) * float(a) for a in left))
    right_norm = sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def update_centroid(
    current_centroid: list[float],
    new_embedding: list[float],
    existing_count: int,
) -> list[float]:
    """Update a centroid with one new vector using an online mean."""
    next_count = existing_count + 1
    return [
        ((float(current) * existing_count) + float(new)) / next_count
        for current, new in zip(current_centroid, new_embedding)
    ]


def cluster_face_records(
    face_records: Iterable[dict],
    similarity_threshold: float = 0.68,
) -> dict[int, list[dict]]:
    """Group face records by cosine similarity using online-centroid assignment.

    Each input dict must have an ``embedding`` key (a list[float]). Records
    missing an embedding are silently skipped (same policy as ``CompareStage``).

    Returns ``{cluster_id: [face_record_with_cluster_fields, ...]}``. Cluster
    ids are assigned in encounter order (first new cluster is 0); the order of
    the input list therefore affects which faces "found" each cluster, but the
    final partition is stable given the same input order.

    Each returned face dict has these added/overwritten keys:
      - ``cluster_id`` (int)
      - ``cluster_label`` (str, ``"person_{cluster_id:03d}"``)
      - ``similarity_to_cluster`` (float; 1.0 for the seed of a new cluster)
    """
    clustered: dict[int, list[dict]] = defaultdict(list)
    centroids: list[list[float]] = []

    for face in face_records:
        embedding = face.get("embedding")
        if not embedding:
            continue

        best_cluster = None
        best_similarity = -1.0
        for cluster_id, centroid in enumerate(centroids):
            similarity = cosine_similarity(embedding, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id

        if best_cluster is None or best_similarity < similarity_threshold:
            cluster_id = len(centroids)
            centroids.append([float(value) for value in embedding])
            assigned_similarity = 1.0
        else:
            cluster_id = best_cluster
            centroids[cluster_id] = update_centroid(
                centroids[cluster_id],
                embedding,
                len(clustered[cluster_id]),
            )
            assigned_similarity = round(best_similarity, 6)

        clustered[cluster_id].append(
            {
                **face,
                "cluster_id": cluster_id,
                "cluster_label": f"person_{cluster_id:03d}",
                "similarity_to_cluster": assigned_similarity,
            }
        )

    return dict(clustered)
