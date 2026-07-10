"""Cluster-centroid neighbor suggestions: the merge-discovery layer.

Over-splitting is the pipeline's dominant error direction (precision-first
thresholds put the same person in several clusters per event family), and
spotting splits by eye across a 23K-row index is hopeless. This precomputes,
for every cluster, its nearest other clusters by centroid cosine; the review
server renders them as the "similar clusters" strip on group pages and
corpus-wide on /merges. identity.merge stays the only unification mechanism
— this is discovery, not action.

Same-person-split pairs measured on run_002 sit at centroid cos 0.79-0.86
(the genuine cross-file range), so min_cos=0.70 keeps every plausible split
visible without flooding the strip with strangers.
"""

from __future__ import annotations

import numpy as np

from unlabeled_media_tagger.clustering.knn_graph import exact_topk_sample

DEFAULT_K = 6
DEFAULT_MIN_COS = 0.70


def cluster_neighbors(members_of_cluster: dict, embeddings: np.ndarray,
                      events_of_cluster: dict | None = None, *,
                      k: int = DEFAULT_K,
                      min_cos: float = DEFAULT_MIN_COS) -> list:
    """Top-k nearest clusters by centroid cosine, most similar first.

    members_of_cluster: cluster_id -> node ids (row indices into
    embeddings). events_of_cluster: cluster_id -> set of event_paths, for
    the shared-events annotation (same person, same shoot). Returns dict
    rows (cluster_id, rank, neighbor_cluster_id, cosine, shared_events)
    keeping only neighbors with cosine >= min_cos.
    """
    cluster_ids = sorted(members_of_cluster, key=str)
    n = len(cluster_ids)
    if n < 2:
        return []

    centroids = np.empty((n, embeddings.shape[1]), dtype=np.float32)
    for i, cluster_id in enumerate(cluster_ids):
        vec = embeddings[members_of_cluster[cluster_id]].sum(axis=0)
        centroids[i] = vec / max(float(np.linalg.norm(vec)), 1e-12)

    top = exact_topk_sample(centroids, np.arange(n), min(k, n - 1))
    rows = []
    for i, cluster_id in enumerate(cluster_ids):
        sims = centroids[top[i]] @ centroids[i]
        events = (events_of_cluster or {}).get(cluster_id, set())
        for rank, (j, cos) in enumerate(zip(top[i].tolist(),
                                            sims.tolist()), start=1):
            if cos < min_cos:
                break  # neighbors are ordered most-similar first
            neighbor_id = cluster_ids[j]
            shared = len(events
                         & (events_of_cluster or {}).get(neighbor_id, set()))
            rows.append({
                "cluster_id": cluster_id,
                "rank": rank,
                "neighbor_cluster_id": neighbor_id,
                "cosine": round(float(cos), 4),
                "shared_events": shared,
            })
    return rows
