"""Correctness pins for the sparse clustering engine.

The load-bearing test is dense parity: with a complete edge set and the exact
min gate, sparse_cluster must reproduce pipeline/recluster.py::recluster_faces
partition-for-partition — that function is the validated reference
implementation whose semantics we ported.
"""

from __future__ import annotations

import numpy as np
import pytest

from unlabeled_media_tagger.clustering.sparse_cluster import sparse_cluster


def _unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _edges_from_dense(embeddings, threshold):
    sims = embeddings @ embeddings.T
    i, j = np.where(np.triu(sims, k=1) >= threshold)
    return i.astype(np.int64), j.astype(np.int64), sims[i, j].astype(np.float32)


def _vectors_from_gram(gram):
    """Exact unit vectors realizing a target cosine (Gram) matrix."""
    gram = np.asarray(gram, dtype=np.float64)
    chol = np.linalg.cholesky(gram)  # raises if targets are geometrically infeasible
    return chol.astype(np.float32)


def _partition(labels_or_clusters):
    """Normalize either engine's output to a set of frozensets of indices."""
    if isinstance(labels_or_clusters, dict):  # recluster_faces output
        return {frozenset(face["idx"] for face in faces)
                for faces in labels_or_clusters.values()}
    labels = labels_or_clusters
    return {frozenset(np.where(labels == c)[0].tolist())
            for c in set(labels.tolist())}


# ---------------------------------------------------------------- dense parity


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_parity_with_dense_recluster_on_random_mixtures(seed):
    from unlabeled_media_tagger.pipeline.recluster import (
        normalize_embeddings,
        recluster_faces,
    )

    rng = np.random.default_rng(seed)
    n_people, faces_per, dim = 12, 8, 64
    raw, frames = [], []
    for person in range(n_people):
        base = rng.standard_normal(dim)
        for k in range(faces_per):
            raw.append(base + 0.15 * rng.standard_normal(dim))
            # Round-robin frames: some same-person faces share a frame, which
            # must block their merge in both engines.
            frames.append(f"frame_{person % 3}_{k % 4}.jpg")
    # Noise singletons.
    for k in range(10):
        raw.append(rng.standard_normal(dim) * 2)
        frames.append(f"noise_{k}.jpg")

    records = [{"embedding": list(map(float, vec)), "frame_path": frame,
                "idx": idx}
               for idx, (vec, frame) in enumerate(zip(raw, frames))]
    dense_clusters = recluster_faces(
        records, edge_threshold=0.74,
        merge_min_similarity=0.62, merge_mean_similarity=0.70,
    )

    embeddings = normalize_embeddings(records)  # identical float32 rounding
    edges = _edges_from_dense(embeddings, 0.74)
    same_frame = [(a, b) for a in range(len(records))
                  for b in range(a + 1, len(records))
                  if frames[a] == frames[b]]
    result = sparse_cluster(
        embeddings, *edges,
        cannot_link_pairs=same_frame,
        merge_min=0.62, merge_mean=0.70,
        min_cluster_size=1, min_gate="exact", worst_member_gate=False,
    )

    assert _partition(result.labels) == _partition(dense_clusters)


# ------------------------------------------------------------------ constraints


def test_cannot_link_blocks_merge_transitively():
    # Three near-identical vectors; 0-1 cannot link. 0 merges 2 first
    # (strongest edge), after which 1 must be locked out of the whole cluster.
    emb = _unit_rows(_vectors_from_gram([
        [1.0, 0.97, 0.98],
        [0.97, 1.0, 0.96],
        [0.98, 0.96, 1.0],
    ]))
    edges = _edges_from_dense(emb, 0.5)
    result = sparse_cluster(
        emb, *edges, cannot_link_pairs=[(0, 1)],
        merge_min=0.5, merge_mean=0.5, min_cluster_size=1,
    )
    assert _partition(result.labels) == {frozenset({0, 2}), frozenset({1})}
    assert result.stats["g0_blocked"] >= 1


def _from_angles(*degrees):
    """Exact unit vectors on the 2-D circle: cos(i,j) = cos(angle_i - angle_j)."""
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return np.stack([np.cos(radians), np.sin(radians)]).T.astype(np.float32)


def test_mean_gate_blocks_chaining():
    # 0-1 similar (36 deg), 1-2 similar (36 deg), 0-2 dissimilar (72 deg): the
    # exact all-pairs mean must stop the chain even though each individual
    # edge clears the edge threshold.
    emb = _from_angles(0, 36, 72)
    edges = _edges_from_dense(emb, 0.75)  # keeps (0,1) and (1,2), not (0,2)
    result = sparse_cluster(
        emb, *edges, merge_min=0.05, merge_mean=0.70,
        min_cluster_size=1, min_gate="exact", worst_member_gate=False,
    )
    # {0,1} merge (mean .809); edge (1,2): cross mean (.309+.809)/2 = .559 < .70.
    assert _partition(result.labels) == {frozenset({0, 1}), frozenset({2})}
    assert result.stats["g1_blocked"] == 1


# -------------------------------------------------------- memoization semantics


def test_rejected_pair_reevaluated_after_root_grows():
    """A grown cluster must be re-gated, not stale-skipped (version keys).

    Edge sims are FABRICATED to control processing order (the engine takes the
    edge list as given; with min_gate='exact' the sims only set order), while
    the gate math runs on the real 2-D geometry:
      node angles 0deg, 8deg, 44deg, 20deg ->
      (0,1)=.990  (0,3)=.940  (1,3)=.978  (0,2)=.719  (1,2)=.809  (3,2)=.913
    """
    emb = _from_angles(0, 8, 44, 20)
    edges_i = np.array([0, 1, 1, 3], dtype=np.int64)
    edges_j = np.array([1, 2, 3, 2], dtype=np.int64)
    edges_sim = np.array([0.99, 0.98, 0.97, 0.96], dtype=np.float32)
    result = sparse_cluster(
        emb, edges_i, edges_j, edges_sim,
        merge_min=0.50, merge_mean=0.80,
        min_cluster_size=1, min_gate="exact", worst_member_gate=False,
    )
    # (0,1) merge; (1,2): {0,1}x{2} mean (.719+.809)/2 = .764 < .80 REJECT;
    # (1,3): mean (.940+.978)/2 = .959 merge -> {0,1,3}, version bump;
    # (3,2): re-evaluated (new version): mean (.719+.809+.913)/3 = .814 >= .80
    # -> merges. A stale root-only memo would have skipped it (2 clusters).
    assert _partition(result.labels) == {frozenset({0, 1, 2, 3})}
    assert result.stats["g1_blocked"] == 1
    assert result.stats["merges"] == 3


def test_memo_skips_weaker_edge_between_unchanged_roots():
    # Real geometry: pairs {0@0,1@8} and {2@60,3@68}; cross mean
    # (cos60+cos68+cos52+cos60)/4 = .498 < .80. Fabricated order puts two
    # cross edges after both pair merges; the second must be a memo skip.
    emb = _from_angles(0, 8, 60, 68)
    edges_i = np.array([0, 2, 0, 1], dtype=np.int64)
    edges_j = np.array([1, 3, 2, 3], dtype=np.int64)
    edges_sim = np.array([0.99, 0.98, 0.70, 0.65], dtype=np.float32)
    result = sparse_cluster(
        emb, edges_i, edges_j, edges_sim,
        merge_min=0.30, merge_mean=0.80,
        min_cluster_size=1, min_gate="exact", worst_member_gate=False,
    )
    assert _partition(result.labels) == {frozenset({0, 1}), frozenset({2, 3})}
    assert result.stats["g1_blocked"] == 1
    assert result.stats["memo_skipped"] == 1


# ------------------------------------------------------------------ misc pins


def test_min_cluster_size_leaves_small_groups_unassigned():
    emb = _unit_rows(_vectors_from_gram([
        [1.0, 0.95, 0.0],
        [0.95, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))
    edges = _edges_from_dense(emb, 0.7)
    result = sparse_cluster(emb, *edges, merge_min=0.5, merge_mean=0.5,
                            min_cluster_size=3)
    assert result.n_clusters == 0
    assert (result.labels == -1).all()
    assert result.stats["unassigned"] == 3


def test_similarity_to_cluster_is_cos_to_centroid():
    emb = _unit_rows(_vectors_from_gram([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0],
    ]))
    edges = _edges_from_dense(emb, 0.5)
    result = sparse_cluster(emb, *edges, merge_min=0.5, merge_mean=0.5,
                            min_cluster_size=3)
    assert result.n_clusters == 1
    centroid = emb.sum(axis=0)
    centroid /= np.linalg.norm(centroid)
    np.testing.assert_allclose(result.similarity_to_cluster, emb @ centroid,
                               rtol=1e-5)


def test_determinism():
    rng = np.random.default_rng(42)
    emb = _unit_rows(rng.standard_normal((200, 32)))
    edges = _edges_from_dense(emb, 0.3)
    first = sparse_cluster(emb, *edges, merge_min=0.2, merge_mean=0.25,
                           min_cluster_size=2)
    second = sparse_cluster(emb, *edges, merge_min=0.2, merge_mean=0.25,
                            min_cluster_size=2)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_sparse_min_gate_and_worst_member_gate_run():
    """Production path (sparse G2 + G3) executes and stays conservative."""
    rng = np.random.default_rng(3)
    people = []
    for p in range(5):
        base = rng.standard_normal(64)
        people.extend(base + 0.1 * rng.standard_normal(64) for _ in range(6))
    emb = _unit_rows(np.stack(people))
    edges = _edges_from_dense(emb, 0.74)
    result = sparse_cluster(emb, *edges, merge_min=0.62, merge_mean=0.70,
                            min_cluster_size=2, min_gate="sparse",
                            worst_member_gate=True)
    # Every emitted cluster must be person-pure (each person is 6 consecutive).
    for cluster in _partition(result.labels[result.labels >= 0].reshape(-1)):
        pass  # partition of filtered labels loses indices; check directly:
    for cluster_id in range(result.n_clusters):
        idx = np.where(result.labels == cluster_id)[0]
        assert len({int(i) // 6 for i in idx}) == 1, f"impure cluster {idx}"
