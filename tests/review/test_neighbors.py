"""Cluster-neighbor suggestions: ranking, threshold, shared events, edges."""

from __future__ import annotations

import numpy as np

from unlabeled_media_tagger.review.neighbors import cluster_neighbors


def _unit(vec):
    return vec / np.linalg.norm(vec)


def _fixture():
    """Clusters a and b share a direction; c is orthogonal to both."""
    rng = np.random.default_rng(0)
    base = _unit(rng.normal(size=32))
    noise = rng.normal(size=32)
    ortho = _unit(noise - (noise @ base) * base)
    emb = np.stack([
        _unit(base + 0.05 * rng.normal(size=32)),   # 0: a
        _unit(base + 0.05 * rng.normal(size=32)),   # 1: a
        _unit(base + 0.05 * rng.normal(size=32)),   # 2: b
        _unit(base + 0.05 * rng.normal(size=32)),   # 3: b
        _unit(ortho + 0.05 * rng.normal(size=32)),  # 4: c
        _unit(ortho + 0.05 * rng.normal(size=32)),  # 5: c
    ]).astype(np.float32)
    members = {"a": [0, 1], "b": [2, 3], "c": [4, 5]}
    events = {"a": {"e1", "e2"}, "b": {"e2", "e3"}, "c": {"e9"}}
    return members, emb, events


def test_ranking_threshold_and_shared_events():
    members, emb, events = _fixture()
    rows = cluster_neighbors(members, emb, events, k=2, min_cos=0.5)

    a_rows = [r for r in rows if r["cluster_id"] == "a"]
    assert a_rows[0]["neighbor_cluster_id"] == "b"
    assert a_rows[0]["cosine"] > 0.95
    assert a_rows[0]["rank"] == 1
    assert a_rows[0]["shared_events"] == 1  # e2
    # c is near-orthogonal: below min_cos, filtered from a's list entirely
    assert all(r["neighbor_cluster_id"] != "c" for r in a_rows)
    # symmetric direction is present too
    b_rows = [r for r in rows if r["cluster_id"] == "b"]
    assert b_rows[0]["neighbor_cluster_id"] == "a"
    # c keeps no neighbors at this threshold
    assert not [r for r in rows if r["cluster_id"] == "c"]


def test_min_cos_zero_keeps_everything_ranked():
    members, emb, events = _fixture()
    rows = cluster_neighbors(members, emb, events, k=5, min_cos=-1.0)
    a_rows = [r for r in rows if r["cluster_id"] == "a"]
    assert [r["rank"] for r in a_rows] == [1, 2]  # k clamps to n-1
    assert a_rows[0]["cosine"] >= a_rows[1]["cosine"]
    assert a_rows[1]["shared_events"] == 0  # a and c share nothing


def test_degenerate_inputs():
    members, emb, _ = _fixture()
    assert cluster_neighbors({"a": [0, 1]}, emb) == []  # single cluster
    assert cluster_neighbors({}, emb) == []
    # events omitted -> shared_events defaults to 0
    rows = cluster_neighbors(members, emb, None, k=1, min_cos=0.5)
    assert all(r["shared_events"] == 0 for r in rows)
