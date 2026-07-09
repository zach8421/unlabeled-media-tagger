"""Regression pins for defects found in the 2026-07-06 adversarial review."""

from __future__ import annotations

import numpy as np
import pytest

from unlabeled_media_tagger.review.store import LabelStore


@pytest.fixture()
def store(tmp_path):
    return LabelStore(tmp_path / "labels")


def test_identity_seq_never_reused_after_revoked_create(store):
    """A revoked identity.create must not release its sequence number:
    still-active confirms reference it, and reuse would merge two people."""
    create = store.append("identity.create", {"identity_id": "identity_0001",
                                              "name": "Alice"})
    store.append("cluster.confirm", {
        "run_id": "r", "cluster_id": 1, "identity_id": "identity_0001",
        "member_face_keys": ["fa/a.jpg"]})
    store.append("revoke", {"target_event_id": create["event_id"]})
    assert store.next_identity_id() == "identity_0002"  # not _0001 again


def test_identityless_remove_keeps_other_identity_assignment(store):
    """face.remove with identity None (unconfirmed cluster in a NEW run) must
    not clobber an assignment an earlier run's confirm created."""
    store.append("identity.create", {"identity_id": "identity_0001",
                                     "name": "Sarah"})
    store.append("cluster.confirm", {
        "run_id": "run1", "cluster_id": 3, "identity_id": "identity_0001",
        "member_face_keys": ["fa/a.jpg"]})
    store.append("face.remove", {
        "face_key": "fa/a.jpg", "run_id": "run2", "cluster_id": 9,
        "identity_id": None, "context_face_keys": ["fb/b.jpg"]})
    state = store.replay()
    assert state.assignments["fa/a.jpg"]["identity_id"] == "identity_0001"
    # The cannot-link signal is still captured.
    assert ("fa/a.jpg", "fb/b.jpg") in state.cannot_link_face_face
    assert ("run2", "9", "fa/a.jpg") in state.removed_from_cluster


def test_reject_after_confirm_withdraws_confirm_assignments(store):
    store.append("identity.create", {"identity_id": "identity_0001"})
    store.append("cluster.confirm", {
        "run_id": "r", "cluster_id": 5, "identity_id": "identity_0001",
        "member_face_keys": ["fa/a.jpg", "fb/b.jpg"]})
    # A manual assign of another face must survive the reject.
    store.append("face.assign", {"face_key": "fc/c.jpg",
                                 "identity_id": "identity_0001"})
    store.append("cluster.reject", {
        "run_id": "r", "cluster_id": 5,
        "member_face_keys": ["fa/a.jpg", "fb/b.jpg"]})
    state = store.replay()
    assert "fa/a.jpg" not in state.assignments
    assert "fb/b.jpg" not in state.assignments
    assert state.assignments["fc/c.jpg"]["identity_id"] == "identity_0001"
    assert state.cluster_review[("r", "5")] == "rejected"


def test_confirm_after_reject_wins_precedence(store):
    store.append("cluster.reject", {"run_id": "r", "cluster_id": 5,
                                    "member_face_keys": ["fa/a.jpg"]})
    store.append("identity.create", {"identity_id": "identity_0001"})
    store.append("cluster.confirm", {
        "run_id": "r", "cluster_id": 5, "identity_id": "identity_0001",
        "member_face_keys": ["fa/a.jpg"]})
    state = store.replay()
    assert state.cluster_review[("r", "5")] == "confirmed"
    assert state.assignments["fa/a.jpg"]["identity_id"] == "identity_0001"


def test_event_ids_unique_under_burst(store):
    ids = {store.append("face.assign",
                        {"face_key": f"f/{i}.jpg",
                         "identity_id": "identity_0001"})["event_id"]
           for i in range(2000)}
    assert len(ids) == 2000


def test_small_active_set_recall_does_not_crash():
    """Pilot runs can have fewer active nodes than k (regression)."""
    from unlabeled_media_tagger.clustering.knn_graph import (
        ann_recall,
        build_hnsw_index,
        knn_edges,
    )

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 64)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = build_hnsw_index(emb)
    recall = ann_recall(emb, index, k=50)  # k > n: must clamp, not raise
    assert 0.0 <= recall <= 1.0
    edges = knn_edges(emb, k=50, threshold=-1.0, index=index)
    assert len(edges[0]) > 0

    tiny = emb[:1]
    tiny_index = build_hnsw_index(tiny)
    assert ann_recall(tiny, tiny_index, k=50) == 1.0  # degenerate: no pairs
