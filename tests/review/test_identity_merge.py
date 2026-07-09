"""Behavior pins for identity.merge (same-person unification, undoable)."""

from __future__ import annotations

import pytest

from unlabeled_media_tagger.review.store import LabelStore


@pytest.fixture()
def store(tmp_path):
    store = LabelStore(tmp_path / "labels")
    for i, (name, faces) in enumerate(
            [("Sarah A", ["f1/a.jpg", "f2/b.jpg"]),
             ("Sarah B", ["f3/c.jpg"]),
             ("Marcus", ["f4/d.jpg"])], start=1):
        store.append("identity.create",
                     {"identity_id": f"identity_{i:04d}", "name": name})
        store.append("cluster.confirm", {
            "run_id": "r", "cluster_id": i, "identity_id": f"identity_{i:04d}",
            "member_face_keys": faces})
    return store


def test_merge_pools_assignments_under_target(store):
    store.append("identity.merge", {"source_identity_id": "identity_0002",
                                    "target_identity_id": "identity_0001"})
    state = store.replay()
    assert state.canonical_identity("identity_0002") == "identity_0001"
    assert state.assignments["f3/c.jpg"]["identity_id"] == "identity_0001"
    assert state.assignments["f1/a.jpg"]["identity_id"] == "identity_0001"
    assert state.assignments["f4/d.jpg"]["identity_id"] == "identity_0003"
    assert state.identities["identity_0002"]["merged_into"] == "identity_0001"
    # The source identity record survives (non-destructive reference layer).
    assert state.identities["identity_0002"]["name"] == "Sarah B"


def test_revoking_merge_restores_split_exactly(store):
    before = store.replay()
    merge = store.append("identity.merge",
                         {"source_identity_id": "identity_0002",
                          "target_identity_id": "identity_0001"})
    store.append("revoke", {"target_event_id": merge["event_id"]})
    after = store.replay()
    assert after.assignments == before.assignments
    assert after.merged_into == {}
    assert "merged_into" not in after.identities["identity_0002"]


def test_merge_chain_resolves_transitively(store):
    store.append("identity.merge", {"source_identity_id": "identity_0002",
                                    "target_identity_id": "identity_0001"})
    store.append("identity.merge", {"source_identity_id": "identity_0001",
                                    "target_identity_id": "identity_0003"})
    state = store.replay()
    assert state.canonical_identity("identity_0002") == "identity_0003"
    assert state.canonical_identity("identity_0001") == "identity_0003"
    assert all(a["identity_id"] == "identity_0003"
               for a in state.assignments.values())


def test_merge_cycle_does_not_hang(store):
    store.append("identity.merge", {"source_identity_id": "identity_0001",
                                    "target_identity_id": "identity_0002"})
    store.append("identity.merge", {"source_identity_id": "identity_0002",
                                    "target_identity_id": "identity_0001"})
    state = store.replay()  # must terminate; endpoints resolve consistently
    assert state.canonical_identity("identity_0001") in (
        "identity_0001", "identity_0002")


def test_cannot_links_follow_the_merge(store):
    store.append("face.remove", {"face_key": "fx/z.jpg",
                                 "identity_id": "identity_0002",
                                 "context_face_keys": []})
    store.append("identity.merge", {"source_identity_id": "identity_0002",
                                    "target_identity_id": "identity_0001"})
    state = store.replay()
    # "Not Sarah B" + "Sarah B is Sarah A" => not Sarah A (canonical).
    assert ("fx/z.jpg", "identity_0001") in state.cannot_link_face_identity


def test_growth_verified_set_pools_merged_identities(store, tmp_path):
    """run_growth's verified set must treat merged identities as one."""
    store.append("identity.merge", {"source_identity_id": "identity_0002",
                                    "target_identity_id": "identity_0001"})
    state = store.replay()
    verified = sorted({
        iid for iid in (state.canonical_identity(c["identity_id"])
                        for c in state.confirms)
        if iid in state.identities and not state.identities[iid]["retired"]})
    assert verified == ["identity_0001", "identity_0003"]
