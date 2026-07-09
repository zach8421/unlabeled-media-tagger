"""Behavior pins for the append-only label store."""

from __future__ import annotations

import json

import pytest

from unlabeled_media_tagger.review.store import LabelStore, replay_events


@pytest.fixture()
def store(tmp_path):
    return LabelStore(tmp_path / "labels")


def _confirm(store, identity="identity_0001", run="run_a", cluster=7,
             members=("f1/a.jpg", "f1/b.jpg", "f2/c.jpg")):
    store.append("identity.create", {"identity_id": identity,
                                     "name": "Sarah", "tracked": True})
    return store.append("cluster.confirm", {
        "run_id": run, "cluster_id": cluster, "identity_id": identity,
        "member_face_keys": list(members),
    })


def test_append_replay_round_trip(store):
    _confirm(store)
    state = store.replay()
    assert state.identities["identity_0001"]["name"] == "Sarah"
    assert state.assignments["f1/a.jpg"]["identity_id"] == "identity_0001"
    assert len(state.assignments) == 3
    assert state.confirms[0]["n_members"] == 3


def test_face_remove_unassigns_and_creates_cannot_links(store):
    _confirm(store)
    store.append("face.remove", {
        "face_key": "f1/b.jpg", "run_id": "run_a", "cluster_id": 7,
        "identity_id": "identity_0001",
        "context_face_keys": ["f1/a.jpg", "f2/c.jpg"],
    })
    state = store.replay()
    assert "f1/b.jpg" not in state.assignments
    assert ("f1/b.jpg", "identity_0001") in state.cannot_link_face_identity
    assert ("f1/a.jpg", "f1/b.jpg") in state.cannot_link_face_face
    assert ("f1/b.jpg", "f2/c.jpg") in state.cannot_link_face_face


def test_revoke_restores_prior_state(store):
    _confirm(store)
    before = store.replay()
    remove = store.append("face.remove", {
        "face_key": "f1/b.jpg", "identity_id": "identity_0001",
        "context_face_keys": ["f1/a.jpg"],
    })
    store.append("revoke", {"target_event_id": remove["event_id"]})
    after = store.replay()
    assert after.assignments == before.assignments
    assert after.cannot_link_face_face == set()
    assert after.cannot_link_face_identity == set()


def test_revoke_of_revoke_reapplies_event(store):
    _confirm(store)
    remove = store.append("face.remove", {
        "face_key": "f1/b.jpg", "identity_id": "identity_0001",
        "context_face_keys": [],
    })
    undo = store.append("revoke", {"target_event_id": remove["event_id"]})
    store.append("revoke", {"target_event_id": undo["event_id"]})
    state = store.replay()
    assert "f1/b.jpg" not in state.assignments  # remove is active again


def test_remove_after_confirm_ordering(store):
    _confirm(store)
    store.append("face.remove", {"face_key": "f1/a.jpg",
                                 "identity_id": "identity_0001",
                                 "context_face_keys": []})
    # A later confirm of the same cluster re-asserts membership.
    store.append("cluster.confirm", {
        "run_id": "run_a", "cluster_id": 7, "identity_id": "identity_0001",
        "member_face_keys": ["f1/a.jpg"],
    })
    state = store.replay()
    assert "f1/a.jpg" in state.assignments
    # But the cannot-link record persists for growth to honor.
    assert ("f1/a.jpg", "identity_0001") in state.cannot_link_face_identity


def test_context_face_cap(store):
    event = store.append("face.remove", {
        "face_key": "f/x.jpg",
        "context_face_keys": [f"f/{i}.jpg" for i in range(1000)],
    })
    assert len(event["payload"]["context_face_keys"]) == 300


def test_unknown_event_type_rejected(store):
    with pytest.raises(ValueError):
        store.append("cluster.explode", {})


def test_next_identity_id_monotonic(store):
    assert store.next_identity_id() == "identity_0001"
    store.append("identity.create", {"identity_id": "identity_0001"})
    store.append("identity.create", {"identity_id": "identity_0007"})
    assert store.next_identity_id() == "identity_0008"


def test_replay_skips_torn_last_line(store):
    _confirm(store)
    with store.events_path.open("ab") as fh:
        fh.write(b'{"event_id": "ev_torn", "type": "face.re')  # crash mid-write
    state = store.replay()  # must not raise
    assert len(state.assignments) == 3
    report = store.verify()
    assert report["torn_tail"] is True
    assert report["bad_lines"] == [report["lines"]]
    assert report["replay_ok"] is True


def test_rebuild_derived_writes_consistent_views(store, tmp_path):
    _confirm(store)
    store.append("face.remove", {
        "face_key": "f1/b.jpg", "identity_id": "identity_0001",
        "context_face_keys": ["f1/a.jpg"],
    })
    store.rebuild_derived()
    derived = store.derived_dir
    assignments = (derived / "assignments.csv").read_text().splitlines()
    assert len(assignments) == 3  # header + 2 remaining faces
    identities = (derived / "identities.csv").read_text().splitlines()
    assert "identity_0001,Sarah,true,false" in identities[1]
    cl_fi = (derived / "cannot_links_face_identity.csv").read_text()
    assert "f1/b.jpg,identity_0001" in cl_fi
    meta = json.loads((derived / "build_meta.json").read_text())
    assert meta["log_lines"] == 3
    assert meta["n_events_active"] == 3


def test_growth_actor_marks_source(store):
    store.append("identity.create", {"identity_id": "identity_0001"})
    store.append("face.assign", {"face_key": "f9/z.jpg",
                                 "identity_id": "identity_0001",
                                 "sim": 0.83, "margin": 0.09},
                 actor="growth")
    state = store.replay()
    assert state.assignments["f9/z.jpg"]["source"] == "growth"


def test_thousand_mixed_events_with_revokes_verify_clean(store):
    events = []
    for i in range(100):
        store.append("identity.create",
                     {"identity_id": f"identity_{i:04d}", "name": f"p{i}"})
        confirm = store.append("cluster.confirm", {
            "run_id": "r", "cluster_id": i,
            "identity_id": f"identity_{i:04d}",
            "member_face_keys": [f"file{i}/{j}.jpg" for j in range(5)],
        })
        events.append(confirm)
        for j in range(3):
            store.append("face.remove", {
                "face_key": f"file{i}/{j}.jpg",
                "identity_id": f"identity_{i:04d}",
                "context_face_keys": [f"file{i}/{k}.jpg" for k in range(5)],
            })
    for event in events[:100:2]:  # revoke half the confirms
        store.append("revoke", {"target_event_id": event["event_id"]})
    report = store.verify()
    assert report["replay_ok"] and not report["bad_lines"]
    assert report["lines"] == 100 * 5 + 50
    state = store.rebuild_derived()
    # Revoked confirms: only faces 3,4 could remain... removes still delete
    # 0-2 from surviving confirms; revoked confirms leave nothing assigned.
    assert all(a["identity_id"].startswith("identity_")
               for a in state.assignments.values())
    for i in range(0, 100, 2):  # revoked confirm -> no assignments remain
        assert not any(k.startswith(f"file{i}/")
                       for k in state.assignments), i
