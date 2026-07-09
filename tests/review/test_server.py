"""Live-server tests: real ThreadingHTTPServer, tmp store, fixture run."""

from __future__ import annotations

import csv
import json
import threading
import urllib.request

import pytest

from unlabeled_media_tagger.review.server import serve
from unlabeled_media_tagger.review.store import LabelStore


@pytest.fixture()
def run_fixture(tmp_path):
    """A 2-cluster run: cluster 0 has 2 tracks (4 faces), cluster 1 has 2."""
    embed_dir = tmp_path / "embed"
    run_dir = tmp_path / "cluster_runs" / "run_t"
    crops = tmp_path / "crops"
    (embed_dir / "tracks").mkdir(parents=True)
    run_dir.mkdir(parents=True)

    jpeg = bytes.fromhex("ffd8ffdb004300ffffffffffffffffffffffffffffffffffff"
                         "ffffffffffffffffffffffffffffffffffffffffffffffffff"
                         "ffffffffffffffffffffffffc0000b080001000101011100ff"
                         "c40014100100000000000000000000000000000000ffda0008"
                         "010100013f10")  # minimal valid JPEG
    members = [("fileA", "a1.jpg", "fileA__t000"),
               ("fileA", "a2.jpg", "fileA__t000"),
               ("fileB", "b1.jpg", "fileB__t000"),
               ("fileB", "b2.jpg", "fileB__t000"),
               ("fileC", "c1.jpg", "fileC__t000"),
               ("fileD", "d1.jpg", "fileD__t000")]
    with (embed_dir / "tracks" / "track_members.csv").open("w",
                                                           newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["track_id", "source_file_id", "crop_file_name"])
        for file_id, crop, track in members:
            writer.writerow([track, file_id, crop])
            (crops / file_id).mkdir(parents=True, exist_ok=True)
            (crops / file_id / crop).write_bytes(jpeg)

    (run_dir / "params.json").write_text(json.dumps(
        {"embed_dir": str(embed_dir)}))
    with (run_dir / "clusters.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "track_id", "source_file_id", "media_type",
                         "n_faces", "medoid_crop_file_name", "face_key",
                         "cluster_id", "cluster_label", "assigned",
                         "similarity_to_cluster", "event_path"])
        writer.writerow([0, "fileA__t000", "fileA", "video", 2, "a1.jpg",
                         "fileA/a1.jpg", 0, "person_00000", "true", 0.93, "ev1"])
        writer.writerow([1, "fileB__t000", "fileB", "video", 2, "b1.jpg",
                         "fileB/b1.jpg", 0, "person_00000", "true", 0.90, "ev1"])
        writer.writerow([2, "fileC__t000", "fileC", "image", 1, "c1.jpg",
                         "fileC/c1.jpg", 1, "person_00001", "true", 0.88, "ev2"])
        writer.writerow([3, "fileD__t000", "fileD", "image", 1, "d1.jpg",
                         "fileD/d1.jpg", 1, "person_00001", "true", 0.87, "ev2"])
    with (run_dir / "cluster_summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "cluster_label", "n_nodes", "n_faces",
                         "n_files", "n_events", "example_events"])
        writer.writerow([0, "person_00000", 2, 4, 2, 1, "ev1"])
        writer.writerow([1, "person_00001", 2, 2, 2, 1, "ev2"])
    return {"labels": tmp_path / "labels", "run_dir": run_dir, "crops": crops}


@pytest.fixture()
def client(run_fixture):
    server = serve(run_fixture["labels"], run_fixture["run_dir"],
                   crops_root=run_fixture["crops"], port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        if payload is None:
            req = urllib.request.Request(base + path)
        else:
            req = urllib.request.Request(
                base + path, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()

    yield request, run_fixture
    server.shutdown()


def test_index_and_group_render(client):
    request, _ = client
    status, body = request("/")
    assert status == 200
    assert b"person_00000" in body and b"unreviewed" in body
    assert b"triage queue" not in body  # no triage.csv in this fixture
    status, body = request("/group/0")
    assert status == 200
    assert body.count(b"class='card'") == 4  # tracks fanned out to faces
    assert b"fileA/a1.jpg" in body


def test_triage_view(run_fixture):
    (run_fixture["run_dir"] / "triage.csv").write_text(
        "rank,section,reason,cluster_id\n"
        "1,largest_faces,top by faces (2),1\n")
    server = serve(run_fixture["labels"], run_fixture["run_dir"],
                   crops_root=run_fixture["crops"], port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as resp:
            body = resp.read()
        assert b"triage queue (1)" in body  # full index links to the queue
        assert b"person_00000" in body      # ...and still shows everything
        with urllib.request.urlopen(base + "/?view=triage") as resp:
            body = resp.read()
        assert b"Triage queue" in body
        assert b"person_00001" in body      # the triage cluster
        assert b"person_00000" not in body  # non-triage clusters filtered out
        assert b"top by faces" in body      # reason column rendered
    finally:
        server.shutdown()


def test_crop_serving_and_traversal_guard(client):
    request, _ = client
    status, body = request("/crop/fileA/a1.jpg")
    assert status == 200 and body[:2] == b"\xff\xd8"
    status, _ = request("/crop/fileA/..%2F..%2Fetc%2Fpasswd")
    assert status in (400, 404)
    status, _ = request("/crop/..%2Ffile/x.jpg")
    assert status in (400, 404)


def test_mark_persists_before_response(client):
    request, fixture = client
    status, body = request("/api/mark", {
        "action": "confirm_cluster", "cluster_id": 0,
        "identity_name": "Sarah Chen"})
    assert status == 200
    event_id = json.loads(body)["event_id"]
    # The event is on disk (not just in memory) the moment the POST returns.
    log = (fixture["labels"] / "events.jsonl").read_text()
    assert event_id in log
    state = LabelStore(fixture["labels"]).replay()
    assert state.assignments["fileA/a1.jpg"]["identity_id"] == "identity_0001"
    assert len(state.assignments) == 4
    assert state.identities["identity_0001"]["name"] == "Sarah Chen"

    status, body = request("/")
    assert b"Sarah Chen" in body


def test_remove_then_confirm_excludes_face(client):
    request, fixture = client
    request("/api/mark", {"action": "remove_face", "cluster_id": 0,
                          "face_key": "fileB/b2.jpg"})
    request("/api/mark", {"action": "confirm_cluster", "cluster_id": 0,
                          "identity_name": "Marcus"})
    state = LabelStore(fixture["labels"]).replay()
    assert "fileB/b2.jpg" not in state.assignments
    assert len(state.assignments) == 3
    # Context cannot-links were captured against the other members.
    assert ("fileA/a1.jpg", "fileB/b2.jpg") in state.cannot_link_face_face


def test_remove_from_confirmed_cluster_creates_identity_cannot_link(client):
    request, fixture = client
    request("/api/mark", {"action": "confirm_cluster", "cluster_id": 0,
                          "identity_name": "Sarah"})
    request("/api/mark", {"action": "remove_face", "cluster_id": 0,
                          "face_key": "fileA/a2.jpg"})
    state = LabelStore(fixture["labels"]).replay()
    assert ("fileA/a2.jpg", "identity_0001") in state.cannot_link_face_identity
    assert "fileA/a2.jpg" not in state.assignments


def test_undo_remove_restores(client):
    request, fixture = client
    request("/api/mark", {"action": "remove_face", "cluster_id": 0,
                          "face_key": "fileA/a1.jpg"})
    status, body = request("/group/0")
    assert b"card removed" in body
    request("/api/mark", {"action": "undo_remove_face", "cluster_id": 0,
                          "face_key": "fileA/a1.jpg"})
    status, body = request("/group/0")
    assert b"card removed" not in body


def test_reject_cluster_and_state_endpoint(client):
    request, _ = client
    request("/api/mark", {"action": "reject_cluster", "cluster_id": 1})
    status, body = request("/api/state")
    state = json.loads(body)
    assert state["reviewed"]["1"] == ["rejected", ""]
    assert state["n_identities"] == 0


def test_bad_action_is_400_not_crash(client):
    request, _ = client
    status, body = request("/api/mark", {"action": "detonate"})
    assert status == 400
    status, _ = request("/")
    assert status == 200  # server survived
