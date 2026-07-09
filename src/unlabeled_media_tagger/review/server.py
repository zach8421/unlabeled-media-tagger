"""Local review server: stdlib HTTP, binds 127.0.0.1 only.

Serves the cluster-run review pages, the crop images, and the mark API.
Every mark is appended to the label store BEFORE the response returns, so a
browser crash or forgotten tab loses nothing. State pages always re-replay
the log — no cache to go stale.

Routes:
  GET  /                          cluster index
  GET  /group/<cluster_id>        per-face grid for one cluster
  GET  /queue                     growth-proposal review queue
  GET  /crop/<source_file_id>/<crop_file_name>   crop JPEG (traversal-checked)
  GET  /api/state                 JSON summary (tests, scripting)
  POST /api/mark                  {action, ...} -> one store event
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from unlabeled_media_tagger.review import pages
from unlabeled_media_tagger.review.store import LabelStore


class RunData:
    """One cluster run's outputs + the face fan-out map, loaded once."""

    def __init__(self, run_dir, crops_root=None, embed_dir=None):
        self.run_dir = Path(run_dir)
        self.params = json.loads((self.run_dir / "params.json").read_text())
        self.run_id = self.run_dir.name
        embed_dir = Path(embed_dir or self.params["embed_dir"])
        self.crops_root = Path(crops_root
                               or "/mnt/media1/folder2_detect/crops")

        with (self.run_dir / "cluster_summary.csv").open(newline="") as fh:
            self.summary = list(csv.DictReader(fh))

        # cluster_id -> its node rows.
        self.nodes_of_cluster = defaultdict(list)
        self.label_of_cluster = {}
        with (self.run_dir / "clusters.csv").open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["cluster_id"] != "":
                    self.nodes_of_cluster[row["cluster_id"]].append(row)
                    self.label_of_cluster[row["cluster_id"]] = \
                        row["cluster_label"]

        # track_id -> member (source_file_id, crop_file_name), only for
        # tracks that appear in a cluster (keeps memory proportional to the
        # reviewed run, not the whole corpus).
        wanted = {row["track_id"]
                  for rows in self.nodes_of_cluster.values() for row in rows}
        self.members_of_track = defaultdict(list)
        members_path = embed_dir / "tracks" / "track_members.csv"
        with members_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["track_id"] in wanted:
                    self.members_of_track[row["track_id"]].append(
                        (row["source_file_id"], row["crop_file_name"]))

        # Optional triage queue (scripts/build_triage_queue.py): ordered
        # cluster subset with a per-row reason; served as /?view=triage.
        self.triage = []
        triage_path = self.run_dir / "triage.csv"
        if triage_path.exists():
            with triage_path.open(newline="") as fh:
                self.triage = list(csv.DictReader(fh))

        self.proposals = []
        proposals_path = self.run_dir / "growth_proposals.jsonl"
        if proposals_path.exists():
            with proposals_path.open() as fh:
                for line in fh:
                    record = json.loads(line)
                    if record.get("decision") in ("review", "attach"):
                        self.proposals.append(record)

    def faces_of_cluster(self, cluster_id: str) -> list:
        faces = []
        for node in self.nodes_of_cluster.get(cluster_id, []):
            for file_id, crop_name in self.members_of_track[node["track_id"]]:
                faces.append({
                    "face_key": f"{file_id}/{crop_name}",
                    "source_file_id": file_id,
                    "crop_file_name": crop_name,
                    "track_id": node["track_id"],
                    "similarity_to_cluster": node["similarity_to_cluster"],
                })
        return faces


class ReviewHandler(BaseHTTPRequestHandler):
    # Injected via functools.partial in serve().
    def __init__(self, *args, store: LabelStore, data: RunData, **kwargs):
        self.store = store
        self.data = data
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # ------------------------------------------------------------- helpers

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, markup: str, code: int = 200):
        self._send(code, markup.encode(), "text/html; charset=utf-8")

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _review_status(self, state):
        """cluster_id -> ('confirmed', name) | ('rejected', '') for this run.

        Uses state.cluster_review (last decision wins) so index and group
        pages agree after confirm->reject or reject->confirm sequences.
        """
        names = {}
        for confirm in state.confirms:
            if confirm["run_id"] == self.data.run_id:
                canonical = state.canonical_identity(confirm["identity_id"])
                identity = state.identities.get(canonical, {})
                names[str(confirm["cluster_id"])] = identity.get("name", "")
        status = {}
        for (run_id, cluster_id), decision in state.cluster_review.items():
            if run_id != self.data.run_id:
                continue
            status[cluster_id] = ((decision, names.get(cluster_id, ""))
                                  if decision == "confirmed"
                                  else ("rejected", ""))
        return status

    def _cluster_identity(self, state, cluster_id: str):
        for confirm in reversed(state.confirms):
            if (confirm["run_id"] == self.data.run_id
                    and str(confirm["cluster_id"]) == cluster_id):
                canonical = state.canonical_identity(confirm["identity_id"])
                identity = state.identities.get(canonical)
                if identity is not None:
                    return {"identity_id": canonical,
                            "name": identity["name"]}
        return None

    def _recent_events(self, limit: int = 15):
        summaries = []
        for event in reversed(self.store.events()[-200:]):
            if event["type"] == "revoke":
                continue
            payload = event["payload"]
            bits = [event["type"]]
            for key in ("face_key", "identity_id", "cluster_id", "name"):
                if payload.get(key) not in (None, ""):
                    bits.append(f"{key}={payload[key]}")
            summaries.append({"event_id": event["event_id"],
                              "type": event["type"],
                              "summary": " ".join(str(b) for b in bits)[:110]})
            if len(summaries) >= limit:
                break
        return summaries

    # -------------------------------------------------------------- routes

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        page = int(query.get("page", ["0"])[0])
        parts = [unquote(p) for p in parsed.path.split("/") if p]

        if parsed.path == "/":
            state = self.store.replay()
            medoid_of = {
                cluster_id: max(rows, key=lambda r:
                                float(r["similarity_to_cluster"] or 0))["face_key"]
                for cluster_id, rows in self.data.nodes_of_cluster.items()}
            triage_view = query.get("view", [""])[0] == "triage"
            return self._html(pages.render_index(
                self.data.run_id, self.data.summary,
                self._review_status(state), page=page, medoid_of=medoid_of,
                triage=self.data.triage if triage_view else None,
                n_triage=len(self.data.triage)))

        if parts[:1] == ["group"] and len(parts) == 2:
            cluster_id = parts[1]
            if cluster_id not in self.data.nodes_of_cluster:
                return self._html("<h1>404 no such cluster</h1>", 404)
            state = self.store.replay()
            faces = self.data.faces_of_cluster(cluster_id)
            removed = {face_key for run, cid, face_key
                       in state.removed_from_cluster
                       if run == self.data.run_id and cid == cluster_id}
            return self._html(pages.render_group(
                self.data.run_id, cluster_id,
                self.data.label_of_cluster[cluster_id], faces,
                identity=self._cluster_identity(state, cluster_id),
                rejected=state.cluster_review.get(
                    (self.data.run_id, cluster_id)) == "rejected",
                removed_keys=removed,
                recent_events=self._recent_events(),
                page=page, sort=query.get("sort", ["central"])[0]))

        if parsed.path == "/queue":
            state = self.store.replay()
            decided = {face_key for face_key in
                       (p["face_key"] for p in self.data.proposals)
                       if face_key in state.assignments
                       or any(face_key == fk for fk, _
                              in state.cannot_link_face_identity)}
            return self._html(pages.render_queue(
                self.data.run_id, self.data.proposals, decided))

        if parts[:1] == ["crop"] and len(parts) == 3:
            return self._serve_crop(parts[1], parts[2])

        if parsed.path == "/api/state":
            state = self.store.replay()
            return self._json({
                "run_id": self.data.run_id,
                "n_clusters": len(self.data.summary),
                "n_identities": len(state.identities),
                "n_assignments": len(state.assignments),
                "n_cannot_link_face_face": len(state.cannot_link_face_face),
                "n_cannot_link_face_identity":
                    len(state.cannot_link_face_identity),
                "reviewed": self._review_status(state),
            })

        return self._html("<h1>404</h1>", 404)

    def _serve_crop(self, file_id: str, crop_name: str):
        if "/" in file_id or ".." in file_id or ".." in crop_name \
                or "/" in crop_name:
            return self._html("bad path", 400)
        crop_path = (self.data.crops_root / file_id / crop_name).resolve()
        if not str(crop_path).startswith(
                str(self.data.crops_root.resolve()) + "/"):
            return self._html("bad path", 400)
        if not crop_path.is_file():
            return self._html("not found", 404)
        self._send(200, crop_path.read_bytes(), "image/jpeg")

    def do_POST(self):
        # CSRF guard: the server binds 127.0.0.1, but any webpage in the
        # reviewer's browser can still fire cross-site POSTs at localhost.
        # Require a local Host, a local Origin when one is sent, and the
        # JSON content type our own pages use (cross-site "simple" requests
        # can only send text/plain-ish types).
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return self._json({"error": "forbidden host"}, 403)
        origin = self.headers.get("Origin")
        if origin is not None:
            origin_host = urlparse(origin).hostname
            if origin_host not in ("127.0.0.1", "localhost"):
                return self._json({"error": "forbidden origin"}, 403)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0]
        if content_type.strip() != "application/json":
            return self._json({"error": "expected application/json"}, 403)

        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        if parsed.path != "/api/mark":
            return self._json({"error": "unknown endpoint"}, 404)
        try:
            # One mark at a time: several actions are replay-then-append
            # read-modify-writes (identity-id allocation, confirm member
            # filtering) that must not interleave under the threaded server.
            with self.store.mutation_lock:
                event = self._handle_mark(payload)
        except (KeyError, ValueError) as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
        return self._json({"ok": True, "event_id": event["event_id"]})

    # ------------------------------------------------------------- actions

    def _handle_mark(self, payload: dict) -> dict:
        action = payload["action"]
        run_id = self.data.run_id
        state = self.store.replay()

        if action == "remove_face":
            cluster_id = str(payload["cluster_id"])
            face_key = payload["face_key"]
            identity = self._cluster_identity(state, cluster_id)
            context = [f["face_key"]
                       for f in self.data.faces_of_cluster(cluster_id)
                       if f["face_key"] != face_key]
            return self.store.append("face.remove", {
                "face_key": face_key, "run_id": run_id,
                "cluster_id": cluster_id,
                "identity_id": identity["identity_id"] if identity else None,
                "context_face_keys": context,
            })

        if action == "undo_remove_face":
            face_key = payload["face_key"]
            cluster_id = str(payload["cluster_id"])
            # Only ACTIVE removes qualify — re-revoking an already-revoked
            # event would be a silent no-op that leaves the live one standing.
            for event in reversed(self.store.active_events()):
                if (event["type"] == "face.remove"
                        and event["payload"].get("face_key") == face_key
                        and event["payload"].get("run_id") == run_id
                        and str(event["payload"].get("cluster_id"))
                        == cluster_id):
                    return self.store.append(
                        "revoke", {"target_event_id": event["event_id"]})
            raise ValueError(f"no active remove event found for {face_key}")

        if action == "confirm_cluster":
            cluster_id = str(payload["cluster_id"])
            name = (payload.get("identity_name") or "").strip()
            existing = self._cluster_identity(state, cluster_id)
            if existing is not None:
                identity_id = existing["identity_id"]
                if name and name != state.identities[identity_id]["name"]:
                    self.store.append("identity.rename",
                                      {"identity_id": identity_id,
                                       "name": name})
            else:
                identity_id = self.store.next_identity_id()
                self.store.append("identity.create", {
                    "identity_id": identity_id, "name": name,
                    "tracked": True})
            removed = {face_key for run, cid, face_key
                       in state.removed_from_cluster
                       if run == run_id and cid == cluster_id}
            members = [f["face_key"]
                       for f in self.data.faces_of_cluster(cluster_id)
                       if f["face_key"] not in removed]
            return self.store.append("cluster.confirm", {
                "run_id": run_id, "cluster_id": cluster_id,
                "identity_id": identity_id, "member_face_keys": members,
            })

        if action == "reject_cluster":
            cluster_id = str(payload["cluster_id"])
            members = [f["face_key"]
                       for f in self.data.faces_of_cluster(cluster_id)]
            return self.store.append("cluster.reject", {
                "run_id": run_id, "cluster_id": cluster_id,
                "member_face_keys": members,
            })

        if action == "accept_proposal":
            identity_id = payload["identity_id"]
            proposal = next(
                (p for p in self.data.proposals
                 if p["face_key"] == payload["face_key"]
                 and p["identity_id"] == identity_id), None)
            # Accepting a track proposal fans out to every member face.
            keys = ((proposal or {}).get("member_face_keys")
                    or [payload["face_key"]])
            # An explicit accept overrides an earlier "Not this person":
            # revoke the active reject-born removes for these faces, else
            # assignment and cannot-link would both be live.
            for event in self.store.active_events():
                if (event["type"] == "face.remove"
                        and event["payload"].get("identity_id") == identity_id
                        and event["payload"].get("face_key") in keys):
                    self.store.append(
                        "revoke", {"target_event_id": event["event_id"],
                                   "reason": "superseded by accept_proposal"})
            event = None
            for face_key in keys:
                current = state.assignments.get(face_key)
                if current and current["identity_id"] != identity_id:
                    continue  # never silently steal a face from another identity
                event = self.store.append("face.assign", {
                    "face_key": face_key,
                    "identity_id": identity_id,
                    "sim": (proposal or {}).get("score"),
                    "margin": (proposal or {}).get("margin"),
                })
            if event is None:
                raise ValueError(
                    "every member face is already assigned to a different "
                    "identity; undo those assignments first")
            return event

        if action == "accept_all_attach":
            event = None
            for proposal in self.data.proposals:
                if proposal.get("decision") != "attach":
                    continue
                keys = (proposal.get("member_face_keys")
                        or [proposal["face_key"]])
                if any((k, proposal["identity_id"])
                       in state.cannot_link_face_identity for k in keys):
                    continue  # user already said "not this person"
                already = [state.assignments.get(k) for k in keys]
                if any(a is not None for a in already):
                    continue  # decided (or partially owned) — skip in bulk mode
                for face_key in keys:
                    event = self.store.append("face.assign", {
                        "face_key": face_key,
                        "identity_id": proposal["identity_id"],
                        "sim": proposal.get("score"),
                        "margin": proposal.get("margin"),
                    })
            if event is None:
                raise ValueError("no undecided attach proposals")
            return event

        if action == "reject_proposal":
            return self.store.append("face.remove", {
                "face_key": payload["face_key"],
                "identity_id": payload["identity_id"],
                "context_face_keys": [],
            })

        if action == "merge_cluster_identities":
            # "These confirmed clusters are the same person": merge their
            # identities into the FIRST selected cluster's identity. One
            # identity.merge event per source — each individually revocable.
            cluster_ids = [str(c) for c in payload.get("cluster_ids", [])]
            identities = []
            for cluster_id in cluster_ids:
                identity = self._cluster_identity(state, cluster_id)
                if identity is None:
                    raise ValueError(
                        f"cluster {cluster_id} has no confirmed identity — "
                        f"confirm it before merging")
                canonical = state.canonical_identity(identity["identity_id"])
                if canonical not in identities:
                    identities.append(canonical)
            if len(identities) < 2:
                raise ValueError("select 2+ clusters with distinct identities")
            target = identities[0]
            event = None
            for source in identities[1:]:
                event = self.store.append("identity.merge", {
                    "source_identity_id": source,
                    "target_identity_id": target,
                })
            return event

        if action == "undo":
            return self.store.append(
                "revoke", {"target_event_id": payload["event_id"]})

        raise ValueError(f"unknown action {action!r}")


def serve(labels_dir, run_dir, crops_root=None, port: int = 8765,
          embed_dir=None):
    store = LabelStore(labels_dir)
    store.rebuild_derived()
    data = RunData(run_dir, crops_root=crops_root, embed_dir=embed_dir)
    handler = partial(ReviewHandler, store=store, data=data)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"review server: http://127.0.0.1:{server.server_port}/ "
          f"(run {data.run_id}, {len(data.summary)} clusters, "
          f"{len(data.proposals)} queued proposals)", flush=True)
    return server
