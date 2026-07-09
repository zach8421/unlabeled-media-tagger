"""Append-only label store: events.jsonl is the truth, everything else derives.

Every human judgment (and every growth-pass acceptance) is one JSON line
appended to events.jsonl. State — identities, face assignments, cannot-links,
confirms — is rebuilt deterministically by replaying the log. This gives
feedback-loop.md's requirements for free: nothing is ever destroyed, every
action is a timestamped record, and undo is appending a `revoke` that
references the original event (revokes are themselves revocable).

Event envelope: {"event_id", "ts", "actor": "user"|"growth"|"ingest",
                 "type", "payload"}

Types (payload fields):
  identity.create   {identity_id, name?, tracked}
  identity.rename   {identity_id, name}
  identity.retire   {identity_id}
  identity.merge    {source_identity_id, target_identity_id}  — "same person";
                    non-destructive alias: the source identity's assignments
                    RESOLVE to the target at replay time, nothing is
                    rewritten; revoking the merge splits them again exactly.
  cluster.confirm   {run_id, cluster_id, identity_id, member_face_keys[]}
  cluster.reject    {run_id, cluster_id, member_face_keys[]}
  face.remove       {face_key, run_id?, cluster_id?, identity_id?,
                     context_face_keys[]}   <- the cannot-link generator
  face.assign       {face_key, identity_id, sim?, margin?}
  revoke            {target_event_id, reason?}

face_key = "<source_file_id>/<crop_file_name>" — matches the on-disk crop
locator; identity membership is face-level, so it survives re-clustering
(cluster ids are always qualified by run_id and never persisted as identity).

Concurrency: single-writer via flock on events.jsonl.lock around each append;
appends are single O_APPEND writes (< 4 KB lines are atomic on Linux).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CONTEXT_FACE_CAP = 300  # keeps face.remove lines small enough for atomic writes

EVENT_TYPES = {
    "identity.create", "identity.rename", "identity.retire",
    "identity.merge",
    "cluster.confirm", "cluster.reject",
    "face.remove", "face.assign",
    "revoke",
}

DERIVED_FILES = (
    "identities.csv", "assignments.csv", "confirms.csv",
    "cannot_links_face_identity.csv", "cannot_links_face_face.csv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class State:
    """Replayed view of the log. All dict keys are stable identifiers."""

    identities: dict = field(default_factory=dict)
    # face_key -> {"identity_id", "source", "event_id", "ts"}
    assignments: dict = field(default_factory=dict)
    cannot_link_face_identity: set = field(default_factory=set)
    cannot_link_face_face: set = field(default_factory=set)
    confirms: list = field(default_factory=list)
    rejected_clusters: set = field(default_factory=set)  # (run_id, cluster_id)
    # (run_id, str(cluster_id)) -> 'confirmed' | 'rejected' — LAST decision
    # wins, so index and group pages agree on precedence.
    cluster_review: dict = field(default_factory=dict)
    # (run_id, str(cluster_id), face_key) — pre/post-confirm removals, so the
    # UI can render removed faces and confirm can exclude them.
    removed_from_cluster: set = field(default_factory=set)
    # source identity -> canonical identity, after resolving merge chains.
    merged_into: dict = field(default_factory=dict)

    def canonical_identity(self, identity_id: str) -> str:
        return self.merged_into.get(identity_id, identity_id)
    n_events_active: int = 0
    max_identity_seq: int = 0


class LabelStore:
    def __init__(self, labels_dir):
        self.labels_dir = Path(labels_dir)
        self.events_path = self.labels_dir / "events.jsonl"
        self.lock_path = self.labels_dir / "events.jsonl.lock"
        self.derived_dir = self.labels_dir / "derived"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        # Intra-process guard for read-modify-write sequences (e.g. replay ->
        # next_identity_id -> append) under a threaded server. The flock in
        # append() only covers the single write.
        self.mutation_lock = threading.Lock()

    # ------------------------------------------------------------- appending

    def append(self, event_type: str, payload: dict,
               actor: str = "user") -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type!r}")
        payload = dict(payload)
        if event_type == "face.remove":
            context = payload.get("context_face_keys") or []
            payload["context_face_keys"] = list(context)[:CONTEXT_FACE_CAP]
        event = {
            # 64 random bits: burst appends (accept-all fan-out) must never
            # collide, because revoke resolution keys on event_id.
            "event_id": "ev_" + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S") + "_" + secrets.token_hex(8),
            "ts": utc_now(),
            "actor": actor,
            "type": event_type,
            "payload": payload,
        }
        line = json.dumps(event, separators=(",", ":")) + "\n"
        with self.lock_path.open("w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            fd = os.open(self.events_path,
                         os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        return event

    def next_identity_id(self) -> str:
        state = self.replay()
        return f"identity_{state.max_identity_seq + 1:04d}"

    # --------------------------------------------------------------- reading

    def events(self, *, skip_bad_lines: bool = True) -> list:
        if not self.events_path.exists():
            return []
        out = []
        with self.events_path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    if not skip_bad_lines:
                        raise ValueError(f"bad JSON at line {line_no}")
        return out

    def active_events(self) -> list:
        """Events surviving revoke-chain resolution (revokes excluded)."""
        return _active_events(self.events())

    def replay(self) -> State:
        return replay_events(self.events())

    # ------------------------------------------------------- derived views

    def rebuild_derived(self) -> State:
        state = self.replay()
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        import csv

        def atomic_csv(name, header, rows):
            tmp = self.derived_dir / (name + ".tmp")
            with tmp.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)
            os.replace(tmp, self.derived_dir / name)

        face_counts = {}
        for info in state.assignments.values():
            face_counts[info["identity_id"]] = \
                face_counts.get(info["identity_id"], 0) + 1
        atomic_csv(
            "identities.csv",
            ["identity_id", "name", "tracked", "retired", "merged_into",
             "created_at", "updated_at", "n_faces"],
            [[iid, i["name"], str(i["tracked"]).lower(),
              str(i["retired"]).lower(), i.get("merged_into", ""),
              i["created_at"], i["updated_at"], face_counts.get(iid, 0)]
             for iid, i in sorted(state.identities.items())])
        atomic_csv(
            "assignments.csv",
            ["face_key", "identity_id", "source", "event_id", "ts"],
            [[k, a["identity_id"], a["source"], a["event_id"], a["ts"]]
             for k, a in sorted(state.assignments.items())])
        atomic_csv(
            "confirms.csv",
            ["run_id", "cluster_id", "identity_id", "n_members",
             "event_id", "ts"],
            [[c["run_id"], c["cluster_id"], c["identity_id"],
              c["n_members"], c["event_id"], c["ts"]]
             for c in state.confirms])
        atomic_csv(
            "cannot_links_face_identity.csv",
            ["face_key", "identity_id"],
            sorted(state.cannot_link_face_identity))
        atomic_csv(
            "cannot_links_face_face.csv",
            ["face_key_a", "face_key_b"],
            sorted(state.cannot_link_face_face))

        raw = self.events_path.read_bytes() if self.events_path.exists() else b""
        meta = {
            "rebuilt_at": utc_now(),
            "log_lines": raw.count(b"\n"),
            "log_sha256": hashlib.sha256(raw).hexdigest(),
            "n_events_active": state.n_events_active,
        }
        tmp = self.derived_dir / "build_meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, self.derived_dir / "build_meta.json")
        return state

    # ----------------------------------------------------------- inspection

    def verify(self) -> dict:
        """Integrity report: parseable lines, torn tail, replay success."""
        report = {"exists": self.events_path.exists(), "lines": 0,
                  "bad_lines": [], "torn_tail": False}
        if not report["exists"]:
            return report
        raw = self.events_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            report["torn_tail"] = True
        for line_no, line in enumerate(raw.splitlines(), 1):
            report["lines"] += 1
            try:
                event = json.loads(line)
                if event.get("type") not in EVENT_TYPES:
                    report["bad_lines"].append(line_no)
            except ValueError:
                report["bad_lines"].append(line_no)
        report["replay_ok"] = True
        try:
            state = self.replay()
            report["n_events_active"] = state.n_events_active
            report["n_identities"] = len(state.identities)
            report["n_assignments"] = len(state.assignments)
        except Exception as exc:  # noqa: BLE001 - report, caller decides
            report["replay_ok"] = False
            report["replay_error"] = f"{type(exc).__name__}: {exc}"
        return report


def _active_events(events: list) -> list:
    """Resolve revoke chains: an event is inactive iff an ACTIVE revoke
    targets it. Revokes only reference earlier events, so the recursion
    terminates walking forward."""
    revokes_of: dict[str, list] = {}
    for event in events:
        if event["type"] == "revoke":
            target = event["payload"].get("target_event_id")
            if target:
                revokes_of.setdefault(target, []).append(event["event_id"])

    active_memo: dict[str, bool] = {}

    def is_active(event_id: str) -> bool:
        if event_id in active_memo:
            return active_memo[event_id]
        active = not any(is_active(r) for r in revokes_of.get(event_id, ()))
        active_memo[event_id] = active
        return active

    return [e for e in events
            if e["type"] != "revoke" and is_active(e["event_id"])]


def replay_events(events: list) -> State:
    state = State()
    # Sequence numbers come from ALL creates, revoked ones included: a
    # revoked identity's id must never be reissued, because still-active
    # confirms/assigns may reference it (reuse would merge two people).
    for event in events:
        if event["type"] == "identity.create":
            iid = event["payload"].get("identity_id", "")
            try:  # identity_0042 -> 42, for next_identity_id()
                state.max_identity_seq = max(
                    state.max_identity_seq, int(iid.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                pass

    confirm_cluster_of_event = {}  # confirm event_id -> (run_id, cluster_id)
    raw_merges: dict = {}  # source identity -> target identity (active merges)
    for event in _active_events(events):
        payload = event["payload"]
        etype = event["type"]
        state.n_events_active += 1

        if etype == "identity.create":
            iid = payload["identity_id"]
            state.identities[iid] = {
                "name": payload.get("name") or "",
                "tracked": bool(payload.get("tracked", True)),
                "retired": False,
                "created_at": event["ts"], "updated_at": event["ts"],
            }

        elif etype == "identity.rename":
            identity = state.identities.get(payload["identity_id"])
            if identity is not None:
                identity["name"] = payload.get("name") or ""
                identity["updated_at"] = event["ts"]

        elif etype == "identity.retire":
            identity = state.identities.get(payload["identity_id"])
            if identity is not None:
                identity["retired"] = True
                identity["updated_at"] = event["ts"]

        elif etype == "identity.merge":
            source = payload.get("source_identity_id")
            target = payload.get("target_identity_id")
            if source and target and source != target:
                raw_merges[source] = target  # latest merge for a source wins

        elif etype == "cluster.confirm":
            iid = payload["identity_id"]
            for face_key in payload.get("member_face_keys", []):
                state.assignments[face_key] = {
                    "identity_id": iid, "source": "confirm",
                    "event_id": event["event_id"], "ts": event["ts"],
                }
            run_cluster = (payload.get("run_id", ""),
                           str(payload.get("cluster_id", "")))
            confirm_cluster_of_event[event["event_id"]] = run_cluster
            state.cluster_review[run_cluster] = "confirmed"
            state.confirms.append({
                "run_id": payload.get("run_id", ""),
                "cluster_id": payload.get("cluster_id", ""),
                "identity_id": iid,
                "n_members": len(payload.get("member_face_keys", [])),
                "event_id": event["event_id"], "ts": event["ts"],
            })

        elif etype == "cluster.reject":
            run_cluster = (payload.get("run_id", ""),
                           str(payload.get("cluster_id", "")))
            state.rejected_clusters.add(run_cluster)
            state.cluster_review[run_cluster] = "rejected"
            # Rejecting a previously-confirmed cluster withdraws the
            # assignments that confirm created (manual/growth assigns stay).
            for face_key in payload.get("member_face_keys", []):
                current = state.assignments.get(face_key)
                if (current is not None and current["source"] == "confirm"
                        and confirm_cluster_of_event.get(current["event_id"])
                        == run_cluster):
                    del state.assignments[face_key]

        elif etype == "face.remove":
            face_key = payload["face_key"]
            identity_id = payload.get("identity_id")
            current = state.assignments.get(face_key)
            # Only unassign when the removal names the same identity.
            # An identity-less removal (unconfirmed machine cluster) records
            # cannot-links but must not clobber an assignment some OTHER
            # run's confirm created for this face.
            if current is not None and identity_id is not None \
                    and current["identity_id"] == identity_id:
                del state.assignments[face_key]
            if identity_id:
                state.cannot_link_face_identity.add((face_key, identity_id))
            if payload.get("run_id") is not None \
                    and payload.get("cluster_id") is not None:
                state.removed_from_cluster.add(
                    (payload["run_id"], str(payload["cluster_id"]), face_key))
            for other in payload.get("context_face_keys", []):
                if other != face_key:
                    state.cannot_link_face_face.add(
                        tuple(sorted((face_key, other))))

        elif etype == "face.assign":
            state.assignments[payload["face_key"]] = {
                "identity_id": payload["identity_id"],
                "source": "growth" if event["actor"] == "growth" else "manual",
                "event_id": event["event_id"], "ts": event["ts"],
            }

    # Resolve merge chains (A->B, B->C  =>  A->C, B->C) and rewrite the VIEW
    # only — raw payloads are never touched, so revoking a merge restores the
    # split exactly. Cycles (possible in a hand-edited log) break at the
    # first revisited node rather than looping.
    for source in raw_merges:
        seen = {source}
        target = raw_merges[source]
        while target in raw_merges and target not in seen:
            seen.add(target)
            target = raw_merges[target]
        state.merged_into[source] = target
    if state.merged_into:
        for face_key, info in state.assignments.items():
            info["identity_id"] = state.canonical_identity(info["identity_id"])
        state.cannot_link_face_identity = {
            (face_key, state.canonical_identity(iid))
            for face_key, iid in state.cannot_link_face_identity}
        for source, target in state.merged_into.items():
            identity = state.identities.get(source)
            if identity is not None:
                identity["merged_into"] = target

    return state
