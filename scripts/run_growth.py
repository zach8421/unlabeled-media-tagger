#!/usr/bin/env python
"""Growth pass driver: proposals for attaching unassigned nodes to identities.

Reads a cluster run + the label store's derived views, votes every unassigned
node against verified identities' member faces (clustering/grow.py), and
writes <run-dir>/growth_proposals.jsonl. PROPOSALS ONLY: nothing is written
to the label store; accepts happen in the review server's /queue page (which
fans a node's member faces out to face.assign events).

Deterministic and idempotent: proposals carry no timestamps and are sorted,
so unchanged inputs reproduce a byte-identical file (params_hash included in
each record and in growth_meta.json).

Also the M6 verification harness: --leave-one-out 0.2 hides 20% of each
verified identity's member faces, treats them as candidates, and reports
re-attach rate and false attaches instead of writing proposals.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/run_growth.py \
      --run-dir /mnt/media1/folder2_cluster/run_001 \
      [--labels-dir /mnt/media1/folder2_labels] [--tau-attach 0.78]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unlabeled_media_tagger.clustering.grow import score_candidates
from unlabeled_media_tagger.review.store import LabelStore


def load_face_locations(embed_dir: Path, wanted: set) -> dict:
    """face_key -> (shard, row) for the wanted keys."""
    out = {}
    with (embed_dir / "faces_index.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = f"{row['source_file_id']}/{row['crop_file_name']}"
            if key in wanted:
                out[key] = (int(row["shard"]), int(row["row"]))
    return out


def face_vectors(embed_dir: Path, locations: dict, keys: list):
    """-> (matrix, kept_keys). The label store outlives embed runs, so an
    assigned face_key can be absent from the current faces_index (stale run,
    read-failed crop); those are skipped with a warning, never a crash."""
    shard_cache = {}
    rows, kept = [], []
    missing = 0
    for key in keys:
        location = locations.get(key)
        if location is None:
            missing += 1
            continue
        shard, row = location
        mm = shard_cache.get(shard)
        if mm is None:
            mm = np.load(embed_dir / "embeddings" / f"shard_{shard:05d}.npy",
                         mmap_mode="r")
            shard_cache[shard] = mm
        rows.append(np.asarray(mm[row]))
        kept.append(key)
    if missing:
        print(f"WARNING: {missing} assigned face_keys not in this embed set "
              f"(stale label references) — skipped", flush=True)
    matrix = (np.stack(rows).astype(np.float32) if rows
              else np.empty((0, 512), np.float32))
    return matrix, kept


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels-dir", default="/mnt/media1/folder2_labels")
    ap.add_argument("--tau-attach", type=float, default=None,
                    help="default: calibration tau_edge")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--review-band", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--leave-one-out", type=float, default=None, metavar="FRAC",
                    help="validation mode: hide FRAC of each identity's "
                         "SOURCE FILES (identities spanning >=3 files), "
                         "report re-attach + false-attach rates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / "params.json").read_text())
    embed_dir = Path(params["embed_dir"])

    tau_attach = args.tau_attach
    if tau_attach is None:
        calibration_path = embed_dir / "calibration.json"
        tau_attach = json.loads(calibration_path.read_text())["tau_edge"]

    store = LabelStore(args.labels_dir)
    state = store.rebuild_derived()
    # An identity whose create was revoked can still be referenced by an
    # active confirm — it must not enter the verified set (and would KeyError
    # on name lookup when building proposals). Merged identities resolve to
    # their canonical target, so "same person" merges pool their members.
    verified_ids = sorted({
        iid for iid in (state.canonical_identity(c["identity_id"])
                        for c in state.confirms)
        if iid in state.identities
        and not state.identities[iid]["retired"]})
    if not verified_ids:
        print("no verified identities (confirm at least one cluster first); "
              "nothing to grow.", flush=True)
        return 0
    identity_index = {iid: i for i, iid in enumerate(verified_ids)}

    members_by_identity = defaultdict(list)
    for face_key, info in state.assignments.items():
        if info["identity_id"] in identity_index:
            members_by_identity[info["identity_id"]].append(face_key)

    # Optional leave-one-out split. Held out by WHOLE SOURCE FILE, not by
    # face: per-face hiding leaves same-track sibling frames in the member
    # matrix, so a hidden face just re-finds its own track (measured: 1.0
    # re-attach that says nothing about cross-file attachment).
    rng = np.random.default_rng(args.seed)
    held_out = {}  # face_key -> identity_id (ground truth)
    if args.leave_one_out:
        for iid, keys in members_by_identity.items():
            files = sorted({k.split("/", 1)[0] for k in keys})
            if len(files) < 3:
                continue  # need >= 2 files left for MIN_DISTINCT_FILES
            n_hide = max(1, int(len(files) * args.leave_one_out))
            hidden_files = set(
                rng.choice(files, size=n_hide, replace=False).tolist())
            for key in keys:
                if key.split("/", 1)[0] in hidden_files:
                    held_out[key] = iid
        members_by_identity = {
            iid: [k for k in keys if k not in held_out]
            for iid, keys in members_by_identity.items()}

    candidate_member_keys = []
    identity_of_member_key = {}
    for iid in verified_ids:
        for key in sorted(members_by_identity.get(iid, [])):
            candidate_member_keys.append(key)
            identity_of_member_key[key] = iid
    print(f"verified identities: {len(verified_ids)}  member faces: "
          f"{len(candidate_member_keys)}  tau_attach: {tau_attach}",
          flush=True)

    wanted = set(candidate_member_keys) | set(held_out)
    locations = load_face_locations(embed_dir, wanted)
    member_matrix, member_keys = face_vectors(embed_dir, locations,
                                              candidate_member_keys)
    member_identity_idx = np.asarray(
        [identity_index[identity_of_member_key[k]] for k in member_keys],
        dtype=np.int64)
    member_files = [k.split("/", 1)[0] for k in member_keys]

    # Cannot-link resolution helpers.
    members_set_by_identity = {
        iid: set(members_by_identity.get(iid, [])) for iid in verified_ids}
    linked_faces = defaultdict(set)
    for a, b in state.cannot_link_face_face:
        linked_faces[a].add(b)
        linked_faces[b].add(a)

    def blocked_identities(candidate_face_keys: list) -> set:
        blocked = set()
        for face_key in candidate_face_keys:
            for iid in verified_ids:
                if (face_key, iid) in state.cannot_link_face_identity:
                    blocked.add(identity_index[iid])
            for other in linked_faces.get(face_key, ()):
                for iid in verified_ids:
                    if other in members_set_by_identity[iid]:
                        blocked.add(identity_index[iid])
        return blocked

    if args.leave_one_out:
        return _leave_one_out(args, embed_dir, locations, held_out,
                              member_matrix, member_identity_idx,
                              member_files, verified_ids, identity_index,
                              tau_attach, blocked_identities)

    # Candidates: unassigned nodes of this run (with their member faces).
    # Hub-quarantined nodes are junk-attractor members — never candidates.
    node_rows, members_of_track = [], defaultdict(list)
    with (run_dir / "clusters.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["assigned"] == "false" \
                    and row.get("quarantined", "false") != "true":
                node_rows.append(row)
    wanted_tracks = {row["track_id"] for row in node_rows}
    with (embed_dir / "tracks" / "track_members.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["track_id"] in wanted_tracks:
                members_of_track[row["track_id"]].append(
                    f"{row['source_file_id']}/{row['crop_file_name']}")

    node_embeddings = np.load(embed_dir / "nodes" / "node_embeddings.npy",
                              mmap_mode="r")
    candidate_matrix = np.stack(
        [node_embeddings[int(row["node_id"])] for row in node_rows]
    ).astype(np.float32) if node_rows else np.empty((0, 512), np.float32)
    blocked_per_candidate = [
        blocked_identities(members_of_track[row["track_id"]])
        for row in node_rows]
    print(f"candidates (unassigned nodes): {len(node_rows)}", flush=True)

    verdicts = score_candidates(
        candidate_matrix, member_matrix, member_identity_idx, member_files,
        tau_attach=tau_attach, margin=args.margin,
        review_band=args.review_band, k=args.k,
        blocked_identity_indices=blocked_per_candidate)

    growth_params = {
        "run_dir": str(run_dir), "tau_attach": tau_attach,
        "margin": args.margin, "review_band": args.review_band, "k": args.k,
        "n_members": len(member_keys), "n_verified": len(verified_ids),
        "n_candidates": len(node_rows),
    }
    params_hash = hashlib.sha256(
        json.dumps(growth_params, sort_keys=True).encode()).hexdigest()[:16]

    proposals = []
    counts = defaultdict(int)
    for verdict, row in zip(verdicts, node_rows):
        counts[verdict.decision] += 1
        if verdict.decision not in ("attach", "review"):
            continue
        iid = verified_ids[verdict.identity_index]
        proposals.append({
            "face_key": row["face_key"],
            "node_id": int(row["node_id"]),
            "track_id": row["track_id"],
            "member_face_keys": sorted(members_of_track[row["track_id"]]),
            "identity_id": iid,
            "identity_name": state.identities[iid]["name"],
            "score": verdict.score,
            "margin": round(verdict.score - verdict.runner_up, 6),
            "n_distinct_files": verdict.n_distinct_files,
            "exemplar_face_keys": [member_keys[m]
                                   for m in verdict.exemplar_member_rows],
            "decision": verdict.decision,
            "params_hash": params_hash,
        })
    proposals.sort(key=lambda p: (-p["score"], p["face_key"]))

    out_path = run_dir / "growth_proposals.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for prop in proposals:
            fh.write(json.dumps(prop, sort_keys=True) + "\n")
    tmp.replace(out_path)
    (run_dir / "growth_meta.json").write_text(json.dumps({
        **growth_params, "params_hash": params_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decisions": dict(counts),
    }, indent=1))

    print(f"=== growth done === {dict(counts)}", flush=True)
    print(f"proposals: {out_path} — review at the server's /queue page",
          flush=True)
    return 0


def _leave_one_out(args, embed_dir, locations, held_out, member_matrix,
                   member_identity_idx, member_files, verified_ids,
                   identity_index, tau_attach, blocked_identities):
    if not held_out:
        print("leave-one-out: nothing to hide (need identities spanning "
              ">= 3 source files)", flush=True)
        return 1
    candidates, keys = face_vectors(embed_dir, locations, sorted(held_out))
    blocked = [blocked_identities([k]) for k in keys]
    verdicts = score_candidates(
        candidates, member_matrix, member_identity_idx, member_files,
        tau_attach=tau_attach, margin=args.margin,
        review_band=args.review_band, k=args.k,
        blocked_identity_indices=blocked)
    attach = [v for v in verdicts if v.decision == "attach"]
    correct = sum(1 for v in attach
                  if verified_ids[v.identity_index] == held_out[keys[
                      v.candidate_index]])
    false = len(attach) - correct
    print(json.dumps({
        "held_out": len(keys),
        "attached": len(attach),
        "attached_correct": correct,
        "attached_FALSE": false,
        "reattach_rate": round(correct / len(keys), 4),
        "review": sum(1 for v in verdicts if v.decision == "review"),
        "none": sum(1 for v in verdicts if v.decision == "none"),
    }, indent=1))
    return 0 if false == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
