#!/usr/bin/env python
"""Read-only review-state report: store replay x one run's triage/proposals.

The iterate loop (cluster -> review -> growth -> /queue -> repeat) keeps
asking "where do things stand?" — this answers it in one shot: global store
state, per-run confirm/reject counts, triage-queue progress by section,
growth-proposal profile, and the standing sanity checks (flagged confirms,
duplicate names, single-file identities). Never writes anything.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/review_status.py \
      --run-dir /mnt/media1/folder2_cluster/run_002
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from unlabeled_media_tagger.review.store import LabelStore

SUSPECT_FLAGS = ("degenerate_suspect", "sibling_export_suspect",
                 "static_face_suspect")
TRIAGE_SECTIONS = ("suspect_static", "suspect_degenerate", "suspect_sibling",
                   "largest_faces", "largest_events", "random_uniform",
                   "random_size_weighted")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels-dir", default="/mnt/media1/folder2_labels")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    run_id = run_dir.name

    store = LabelStore(args.labels_dir)
    state = store.replay()

    canonical = {i: state.canonical_identity(i) for i in state.identities}
    active_ids = {c for i, c in canonical.items()
                  if not state.identities.get(c, {}).get("retired")}
    named = {i for i in active_ids if state.identities[i].get("name")}
    faces_of = defaultdict(set)
    files_of = defaultdict(set)
    for face_key, a in state.assignments.items():
        cid = state.canonical_identity(a["identity_id"])
        faces_of[cid].add(face_key)
        files_of[cid].add(face_key.split("/", 1)[0])

    print("=== store ===")
    print(f"events: {len(store.events())} ({state.n_events_active} active)")
    print(f"identities: {len(active_ids)} active ({len(named)} named, "
          f"{sum(1 for i, c in canonical.items() if i != c)} merged away)")
    print(f"assignments: {len(state.assignments)} faces by source "
          f"{dict(Counter(a['source'] for a in state.assignments.values()))}")
    print(f"cannot-links: {len(state.cannot_link_face_identity)} "
          f"face-identity, {len(state.cannot_link_face_face)} face-face")
    spread = Counter(min(len(files_of[i]), 3) for i in active_ids
                     if faces_of[i])
    print(f"identities by distinct files: 1={spread[1]} 2={spread[2]} "
          f"3+={spread[3]}  (only 2+ can auto-attach in growth)")

    print("\n=== review decisions (last wins) ===")
    by_run = defaultdict(Counter)
    for (run, cluster_id), decision in state.cluster_review.items():
        by_run[run][decision] += 1
    for run in sorted(by_run):
        marker = "  <- this run" if run == run_id else ""
        print(f"{run}: {dict(by_run[run])}{marker}")
    status = {cid: d for (r, cid), d in state.cluster_review.items()
              if r == run_id}

    summary_path = run_dir / "cluster_summary.csv"
    summary = {}
    if summary_path.exists():
        with summary_path.open(newline="") as fh:
            summary = {r["cluster_id"]: r for r in csv.DictReader(fh)}

    triage_path = run_dir / "triage.csv"
    if triage_path.exists():
        with triage_path.open(newline="") as fh:
            triage = list(csv.DictReader(fh))
        print(f"\n=== triage progress ({run_id}) ===")
        per_section = defaultdict(Counter)
        for t in triage:
            per_section[t["section"]][status.get(t["cluster_id"],
                                                 "unreviewed")] += 1
        for section in TRIAGE_SECTIONS:
            if per_section[section]:
                print(f"{section:>22}: {dict(per_section[section])}")
        n_done = sum(1 for t in triage if t["cluster_id"] in status)
        print(f"triage reviewed: {n_done}/{len(triage)}")
        outside = {c: d for c, d in status.items()
                   if c not in {t["cluster_id"] for t in triage}}
        print(f"reviewed outside triage: {len(outside)} "
              f"{dict(Counter(outside.values()))}")

    proposals_path = run_dir / "growth_proposals.jsonl"
    if proposals_path.exists():
        with proposals_path.open() as fh:
            proposals = [json.loads(line) for line in fh]
        decided = {p["face_key"] for p in proposals
                   if p["face_key"] in state.assignments
                   or any(p["face_key"] == fk for fk, _
                          in state.cannot_link_face_identity)}
        print(f"\n=== growth queue ({run_id}) ===")
        for decision in ("attach", "review"):
            subset = [p for p in proposals if p["decision"] == decision]
            pending = sum(1 for p in subset if p["face_key"] not in decided)
            print(f"{decision:>7}: {len(subset)} proposals, "
                  f"{pending} pending")

    print("\n=== sanity checks ===")
    confirm_identity = {}
    for confirm in state.confirms:
        if confirm["run_id"] == run_id:
            confirm_identity[str(confirm["cluster_id"])] = \
                state.canonical_identity(confirm["identity_id"])
    flagged = defaultdict(list)
    for cluster_id, decision in status.items():
        if decision != "confirmed":
            continue
        row = summary.get(cluster_id, {})
        for flag in SUSPECT_FLAGS:
            if row.get(flag) == "true":
                flagged[flag].append(cluster_id)
    print(f"confirmed clusters with suspect flags: "
          f"{ {f: len(cs) for f, cs in flagged.items()} or 0}")
    for cluster_id in flagged.get("static_face_suspect", []):
        print(f"  NOTE static-flagged confirm: cluster {cluster_id} "
              f"(artwork? re-check)")

    names_seen = defaultdict(list)
    for i in named:
        names_seen[state.identities[i]["name"].strip().lower()].append(i)
    dupes = {n: ids for n, ids in names_seen.items() if len(ids) > 1}
    print(f"duplicate names on distinct identities: {len(dupes)}")
    for name, ids in sorted(dupes.items()):
        print(f"  {name!r}: {ids} — same person? merge in the index UI")
    single = [i for i in active_ids
              if faces_of[i] and len(files_of[i]) == 1]
    print(f"single-file identities (review-band only, no auto-attach): "
          f"{len(single)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
