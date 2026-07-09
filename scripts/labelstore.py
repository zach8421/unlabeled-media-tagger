#!/usr/bin/env python
"""Label-store CLI: rebuild, stats, log tail, undo, verify.

  PYTHONPATH=src ./.venv/bin/python scripts/labelstore.py <cmd> [args]
    rebuild                rebuild derived/ views from the event log
    stats                  identity/assignment/constraint counts
    log [--tail N]         print recent events
    undo EVENT_ID          append a revoke for EVENT_ID
    verify [--repair]      integrity check; --repair truncates a torn last
                           line (after backing the log up)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone

from unlabeled_media_tagger.review.store import LabelStore


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["rebuild", "stats", "log", "undo", "verify"])
    ap.add_argument("argument", nargs="?", default=None)
    ap.add_argument("--labels-dir", default="/mnt/media1/folder2_labels")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args(argv)

    store = LabelStore(args.labels_dir)

    if args.command == "rebuild":
        state = store.rebuild_derived()
        print(f"rebuilt derived/ — {len(state.identities)} identities, "
              f"{len(state.assignments)} assignments")
        return 0

    if args.command == "stats":
        state = store.replay()
        print(json.dumps({
            "events_active": state.n_events_active,
            "identities": len(state.identities),
            "assignments": len(state.assignments),
            "confirms": len(state.confirms),
            "rejected_clusters": len(state.rejected_clusters),
            "cannot_link_face_identity":
                len(state.cannot_link_face_identity),
            "cannot_link_face_face": len(state.cannot_link_face_face),
        }, indent=1))
        return 0

    if args.command == "log":
        for event in store.events()[-args.tail:]:
            print(f"{event['ts']}  {event['event_id']}  {event['actor']:6s} "
                  f"{event['type']:18s} "
                  f"{json.dumps(event['payload'])[:110]}")
        return 0

    if args.command == "undo":
        if not args.argument:
            print("usage: labelstore.py undo EVENT_ID")
            return 2
        event = store.append("revoke", {"target_event_id": args.argument})
        print(f"revoked {args.argument} via {event['event_id']}")
        store.rebuild_derived()
        return 0

    if args.command == "verify":
        report = store.verify()
        print(json.dumps(report, indent=1))
        if report.get("torn_tail") and args.repair:
            backup = store.events_path.with_suffix(
                ".jsonl.bak." + datetime.now(timezone.utc)
                .strftime("%Y%m%dT%H%M%S"))
            shutil.copyfile(store.events_path, backup)
            raw = store.events_path.read_bytes()
            fixed = raw[: raw.rfind(b"\n") + 1]
            store.events_path.write_bytes(fixed)
            print(f"repaired: torn tail truncated (backup at {backup})")
        healthy = (report.get("replay_ok")
                   and not report.get("torn_tail")
                   and not report.get("bad_lines"))
        return 0 if healthy else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
