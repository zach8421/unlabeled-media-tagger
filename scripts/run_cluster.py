#!/usr/bin/env python
"""High-precision first-pass clustering over the node layer.

Wires together knn_graph -> constraints -> sparse_cluster and writes one
cluster run directory. Re-runnable: confirmed identities never resurface
because nodes with any human-assigned member face are excluded up front
(--labels-dir); below-threshold nodes land in the unassigned pool
(assigned=False rows), never force-assigned.

Pilot runs: --event-filter SUBSTR (repeatable) clusters only nodes whose
event_path contains a substring — the M4 gate pilots 3-5 events end-to-end
before any full run.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/run_cluster.py \
      --embed-dir /mnt/media1/folder2_embed \
      --out-dir /mnt/media1/folder2_cluster/run_001 \
      [--calibration /mnt/media1/folder2_embed/calibration.json] \
      [--labels-dir /mnt/media1/folder2_labels] \
      [--event-filter 'SOME EVENT FOLDER' ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unlabeled_media_tagger.clustering import constraints as constraints_mod
from unlabeled_media_tagger.clustering.knn_graph import (
    ann_recall,
    build_hnsw_index,
    knn_edges,
)
from unlabeled_media_tagger.clustering.sparse_cluster import sparse_cluster

CLUSTERS_COLUMNS = [
    "node_id", "track_id", "source_file_id", "media_type", "n_faces",
    "medoid_crop_file_name", "face_key",
    "cluster_id", "cluster_label", "assigned", "quarantined",
    "similarity_to_cluster", "event_path",
]


def load_nodes(nodes_dir: Path) -> list:
    with (nodes_dir / "nodes.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def assigned_node_ids(labels_dir, tracks_dir, node_id_of_track) -> set:
    """Nodes with any member face already assigned to an identity."""
    if labels_dir is None:
        return set()
    path = Path(labels_dir) / "derived" / "assignments.csv"
    if not path.exists():
        return set()
    assigned_faces = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            assigned_faces.add(row["face_key"])
    node_of_face = constraints_mod.load_node_of_face(
        tracks_dir, node_id_of_track)
    return {node_of_face[f] for f in assigned_faces if f in node_of_face}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--calibration", default=None,
                    help="calibration.json to take thresholds from")
    ap.add_argument("--tau-edge", type=float, default=None)
    ap.add_argument("--merge-mean", type=float, default=None)
    ap.add_argument("--merge-min", type=float, default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--knn-backend", default="auto",
                    choices=["auto", "exact-gpu", "hnsw", "exact"],
                    help="kNN backend above the small-set exact fallback. "
                         "exact-gpu: chunked GEMM top-k via TF (recall 1.0, "
                         "~1 min at 350K nodes); exact: same on CPU (~20 "
                         "min at 350K); hnsw: faiss HNSW + recall gate "
                         "(measured 0.90 on FOLDER 2 — hub nodes break "
                         "HNSW navigability; see knn_graph.py). auto: "
                         "exact-gpu when a GPU is visible, else hnsw.")
    ap.add_argument("--ef-search", type=int, default=None,
                    help="HNSW efSearch override (hnsw backend only)")
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--min-cluster-size-image", type=int, default=2)
    ap.add_argument("--size-cap", type=int, default=20_000)
    ap.add_argument("--recall-sample", type=int, default=5000)
    ap.add_argument("--degenerate-sim", type=float, default=0.97,
                    help="Flag clusters with mean similarity above this...")
    ap.add_argument("--degenerate-files", type=int, default=50,
                    help="...across more than this many source files")
    ap.add_argument("--static-sim", type=float, default=0.985,
                    help="A video track with >= --static-min-faces frames "
                         "whose min intra-track similarity is above this is "
                         "'static': the face did not change across 15+ s of "
                         "footage — posters, murals, sketches, memorial "
                         "photos. Live faces essentially never hold this "
                         "(0.5%% in files with confirmed people vs 1.2%% "
                         "corpus-wide). Feeds the static_face_suspect badge.")
    ap.add_argument("--static-min-faces", type=int, default=4)
    ap.add_argument("--hub-degree", type=int, default=20,
                    help="Quarantine nodes with at least this many CROSS-FILE "
                         "edges in the kNN graph before clustering. Real "
                         "tracks essentially never have 20+ cross-file "
                         "neighbors above ~0.9 (measured: 0.1%%); "
                         "low-information degenerate embeddings (dark/blur/"
                         "occluded crops) have hundreds. 0 disables.")
    ap.add_argument("--event-filter", action="append", default=None,
                    help="Cluster only nodes whose event_path contains this "
                         "substring (repeatable; pilot runs)")
    args = ap.parse_args(argv)

    embed_dir = Path(args.embed_dir)
    nodes_dir = embed_dir / "nodes"
    tracks_dir = embed_dir / "tracks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Thresholds: CLI > calibration file > error.
    calibration = {}
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text())
    elif (embed_dir / "calibration.json").exists():
        calibration = json.loads((embed_dir / "calibration.json").read_text())
    tau_edge = args.tau_edge if args.tau_edge is not None \
        else calibration.get("tau_edge")
    merge_mean = args.merge_mean if args.merge_mean is not None \
        else calibration.get("merge_mean")
    merge_min = args.merge_min if args.merge_min is not None \
        else calibration.get("merge_min")
    if None in (tau_edge, merge_mean, merge_min):
        print("No thresholds: pass --tau-edge/--merge-mean/--merge-min or "
              "run calibrate first (calibration.json).", flush=True)
        return 2

    # The review server appends events but only rebuilds derived/ at its own
    # startup — rebuild here so confirms/removals made minutes ago count.
    if args.labels_dir and Path(args.labels_dir).exists():
        from unlabeled_media_tagger.review.store import LabelStore

        LabelStore(args.labels_dir).rebuild_derived()

    nodes = load_nodes(nodes_dir)
    embeddings = np.load(nodes_dir / "node_embeddings.npy")
    print(f"nodes: {len(nodes)}  thresholds: edge={tau_edge} "
          f"mean={merge_mean} min={merge_min}", flush=True)

    node_id_of_track = {row["track_id"]: int(row["node_id"]) for row in nodes}

    # Active set: event filter (pilot) minus already-assigned nodes.
    active_mask = np.ones(len(nodes), dtype=bool)
    if args.event_filter:
        filters = [f.lower() for f in args.event_filter]
        for i, row in enumerate(nodes):
            event = row["event_path"].lower()
            active_mask[i] = any(f in event for f in filters)
        print(f"event filter: {int(active_mask.sum())} nodes match "
              f"{args.event_filter}", flush=True)
    already = assigned_node_ids(args.labels_dir, tracks_dir, node_id_of_track)
    for node_id in already:
        active_mask[node_id] = False
    if already:
        print(f"excluded {len(already)} nodes with assigned member faces",
              flush=True)

    active_ids = np.where(active_mask)[0]
    compact_of_node = {int(n): c for c, n in enumerate(active_ids)}
    active_emb = np.ascontiguousarray(embeddings[active_ids])

    t0 = time.time()
    from unlabeled_media_tagger.clustering.knn_graph import (
        EXACT_FALLBACK_N, ExactGpuIndex, exact_gpu_available)
    backend = args.knn_backend
    if backend == "auto":
        backend = "exact-gpu" if exact_gpu_available() else "hnsw"
    if len(active_ids) <= EXACT_FALLBACK_N or backend == "exact":
        # Pilot-sized sets: exact kNN is seconds and HNSW recall on small
        # sets is unreliable — recall is 1.0 by construction.
        backend = "exact"
        edges = knn_edges(active_emb, k=args.k, threshold=tau_edge,
                          exact=True)
        recall = 1.0
        print(f"kNN graph (exact CPU): {len(edges[0])} edges >= "
              f"{tau_edge} in {time.time() - t0:.0f}s", flush=True)
    elif backend == "exact-gpu":
        index = ExactGpuIndex(active_emb)
        recall = 1.0  # exact by construction — no recall gate needed
        edges = knn_edges(active_emb, k=args.k, threshold=tau_edge,
                          index=index)
        print(f"kNN graph (exact GPU): {len(edges[0])} edges >= {tau_edge} "
              f"in {time.time() - t0:.0f}s | recall 1.0 by construction",
              flush=True)
    else:
        hnsw_kwargs = ({"ef_search": args.ef_search}
                       if args.ef_search else {})
        index = build_hnsw_index(active_emb, **hnsw_kwargs)
        recall = ann_recall(active_emb, index, k=args.k,
                            sample=args.recall_sample)
        edges = knn_edges(active_emb, k=args.k, threshold=tau_edge,
                          index=index)
        print(f"kNN graph: {len(edges[0])} edges >= {tau_edge} in "
              f"{time.time() - t0:.0f}s | ANN recall@{args.k} = {recall:.4f}"
              f"{'  (LOW — use --knn-backend exact-gpu)' if recall < 0.98 else ''}",
              flush=True)

    # Hubness quarantine: a node with many cross-file edges this strong is a
    # junk attractor member, not a person (see --hub-degree help). Applied to
    # the edge list BEFORE the merge loop so quarantined nodes neither merge
    # nor land in the unassigned pool (growth must not see them either).
    quarantined_compact: set = set()
    if args.hub_degree > 0 and len(edges[0]) > 0:
        edge_i, edge_j, edge_sim = edges
        file_of = np.asarray(
            [nodes[int(active_ids[c])]["source_file_id"]
             for c in range(len(active_ids))])
        cross = file_of[edge_i] != file_of[edge_j]
        degree = np.zeros(len(active_ids), dtype=np.int32)
        np.add.at(degree, edge_i[cross], 1)
        np.add.at(degree, edge_j[cross], 1)
        quarantined_compact = set(
            np.where(degree >= args.hub_degree)[0].tolist())
        if quarantined_compact:
            keep = ~(np.isin(edge_i, list(quarantined_compact))
                     | np.isin(edge_j, list(quarantined_compact)))
            edges = (edge_i[keep], edge_j[keep], edge_sim[keep])
            print(f"hub quarantine: {len(quarantined_compact)} nodes with "
                  f">= {args.hub_degree} cross-file edges removed "
                  f"({int((~keep).sum())} edges dropped)", flush=True)

    pairs, impure_faces, constraint_counts = \
        constraints_mod.resolve_constraints(
            nodes_dir, tracks_dir, node_id_of_track, args.labels_dir)
    compact_pairs = [(compact_of_node[a], compact_of_node[b])
                     for a, b in pairs
                     if a in compact_of_node and b in compact_of_node]
    print(f"constraints: {constraint_counts} "
          f"({len(compact_pairs)} pairs active)", flush=True)
    if impure_faces:
        impure_path = out_dir / "track_impure_face_keys.txt"
        impure_path.write_text("\n".join(sorted(impure_faces)) + "\n")
        print(f"NOTE: {len(impure_faces)} face_keys have cannot-links inside "
              f"one track -> {impure_path}; feed to build_tracks.py "
              f"--exclude-keys on next track rebuild.", flush=True)

    t0 = time.time()
    result = sparse_cluster(
        active_emb, *edges,
        cannot_link_pairs=compact_pairs,
        merge_min=merge_min, merge_mean=merge_mean,
        min_cluster_size=2,  # engine floor; per-media policy applied below
        size_cap=args.size_cap,
    )
    print(f"merge loop: {time.time() - t0:.0f}s  {result.stats}", flush=True)

    # Per-media minimum cluster size: video-bearing clusters need
    # min_cluster_size nodes; image-only clusters min_cluster_size_image.
    members_of = defaultdict(list)
    for compact, label in enumerate(result.labels.tolist()):
        if label >= 0:
            members_of[label].append(compact)
    keep = {}
    for label, compacts in members_of.items():
        has_video = any(
            nodes[int(active_ids[c])]["media_type"] == "video"
            for c in compacts)
        floor = args.min_cluster_size if has_video \
            else args.min_cluster_size_image
        if len(compacts) >= floor:
            keep[label] = len(keep)

    clusters_path = out_dir / "clusters.csv"
    label_counts = Counter()
    with clusters_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLUSTERS_COLUMNS)
        writer.writeheader()
        for compact, node_idx in enumerate(active_ids.tolist()):
            row = nodes[node_idx]
            quarantined = compact in quarantined_compact
            raw_label = int(result.labels[compact])
            label = (keep.get(raw_label, -1)
                     if raw_label >= 0 and not quarantined else -1)
            label_counts[label >= 0] += 1
            writer.writerow({
                "node_id": row["node_id"],
                "track_id": row["track_id"],
                "source_file_id": row["source_file_id"],
                "media_type": row["media_type"],
                "n_faces": row["n_faces"],
                "medoid_crop_file_name": row["medoid_crop_file_name"],
                "face_key":
                    f"{row['source_file_id']}/{row['medoid_crop_file_name']}",
                "cluster_id": label if label >= 0 else "",
                "cluster_label": f"person_{label:05d}" if label >= 0 else "",
                "assigned": str(label >= 0).lower(),
                "quarantined": str(quarantined).lower(),
                "similarity_to_cluster":
                    round(float(result.similarity_to_cluster[compact]), 6)
                    if label >= 0 else "",
                "event_path": row["event_path"],
            })

    # Cluster-level summary for review ranking. degenerate_suspect flags the
    # junk-attractor signature: many source files whose node vectors sit
    # implausibly close to one centroid. Genuine cross-file track sims run
    # ~0.7-0.9 (calibration), so mean > 0.97 across > 50 files is physically
    # implausible for a real person — but routine for low-information
    # embeddings (dark/blurred/extreme-profile crops, scenes before the
    # confidence gate). Verified on pilot_002: the two impure mega-clusters
    # sat at 0.977/0.982 over 401/582 files.
    # Static (artwork-suspect) nodes: frozen embedding across many frames.
    static_node_ids = {
        int(row["node_id"]) for row in nodes
        if row["media_type"] == "video"
        and int(row["n_faces"]) >= args.static_min_faces
        and float(row["min_intra_sim"] or 0) >= args.static_sim}

    summary = defaultdict(lambda: {"n_nodes": 0, "n_faces": 0, "sim_sum": 0.0,
                                   "files": set(), "events": set(),
                                   "frame_tokens": Counter(),
                                   "static": 0, "multi_frame": 0})
    frame_re = re.compile(r"_f(\d{6})")
    with clusters_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["cluster_id"] == "":
                continue
            s = summary[int(row["cluster_id"])]
            s["n_nodes"] += 1
            s["n_faces"] += int(row["n_faces"])
            s["sim_sum"] += float(row["similarity_to_cluster"])
            s["files"].add(row["source_file_id"])
            s["events"].add(row["event_path"])
            if (row["media_type"] == "video"
                    and int(row["n_faces"]) >= args.static_min_faces):
                s["multi_frame"] += 1
                s["static"] += int(row["node_id"]) in static_node_ids
            match = frame_re.search(row["medoid_crop_file_name"])
            if match:
                s["frame_tokens"][match.group(1)] += 1
    n_degenerate = n_sibling = n_static = 0
    with (out_dir / "cluster_summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "cluster_label", "n_nodes", "n_faces",
                         "n_files", "n_events", "mean_similarity",
                         "degenerate_suspect", "sibling_export_suspect",
                         "static_face_suspect", "example_events"])
        for cid in sorted(summary):
            s = summary[cid]
            mean_sim = s["sim_sum"] / s["n_nodes"]
            degenerate = (mean_sim > args.degenerate_sim
                          and len(s["files"]) > args.degenerate_files)
            n_degenerate += degenerate
            # Sibling-export / overlay signature: most members are the SAME
            # source moment (identical frame token) across different files —
            # sibling exports of one clip, often a title card covering the
            # face (the embedding is the overlay, not a person).
            top_token = (s["frame_tokens"].most_common(1)[0][1]
                         if s["frame_tokens"] else 0)
            sibling = (s["n_nodes"] >= 5
                       and top_token / s["n_nodes"] >= 0.7
                       and len(s["files"]) >= 3)
            n_sibling += sibling
            static = (s["static"] >= 1
                      and s["static"] / max(1, s["multi_frame"]) >= 0.5)
            n_static += static
            writer.writerow([cid, f"person_{cid:05d}", s["n_nodes"],
                             s["n_faces"], len(s["files"]), len(s["events"]),
                             round(mean_sim, 4), str(degenerate).lower(),
                             str(sibling).lower(), str(static).lower(),
                             " | ".join(sorted(s["events"])[:3])])
    if n_sibling:
        print(f"NOTE: {n_sibling} cluster(s) flagged sibling_export_suspect "
              f"(one repeated frame moment across sibling files — often a "
              f"graphic overlay).", flush=True)
    if n_static:
        print(f"NOTE: {n_static} cluster(s) flagged static_face_suspect "
              f"(frozen embedding across 15+ s — posters/murals/sketches/"
              f"memorial photos).", flush=True)
    if n_degenerate:
        print(f"WARNING: {n_degenerate} cluster(s) flagged degenerate_suspect "
              f"(mean sim > {args.degenerate_sim} across > "
              f"{args.degenerate_files} files) — inspect before confirming; "
              f"do not feed to growth unvetted.", flush=True)

    params = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embed_dir": str(embed_dir),
        "labels_dir": args.labels_dir,
        "event_filter": args.event_filter,
        "tau_edge": tau_edge, "merge_mean": merge_mean,
        "merge_min": merge_min, "k": args.k,
        "min_cluster_size": args.min_cluster_size,
        "min_cluster_size_image": args.min_cluster_size_image,
        "size_cap": args.size_cap,
        "hub_degree": args.hub_degree,
        "n_nodes_quarantined_hub": len(quarantined_compact),
        "knn_backend": backend,
        "ann_recall": round(recall, 4),
        "n_nodes_total": len(nodes),
        "n_nodes_active": int(len(active_ids)),
        "n_nodes_excluded_assigned": len(already),
        "constraints": constraint_counts,
        "engine_stats": result.stats,
        "n_clusters": len(keep),
        "n_assigned_nodes": label_counts[True],
        "n_unassigned_nodes": label_counts[False],
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=1))

    print(f"\n=== cluster run done === clusters={len(keep)} "
          f"assigned={label_counts[True]} unassigned={label_counts[False]}",
          flush=True)
    print(f"output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
