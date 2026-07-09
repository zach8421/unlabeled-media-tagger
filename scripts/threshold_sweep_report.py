#!/usr/bin/env python
"""Visual threshold sweep: node pairs in similarity bands around candidates.

Renders side-by-side medoid crops for sampled node pairs in +/-band windows
around each candidate tau, plus the same-frame impostor tail. A band where
you see same-person pairs supports lowering tau; different-person pairs mean
tau must stay above that band.

QUARANTINE-AWARE (2026-07-07): the merge loop never sees hub-quarantined
nodes (junk attractors: dark/blurred/occluded low-information embeddings),
so the sweep samples only edges between SURVIVING nodes — the population
whose false-merge risk tau actually controls. The report still states what
fraction of above-tau edges involved a quarantined endpoint, and the
impostor tail is restricted to surviving nodes for the same reason.

Same static-HTML idiom as blur_filter_tune.py; thumbnails are click-to-open.

Usage:
  PYTHONPATH=src ./.venv/bin/python scripts/threshold_sweep_report.py \
      --embed-dir /mnt/media1/folder2_embed \
      --out outputs/threshold_sweep/sweep.html \
      [--taus 0.82 0.86 0.897 0.93] [--band 0.02] [--per-band 40]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

from unlabeled_media_tagger.clustering.knn_graph import knn_edges

STYLE = """
body { font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 26px; }
.stats { background: #fffbe8; border: 1px solid #e5d9a0; border-radius: 8px;
         padding: 10px 14px; font-size: 14px; }
.pair { display: inline-block; border: 1px solid #ddd; border-radius: 8px;
        padding: 6px; margin: 5px; background: #fff; text-align: center; }
.pair img { width: 180px; height: 180px; object-fit: cover; border-radius: 4px; }
.pair .s { font-size: 13px; color: #333; margin-top: 3px; }
.pair .m { font-size: 10px; color: #888; max-width: 380px;
           overflow: hidden; text-overflow: ellipsis; }
.impostor { border-color: #c00; }
"""


def load_nodes(nodes_dir: Path) -> list:
    with (nodes_dir / "nodes.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def crop_url(crops_root: Path, node: dict) -> str:
    path = (crops_root / node["source_file_id"]
            / node["medoid_crop_file_name"]).resolve()
    # Root-relative absolute path (no file:// scheme) so the report works
    # served over HTTP from filesystem root, not just as a local file.
    # Percent-encode: crop names inherit Drive stems with '#' and '%'.
    return html.escape(quote(str(path)))


def render_pair(nodes, crops_root, a, b, sim, impostor=False) -> str:
    cls = "pair impostor" if impostor else "pair"
    node_a, node_b = nodes[a], nodes[b]
    url_a, url_b = crop_url(crops_root, node_a), crop_url(crops_root, node_b)
    return (
        f"<div class='{cls}'>"
        f"<a href='{url_a}' target='_blank'>"
        f"<img src='{url_a}' loading='lazy'></a>"
        f"<a href='{url_b}' target='_blank'>"
        f"<img src='{url_b}' loading='lazy'></a>"
        f"<div class='s'>sim {sim:.4f}</div>"
        f"<div class='m'>{html.escape(node_a['event_path'][:44])} vs "
        f"{html.escape(node_b['event_path'][:44])}</div>"
        f"</div>"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--crops-root", default="/mnt/media1/folder2_detect/crops")
    ap.add_argument("--out", default="outputs/threshold_sweep/sweep.html")
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help="Candidate thresholds (default: calibration tau_edge "
                         "and +/-0.03)")
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--per-band", type=int, default=40)
    ap.add_argument("--impostor-tail", type=int, default=60)
    ap.add_argument("--hub-sim", type=float, default=0.90,
                    help="Edges at/above this similarity count toward hubness")
    ap.add_argument("--hub-degree", type=int, default=20,
                    help="Quarantine nodes with >= this many cross-file "
                         "edges at hub-sim (match run_cluster.py). 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    embed_dir = Path(args.embed_dir)
    nodes_dir = embed_dir / "nodes"
    crops_root = Path(args.crops_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    taus = args.taus
    if taus is None:
        calibration = json.loads((embed_dir / "calibration.json").read_text())
        tau = calibration["tau_edge"]
        taus = [round(tau - 0.03, 3), tau, round(tau + 0.03, 3)]
    tau_main = sorted(taus)[len(taus) // 2]

    nodes = load_nodes(nodes_dir)
    embeddings = np.load(nodes_dir / "node_embeddings.npy")
    rng = random.Random(args.seed)

    floor = min(min(taus) - args.band, args.hub_sim) - 0.01
    print(f"building kNN edges >= {floor:.3f} over {len(nodes)} nodes...",
          flush=True)
    # Exact graph on GPU when available (recall 1.0; HNSW measured 0.90 on
    # this corpus — see knn_graph.py), else knn_edges' own auto-select.
    from unlabeled_media_tagger.clustering.knn_graph import (
        EXACT_FALLBACK_N, ExactGpuIndex, exact_gpu_available)
    index = (ExactGpuIndex(np.ascontiguousarray(embeddings, dtype=np.float32))
             if len(nodes) > EXACT_FALLBACK_N and exact_gpu_available()
             else None)
    edges_i, edges_j, edges_sim = knn_edges(embeddings, k=50, threshold=floor,
                                            index=index)
    print(f"{len(edges_i)} edges", flush=True)

    # Hub quarantine, identical rule to run_cluster.py: cross-file degree at
    # sim >= hub_sim.
    quarantined = np.zeros(len(nodes), dtype=bool)
    if args.hub_degree > 0:
        file_of = np.asarray([n["source_file_id"] for n in nodes])
        strong = edges_sim >= args.hub_sim
        cross = strong & (file_of[edges_i] != file_of[edges_j])
        degree = np.zeros(len(nodes), dtype=np.int32)
        np.add.at(degree, edges_i[cross], 1)
        np.add.at(degree, edges_j[cross], 1)
        quarantined = degree >= args.hub_degree
    edge_quarantined = quarantined[edges_i] | quarantined[edges_j]
    surviving = ~edge_quarantined

    above_tau = edges_sim >= tau_main
    frac_q = (edge_quarantined & above_tau).sum() / max(1, above_tau.sum())
    print(f"quarantined nodes: {int(quarantined.sum())} | edges >= "
          f"{tau_main}: {int(above_tau.sum())}, {frac_q * 100:.1f}% touch a "
          f"quarantined node", flush=True)

    parts = ["<meta charset='utf-8'><title>Threshold sweep</title>",
             f"<style>{STYLE}</style>",
             "<h1>Threshold sweep — are pairs in each band the same person?"
             "</h1>",
             f"<div class='stats'>candidates: {taus} | band ±{args.band} | "
             f"hub quarantine: {int(quarantined.sum())} of {len(nodes)} nodes "
             f"(degree ≥ {args.hub_degree} @ sim ≥ {args.hub_sim}) | "
             f"<b>{frac_q * 100:.1f}%</b> of edges ≥ {tau_main} touch a "
             f"quarantined node and are EXCLUDED below — the merge loop "
             f"never sees them</div>"]

    for tau in sorted(taus):
        in_band = np.where(surviving
                           & (edges_sim >= tau - args.band)
                           & (edges_sim < tau + args.band))[0]
        n_dropped = int(((edges_sim >= tau - args.band)
                         & (edges_sim < tau + args.band)
                         & edge_quarantined).sum())
        picks = rng.sample(in_band.tolist(), min(args.per_band, len(in_band)))
        parts.append(f"<h2>band {tau - args.band:.3f} – {tau + args.band:.3f} "
                     f"(around tau {tau}) — {len(in_band)} surviving edges "
                     f"({n_dropped} quarantined-touching excluded), "
                     f"{len(picks)} sampled</h2>")
        for e in sorted(picks, key=lambda e: -edges_sim[e]):
            parts.append(render_pair(nodes, crops_root, int(edges_i[e]),
                                     int(edges_j[e]), float(edges_sim[e])))

    # Impostor tail audit among SURVIVING nodes: strongest same-frame
    # (guaranteed different-people) pairs the merge loop could actually see.
    # This is the true false-merge risk; artwork/posters vs real faces land
    # here too — check for them explicitly.
    pairs = []
    with (nodes_dir / "same_frame_pairs.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = int(row["node_id_a"]), int(row["node_id_b"])
            if not (quarantined[a] or quarantined[b]):
                pairs.append((a, b))
    if pairs:
        arr = np.asarray(pairs)
        sims = np.einsum("ij,ij->i", embeddings[arr[:, 0]],
                         embeddings[arr[:, 1]])
        top = np.argsort(-sims)[: args.impostor_tail]
        parts.append(
            f"<h2>Surviving-nodes impostor tail (top {len(top)} of "
            f"{len(pairs)} same-frame pairs; red border = guaranteed "
            f"different people — or duplicate-person-in-frame/artwork: "
            f"judge each)</h2>")
        for e in top:
            parts.append(render_pair(nodes, crops_root, int(arr[e, 0]),
                                     int(arr[e, 1]), float(sims[e]),
                                     impostor=True))

    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path} (open file://{out_path.resolve()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
