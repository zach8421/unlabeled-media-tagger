"""Label-free threshold calibration from the data's own structure.

Two empirical similarity distributions exist without any human labels:

  IMPOSTOR (different people): same-frame co-occurring pairs — two faces in
  one frame are different people by construction. These share lighting, blur
  and compression, so they are HARD negatives; a threshold cleared by their
  tail is conservative in exactly the direction the precision-first goal
  wants. Measured on node embeddings (same vectors clustering will compare).

  GENUINE (same person): within-track face pairs — build_tracks' strict gates
  make these near-guaranteed same-person. These are EASY positives (same
  video, same session), so the genuine-FNR reported at the chosen threshold
  understates cross-video FNR. Accepted: recall is recovered by the growth
  pass and human merges, never by loosening the first pass.

CONTAMINATION (measured on FOLDER 2, 2026-07-06): the same-frame axiom is
violated for ~7% of pairs at high similarity — the same person appears twice
in one frame with DISJOINT boxes (96% of sim>0.9 same-frame pairs have zero
bbox overlap: projection screens behind speakers, picture-in-picture,
side-by-side broadcast layouts). Untreated, this drags P99.9 to ~0.995 and
makes the threshold useless. Pairs above ``impostor_cap`` (default 0.9) are
therefore excluded as presumed same-person before taking the tail; the raw
percentiles are still reported, and the threshold_sweep_report.py impostor
audit is the human check on this assumption.

Recommendation rule: tau_edge = P99.9(capped impostor bulk), merge gates
anchored at recluster.py's proven offsets (mean = tau - 0.04,
min = tau - 0.12). Acceptance for a clean calibration: tau < genuine P5 —
on FOLDER 2 this does NOT hold (distributions overlap); the visual sweep is
the deciding gate, per the precision-first design.

Run:  PYTHONPATH=src ./.venv/bin/python -m \
          unlabeled_media_tagger.clustering.calibrate --embed-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MEAN_OFFSET = 0.04  # recluster.py: edge 0.74 / mean 0.70
MIN_OFFSET = 0.12  # recluster.py: edge 0.74 / min 0.62

IMPOSTOR_PERCENTILES = (50, 90, 99, 99.5, 99.9, 99.99, 100)
GENUINE_PERCENTILES = (0, 1, 5, 10, 25, 50)


def impostor_similarities(nodes_dir: Path) -> np.ndarray:
    embeddings = np.load(nodes_dir / "node_embeddings.npy", mmap_mode="r")
    pairs_a, pairs_b = [], []
    with (nodes_dir / "same_frame_pairs.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            pairs_a.append(int(row["node_id_a"]))
            pairs_b.append(int(row["node_id_b"]))
    sims = np.empty(len(pairs_a), dtype=np.float32)
    for start in range(0, len(pairs_a), 65_536):
        a = np.asarray(embeddings[pairs_a[start:start + 65_536]])
        b = np.asarray(embeddings[pairs_b[start:start + 65_536]])
        sims[start:start + len(a)] = np.einsum("ij,ij->i", a, b)
    return sims


def genuine_similarities(embed_dir: Path, *, max_tracks: int = 50_000,
                         pairs_per_track: int = 12, seed: int = 0) -> np.ndarray:
    """Within-track face-pair similarities from a random sample of tracks."""
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    with (embed_dir / "tracks" / "track_members.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            members[row["track_id"]].append(
                (row["source_file_id"], row["crop_file_name"]))
    multi = [k for k, v in members.items() if len(v) > 1]
    if len(multi) > max_tracks:
        multi = list(rng.choice(multi, size=max_tracks, replace=False))
    wanted = {face for track in multi for face in members[track]}

    location = {}
    with (embed_dir / "faces_index.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["source_file_id"], row["crop_file_name"])
            if key in wanted:
                location[key] = (int(row["shard"]), int(row["row"]))

    shard_cache = {}

    def vector(key):
        shard, row = location[key]
        mm = shard_cache.get(shard)
        if mm is None:
            mm = np.load(embed_dir / "embeddings" / f"shard_{shard:05d}.npy",
                         mmap_mode="r")
            shard_cache[shard] = mm
        return mm[row]

    sims = []
    for track in multi:
        faces = members[track]
        pair_count = len(faces) * (len(faces) - 1) // 2
        all_pairs = [(a, b) for i, a in enumerate(faces)
                     for b in faces[i + 1:]]
        if pair_count > pairs_per_track:
            picks = rng.choice(pair_count, size=pairs_per_track, replace=False)
            all_pairs = [all_pairs[p] for p in picks]
        for a, b in all_pairs:
            sims.append(float(np.dot(vector(a), vector(b))))
    return np.asarray(sims, dtype=np.float32)


DEFAULT_IMPOSTOR_CAP = 0.9


def calibrate(embed_dir: Path, *, impostor_cap: float = DEFAULT_IMPOSTOR_CAP,
              **genuine_kwargs) -> dict:
    nodes_dir = embed_dir / "nodes"
    impostor = impostor_similarities(nodes_dir)
    genuine = genuine_similarities(embed_dir, **genuine_kwargs)

    bulk = impostor[impostor <= impostor_cap]
    imp_pct = {f"p{p}": round(float(np.percentile(impostor, p)), 4)
               for p in IMPOSTOR_PERCENTILES}
    bulk_pct = {f"p{p}": round(float(np.percentile(bulk, p)), 4)
                for p in (99, 99.5, 99.9)}
    gen_pct = {f"p{p}": round(float(np.percentile(genuine, p)), 4)
               for p in GENUINE_PERCENTILES}

    tau_edge = float(np.percentile(bulk, 99.9))
    report = {
        "n_impostor_pairs": int(len(impostor)),
        "n_impostor_bulk": int(len(bulk)),
        "impostor_cap": impostor_cap,
        "excluded_as_contamination": int(len(impostor) - len(bulk)),
        "n_genuine_pairs": int(len(genuine)),
        "impostor_percentiles_raw": imp_pct,
        "impostor_percentiles_bulk": bulk_pct,
        "genuine_percentiles": gen_pct,
        "tau_edge": round(tau_edge, 4),
        "merge_mean": round(tau_edge - MEAN_OFFSET, 4),
        "merge_min": round(tau_edge - MIN_OFFSET, 4),
        "genuine_fnr_at_tau_edge":
            round(float((genuine < tau_edge).mean()), 4),
        "separation_ok": bool(tau_edge < float(np.percentile(genuine, 5))),
    }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", default="/mnt/media1/folder2_embed")
    ap.add_argument("--out", default=None,
                    help="Write the report JSON here (default: "
                         "<embed-dir>/calibration.json)")
    ap.add_argument("--max-tracks", type=int, default=50_000)
    ap.add_argument("--pairs-per-track", type=int, default=12)
    ap.add_argument("--impostor-cap", type=float,
                    default=DEFAULT_IMPOSTOR_CAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    embed_dir = Path(args.embed_dir)
    report = calibrate(embed_dir, impostor_cap=args.impostor_cap,
                       max_tracks=args.max_tracks,
                       pairs_per_track=args.pairs_per_track, seed=args.seed)

    print(json.dumps(report, indent=1))
    out = Path(args.out) if args.out else embed_dir / "calibration.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    if not report["separation_ok"]:
        print("WARNING: impostor P99.9 >= genuine P5 — distributions overlap; "
              "inspect the top impostor pairs visually before trusting "
              "tau_edge (threshold_sweep_report.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
