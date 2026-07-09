"""Triage-queue selection over a cluster run's summary rows.

23K clusters cannot be reviewed exhaustively; growth only needs a vetted
seed set. This picks a ~200-300 cluster review budget in priority order:

  1. suspect_static / suspect_degenerate — every flagged cluster (rare,
     high-yield: the flags exist because these fooled the pilot).
  2. suspect_sibling — a stratified sample of the sibling-export overlay
     flags. The low-similarity tail (mean_similarity <= sibling_sim_tail)
     is taken in full first: a sibling flag on a *loosely* coherent cluster
     is the likeliest false positive, and false positives there are real
     people we would otherwise skip. The rest is spread evenly across
     single- vs multi-event and small vs large strata so one failure mode
     can't hide in an unsampled corner.
  3. largest_faces / largest_events — the head of the index by n_faces and
     by n_events. Errors here contaminate the most faces and the most
     events, and these clusters are growth's strongest attractors.
  4. random_uniform — an unbiased uniform sample of everything unpicked;
     reviewing U of them bounds cluster-level purity (n=100 pure at 95%
     observed gives a 95% CI of roughly +/-4pp).
  5. random_size_weighted — a probability-proportional-to-n_faces sample of
     the remainder, for a *face-weighted* purity estimate (big clusters
     carry more faces; uniform sampling underweights them).

Selection is seeded and first-section-wins deduped, so the same summary +
seed always yields the same queue.
"""

from __future__ import annotations

import numpy as np

SIBLING_STRATA = (
    ("single-event small", lambda r: r["_events"] <= 1 and r["_nodes"] < 8),
    ("single-event large", lambda r: r["_events"] <= 1 and r["_nodes"] >= 8),
    ("multi-event small", lambda r: r["_events"] > 1 and r["_nodes"] < 8),
    ("multi-event large", lambda r: r["_events"] > 1 and r["_nodes"] >= 8),
)


def build_triage(summary_rows: list, *, top_faces: int = 50,
                 top_events: int = 20, n_random: int = 100,
                 n_size_weighted: int = 25, sibling_sample: int = 60,
                 sibling_sim_tail: float = 0.97, seed: int = 0) -> list:
    """Return ordered triage rows: summary dicts + section/reason keys."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in summary_rows:
        row = dict(row)
        row["_nodes"] = int(row["n_nodes"])
        row["_faces"] = int(row["n_faces"])
        row["_events"] = int(row["n_events"])
        row["_sim"] = float(row.get("mean_similarity") or 0)
        rows.append(row)

    picked: dict = {}  # cluster_id -> (section, reason); first section wins

    def take(row, section, reason):
        if row["cluster_id"] not in picked:
            picked[row["cluster_id"]] = (row, section, reason)

    for row in rows:
        if row.get("static_face_suspect") == "true":
            take(row, "suspect_static",
                 "static_face_suspect: frozen embedding across 15+ s")
        if row.get("degenerate_suspect") == "true":
            take(row, "suspect_degenerate",
                 f"degenerate_suspect: mean sim {row['_sim']:.3f} across "
                 f"{row['n_files']} files")

    siblings = [r for r in rows if r.get("sibling_export_suspect") == "true"
                and r["cluster_id"] not in picked]
    tail = [r for r in siblings if r["_sim"] <= sibling_sim_tail]
    if len(tail) > sibling_sample:
        idx = rng.choice(len(tail), size=sibling_sample, replace=False)
        tail = [tail[i] for i in sorted(idx.tolist())]
    for row in tail:
        take(row, "suspect_sibling",
             f"sibling flag on low-sim cluster (mean {row['_sim']:.3f}) — "
             f"likeliest false positive")
    budget = max(0, sibling_sample - len(tail))
    strata = [[r for r in siblings if r["_sim"] > sibling_sim_tail
               and match(r)] for _, match in SIBLING_STRATA]
    per_stratum = budget // len(SIBLING_STRATA) if budget else 0
    leftover = budget - per_stratum * len(SIBLING_STRATA)
    for (name, _), members in zip(SIBLING_STRATA, strata):
        want = per_stratum + (1 if leftover > 0 else 0)
        leftover -= 1
        if not members or want <= 0:
            continue
        chosen = rng.choice(len(members), size=min(want, len(members)),
                            replace=False)
        for i in sorted(chosen.tolist()):
            take(members[i], "suspect_sibling",
                 f"sibling_export sample ({name})")

    by_faces = sorted(rows, key=lambda r: -r["_faces"])
    taken = 0
    for row in by_faces:
        if taken >= top_faces:
            break
        if row["cluster_id"] not in picked:
            take(row, "largest_faces", f"top by faces ({row['_faces']})")
            taken += 1
    by_events = sorted(rows, key=lambda r: (-r["_events"], -r["_faces"]))
    taken = 0
    for row in by_events:
        if taken >= top_events:
            break
        if row["cluster_id"] not in picked:
            take(row, "largest_events", f"top by events ({row['_events']})")
            taken += 1

    rest = [r for r in rows if r["cluster_id"] not in picked]
    if rest and n_random:
        chosen = rng.choice(len(rest), size=min(n_random, len(rest)),
                            replace=False)
        for i in sorted(chosen.tolist()):
            take(rest[i], "random_uniform", "uniform purity sample")

    rest = [r for r in rows if r["cluster_id"] not in picked]
    if rest and n_size_weighted:
        weights = np.array([r["_faces"] for r in rest], dtype=np.float64)
        weights /= weights.sum()
        chosen = rng.choice(len(rest), size=min(n_size_weighted, len(rest)),
                            replace=False, p=weights)
        for i in sorted(chosen.tolist()):
            take(rest[i], "random_size_weighted",
                 f"face-weighted purity sample ({rest[i]['_faces']} faces)")

    section_rank = {"suspect_static": 0, "suspect_degenerate": 1,
                    "suspect_sibling": 2, "largest_faces": 3,
                    "largest_events": 4, "random_uniform": 5,
                    "random_size_weighted": 6}
    ordered = sorted(picked.values(),
                     key=lambda t: (section_rank[t[1]], -t[0]["_faces"],
                                    t[0]["cluster_id"]))
    out = []
    for rank, (row, section, reason) in enumerate(ordered, start=1):
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean.update({"rank": rank, "section": section, "reason": reason})
        out.append(clean)
    return out
