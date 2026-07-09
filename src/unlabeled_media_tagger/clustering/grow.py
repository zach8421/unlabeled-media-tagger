"""Growth pass: propose attaching unassigned nodes to VERIFIED identities.

kNN-to-verified-members voting, not stored centroids: verified identities are
multi-modal (angles/lighting/age), feedback-loop.md forbids stored centroids
that drift, and per-face removals must drop out of the comparison set exactly
— all of which member-level voting gives for free.

Scoring, per candidate node:
  - top ``k`` most-similar verified member faces overall;
  - per identity appearing there: score = mean of its best ``top_m`` hits;
  - single-outlier guard: an identity is attach-eligible only when its hits
    come from >= 2 distinct source files;
  - ATTACH-proposal iff best score >= tau_attach AND (best - runner_up) >=
    margin AND no cannot-link (face x identity, or face x face against any
    member);
  - REVIEW-proposal iff score lands within ``review_band`` below tau_attach,
    or the margin/2-file guard failed at attaching strength;
  - otherwise no proposal.

This module is pure math + sets; it never touches the label store. The
driver (scripts/run_growth.py) emits proposals ONLY — a human (or their bulk
approval click) turns proposals into store events via the review server.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DEFAULT_K = 10
DEFAULT_TOP_M = 3
DEFAULT_MARGIN = 0.05
DEFAULT_REVIEW_BAND = 0.05
MIN_DISTINCT_FILES = 2


@dataclass
class CandidateVerdict:
    candidate_index: int
    identity_index: int  # -1 when no identity scored
    score: float
    runner_up: float
    n_distinct_files: int
    exemplar_member_rows: list  # member-matrix rows behind the score
    decision: str  # 'attach' | 'review' | 'none' | 'blocked'


def score_candidates(
    candidate_embeddings: np.ndarray,
    member_embeddings: np.ndarray,
    member_identity_index: np.ndarray,
    member_file_ids: list,
    *,
    tau_attach: float,
    margin: float = DEFAULT_MARGIN,
    review_band: float = DEFAULT_REVIEW_BAND,
    k: int = DEFAULT_K,
    top_m: int = DEFAULT_TOP_M,
    blocked_identity_indices=None,  # per-candidate list of sets, or None
    chunk: int = 4096,
) -> list:
    """Vote every candidate against the verified-member matrix.

    ``blocked_identity_indices``: optional list (len = n candidates) of sets
    of identity indices this candidate may never attach to (cannot-links,
    resolved by the caller). Blocked identities still participate in the
    ranking — a blocked winner yields decision 'blocked', and a blocked
    runner-up still suppresses the margin — they just can't be attached to.
    """
    n_candidates = candidate_embeddings.shape[0]
    n_members = member_embeddings.shape[0]
    verdicts: list = []
    if n_members == 0:
        return [CandidateVerdict(i, -1, 0.0, 0.0, 0, [], "none")
                for i in range(n_candidates)]

    k_eff = min(k, n_members)
    for start in range(0, n_candidates, chunk):
        block = candidate_embeddings[start:start + chunk]
        sims = block @ member_embeddings.T  # (b, n_members)
        top_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)

        for row in range(block.shape[0]):
            candidate = start + row
            blocked = (blocked_identity_indices[candidate]
                       if blocked_identity_indices is not None else set())
            hits = defaultdict(list)  # identity_index -> [(sim, member_row)]
            for member_row, sim in zip(top_idx[row].tolist(),
                                       top_sims[row].tolist()):
                hits[int(member_identity_index[member_row])].append(
                    (sim, member_row))

            # Score EVERY identity, blocked ones included: a cannot-link on
            # the strongest identity must not quietly promote a look-alike
            # runner-up to auto-attach with an empty margin. Blocked
            # identities stay in the ranking (they win -> 'blocked'; they
            # place second -> they still suppress the margin) and are only
            # barred from being attached to.
            scored = []
            for identity, pairs in hits.items():
                pairs.sort(reverse=True)
                score = float(np.mean([s for s, _ in pairs[:top_m]]))
                files = {member_file_ids[m] for _, m in pairs}
                scored.append((score, identity, len(files),
                               [m for _, m in pairs[:top_m]]))
            scored.sort(reverse=True)

            score, identity, n_files, exemplars = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if identity in blocked:
                verdicts.append(CandidateVerdict(
                    candidate, identity, round(score, 6), round(runner_up, 6),
                    n_files, exemplars, "blocked"))
                continue
            attachable = (score >= tau_attach
                          and (score - runner_up) >= margin
                          and n_files >= MIN_DISTINCT_FILES)
            reviewable = (score >= tau_attach - review_band)
            decision = ("attach" if attachable
                        else "review" if reviewable else "none")
            verdicts.append(CandidateVerdict(
                candidate, identity, round(score, 6), round(runner_up, 6),
                n_files, exemplars, decision))
    return verdicts
