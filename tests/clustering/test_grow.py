"""Behavior pins for the growth pass scoring rules (clustering/grow.py)."""

from __future__ import annotations

import numpy as np

from unlabeled_media_tagger.clustering.grow import score_candidates


def _from_angles(*degrees):
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return np.stack([np.cos(radians), np.sin(radians)]).T.astype(np.float32)


def _members():
    """Identity 0 around 0deg (files f1,f2,f3); identity 1 around 90deg
    (files g1,g2,g3)."""
    member_emb = _from_angles(0, 4, 8, 90, 94, 98)
    identity_idx = np.array([0, 0, 0, 1, 1, 1])
    files = ["f1", "f2", "f3", "g1", "g2", "g3"]
    return member_emb, identity_idx, files


def test_clear_candidate_attaches():
    member_emb, identity_idx, files = _members()
    candidate = _from_angles(3)  # squarely inside identity 0
    [verdict] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.95, margin=0.05)
    assert verdict.decision == "attach"
    assert verdict.identity_index == 0
    assert verdict.n_distinct_files >= 2
    assert verdict.score > 0.99


def test_margin_failure_goes_to_review():
    member_emb, identity_idx, files = _members()
    # Members center on 4deg and 94deg, so 49deg is the equidistant point.
    candidate = _from_angles(49)
    [verdict] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.70, margin=0.05)
    assert verdict.decision == "review"  # score ~.71 for both, margin ~0


def test_below_band_is_none():
    member_emb, identity_idx, files = _members()
    candidate = _from_angles(180)  # opposite everything
    [verdict] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.90, margin=0.05)
    assert verdict.decision == "none"


def test_single_file_guard_blocks_attach():
    # Identity 0's members all come from ONE file: never auto-attach.
    member_emb = _from_angles(0, 4, 8)
    identity_idx = np.array([0, 0, 0])
    files = ["same_file", "same_file", "same_file"]
    candidate = _from_angles(3)
    [verdict] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.95, margin=0.05)
    assert verdict.decision == "review"  # strong score, but 2-file rule fails
    assert verdict.n_distinct_files == 1


def test_cannot_link_blocks_identity():
    member_emb, identity_idx, files = _members()
    candidate = _from_angles(3)
    [verdict] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.95, margin=0.05,
                                 blocked_identity_indices=[{0}])
    # The blocked identity WINS the ranking -> decision is 'blocked'
    # (reported, never attached) — not a silent fall-through to identity 1.
    assert verdict.decision == "blocked"
    assert verdict.identity_index == 0


def test_blocked_winner_never_promotes_lookalike_runner_up():
    """Regression: blocking the top identity must not turn an ambiguous
    candidate into an auto-attach on the runner-up with an empty margin."""
    # Two look-alike identities 8deg apart; candidate between them.
    member_emb = _from_angles(0, 4, 8, 8, 12, 16)
    identity_idx = np.array([0, 0, 0, 1, 1, 1])
    files = ["f1", "f2", "f3", "g1", "g2", "g3"]
    candidate = _from_angles(7)

    [unblocked] = score_candidates(candidate, member_emb, identity_idx, files,
                                   tau_attach=0.90, margin=0.05)
    assert unblocked.decision == "review"  # ambiguous: margin tiny

    [blocked] = score_candidates(candidate, member_emb, identity_idx, files,
                                 tau_attach=0.90, margin=0.05,
                                 blocked_identity_indices=[{unblocked.identity_index}])
    # Winner is blocked -> 'blocked'; must NOT become 'attach' to the other.
    assert blocked.decision == "blocked"


def test_no_members_yields_none():
    verdicts = score_candidates(
        _from_angles(0, 10), np.empty((0, 2), np.float32),
        np.empty(0, dtype=np.int64), [], tau_attach=0.9)
    assert [v.decision for v in verdicts] == ["none", "none"]


def test_determinism_and_exemplars_point_at_member_rows():
    member_emb, identity_idx, files = _members()
    candidates = _from_angles(2, 6, 91)
    first = score_candidates(candidates, member_emb, identity_idx, files,
                             tau_attach=0.95, margin=0.05)
    second = score_candidates(candidates, member_emb, identity_idx, files,
                              tau_attach=0.95, margin=0.05)
    assert [(v.identity_index, v.score) for v in first] == \
        [(v.identity_index, v.score) for v in second]
    for verdict in first:
        for member_row in verdict.exemplar_member_rows:
            assert identity_idx[member_row] == verdict.identity_index
