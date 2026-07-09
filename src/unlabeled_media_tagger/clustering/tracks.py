"""Within-file track collapse: group one video's faces into person-tracks.

A "track" is one person's appearances inside ONE source file. Collapsing to
tracks before global clustering removes the per-video redundancy (the same
face re-detected every 5 s) that would otherwise let one person's 800 frames
dominate centroid math, and cuts the global clustering input ~3x.

The merge machinery is recluster_faces' (pipeline/recluster.py) exact
semantics on a per-file dense matrix — strongest-first union-find with
min/mean cross-similarity gates — with the same-frame cannot-link keyed on
frame_index (two faces in one frame are different people by construction).
Gates default STRICTER than global clustering (0.80/0.75/0.70 vs
0.74/0.70/0.62): within one file the same person under the same
lighting/session should be highly similar, and an over-merged track poisons
every downstream stage while an under-merged one merely dedups less.

Per-file N is tiny (p99 = 96, max = 854 ok faces), so the dense matrix here
is at most ~6 MB — the N^2 ceiling that killed recluster.py globally does not
apply within a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unlabeled_media_tagger.clustering.union_find import UnionFind

DEFAULT_EDGE_THRESHOLD = 0.80
DEFAULT_MERGE_MIN = 0.70
DEFAULT_MERGE_MEAN = 0.75
REPRESENTATIVES_PER_TRACK = 3


@dataclass
class Track:
    """One person-track within one source file (indices into the input rows)."""

    member_indices: list  # sorted indices into the face_rows given
    medoid_index: int  # member with max mean intra-track similarity
    representative_indices: list = field(default_factory=list)  # top-K quality
    embedding: np.ndarray | None = None  # renormalized mean of member vectors
    mean_intra_sim: float = 1.0
    min_intra_sim: float = 1.0


def _frame_of(row: dict):
    """Normalized frame identity for the same-frame constraint ('' -> None)."""
    value = row.get("frame_index", "")
    if value is None or value == "":
        return None
    return int(value)


def build_tracks_for_file(
    face_rows: list,
    embeddings: np.ndarray,
    *,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    merge_min: float = DEFAULT_MERGE_MIN,
    merge_mean: float = DEFAULT_MERGE_MEAN,
    exclude_indices: set | None = None,
) -> list[Track]:
    """Group one file's faces into tracks.

    ``face_rows`` and ``embeddings`` (unit-norm float32, shape (n, 512)) are
    row-aligned. ``exclude_indices`` are faces a human flagged as
    track-impure (via the label store): they are pinned to singleton tracks
    and never merged, so a bad early merge can be corrected by re-running.

    Returns tracks ordered by their smallest member index (deterministic for
    a deterministic input order).
    """
    n = len(face_rows)
    if n == 0:
        return []
    if embeddings.shape[0] != n:
        raise ValueError(
            f"face_rows ({n}) and embeddings ({embeddings.shape[0]}) misaligned")
    excluded = exclude_indices or set()

    sims = embeddings @ embeddings.T

    union_find = UnionFind(n)
    members = {i: {i} for i in range(n)}
    frames = [_frame_of(row) for row in face_rows]
    frame_sets = {i: ({frames[i]} if frames[i] is not None else set())
                  for i in range(n)}

    if n > 1:
        upper_i, upper_j = np.triu_indices(n, k=1)
        keep = sims[upper_i, upper_j] >= edge_threshold
        edges = sorted(
            zip(sims[upper_i, upper_j][keep].tolist(),
                upper_i[keep].tolist(), upper_j[keep].tolist()),
            key=lambda e: (-e[0], e[1], e[2]),
        )

        for _, left, right in edges:
            if left in excluded or right in excluded:
                continue
            left_root = union_find.find(left)
            right_root = union_find.find(right)
            if left_root == right_root:
                continue
            if frame_sets.get(left_root, set()) & frame_sets.get(right_root, set()):
                continue
            cross = sims[np.ix_(sorted(members[left_root]),
                                sorted(members[right_root]))]
            if float(cross.min()) < merge_min:
                continue
            if float(cross.mean()) < merge_mean:
                continue
            new_root = union_find.union(left_root, right_root)
            old_root = right_root if new_root == left_root else left_root
            members[new_root] = members[left_root] | members[right_root]
            if old_root != new_root:
                members.pop(old_root, None)
            frame_sets[new_root] = (frame_sets.get(left_root, set())
                                    | frame_sets.get(right_root, set()))
            if old_root != new_root:
                frame_sets.pop(old_root, None)

    tracks = []
    for root in sorted(members, key=lambda r: min(members[r])):
        indices = sorted(members[root])
        sub = sims[np.ix_(indices, indices)]
        if len(indices) == 1:
            mean_intra, min_intra = 1.0, 1.0
            medoid = indices[0]
        else:
            off_diag = sub[~np.eye(len(indices), dtype=bool)]
            mean_intra = float(off_diag.mean())
            min_intra = float(off_diag.min())
            medoid = indices[int(np.argmax(sub.mean(axis=1)))]

        def _quality(i):
            try:
                return float(face_rows[i].get("quality_score") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        reps = sorted(indices, key=_quality, reverse=True)
        reps = reps[:REPRESENTATIVES_PER_TRACK]

        mean_vec = embeddings[indices].mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        embedding = (mean_vec / norm if norm > 0 else mean_vec).astype(np.float32)

        tracks.append(Track(
            member_indices=indices,
            medoid_index=medoid,
            representative_indices=reps,
            embedding=embedding,
            mean_intra_sim=round(mean_intra, 6),
            min_intra_sim=round(min_intra, 6),
        ))
    return tracks
