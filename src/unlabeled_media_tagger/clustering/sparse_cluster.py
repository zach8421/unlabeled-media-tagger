"""Sparse high-precision clustering engine: recluster.py's semantics at 700K scale.

pipeline/recluster.py's algorithm is right (strongest-first union-find with
merge quality gates + cannot-link constraints) but its dense N x N matrix
caps it at ~50K faces on this machine. This engine keeps the semantics and
replaces the matrix with a kNN edge list. Gate order per candidate edge:

  G0 constraints  — cannot-link pairs as per-root "enemy sets": merging two
                    roots is refused if either contains a node cannot-linked
                    to the other. Same-frame pairs and user removals are the
                    same mechanism with different provenance.
  G1 mean (EXACT) — per-root running vector sums: for unit vectors the mean
                    of ALL |A|x|B| cross cosine similarities equals
                    (S_A . S_B)/(|A||B|), including pairs the kNN graph never
                    observed. O(d) per check, bit-equal to the dense gate.
                    This is the anti-chaining workhorse.
  G2 min          — 'sparse': min over OBSERVED kNN cross-edges (optimistic,
                    kNN edges skew high). 'exact': true min over all cross
                    pairs (O(|A||B|), for parity tests / small runs).
  G3 worst-member — min over members of cos(member, opposite centroid), both
                    directions; compensates G2-sparse's optimism. Skipped
                    above --size-cap (mean gate + constraints still guard).

Rejected root-pairs are memoized keyed on (root, version): a root's version
bumps on every merge, so a grown cluster is re-evaluated, not stale-skipped.

Everything below threshold or in clusters smaller than min_cluster_size is
UNASSIGNED (label -1) — a first-class output, never force-assigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unlabeled_media_tagger.clustering.union_find import UnionFind

DEFAULT_MERGE_MIN = 0.62
DEFAULT_MERGE_MEAN = 0.70
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_SIZE_CAP = 20_000


@dataclass
class SparseClusterResult:
    labels: np.ndarray  # int64 per node; -1 = unassigned
    similarity_to_cluster: np.ndarray  # float32; cos(node, own centroid); 0 if unassigned
    n_clusters: int
    stats: dict = field(default_factory=dict)


class _Adjacency:
    """CSR view of the undirected edge list for observed-min scans."""

    def __init__(self, n_nodes: int, edges_i, edges_j, edges_sim):
        src = np.concatenate([edges_i, edges_j])
        dst = np.concatenate([edges_j, edges_i])
        sim = np.concatenate([edges_sim, edges_sim])
        order = np.argsort(src, kind="stable")
        self.dst = dst[order]
        self.sim = sim[order]
        self.indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        counts = np.bincount(src, minlength=n_nodes)
        np.cumsum(counts, out=self.indptr[1:])

    def neighbors(self, node: int):
        lo, hi = self.indptr[node], self.indptr[node + 1]
        return self.dst[lo:hi], self.sim[lo:hi]


def sparse_cluster(
    embeddings: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    edges_sim: np.ndarray,
    *,
    cannot_link_pairs=(),
    merge_min: float = DEFAULT_MERGE_MIN,
    merge_mean: float = DEFAULT_MERGE_MEAN,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_gate: str = "sparse",  # 'sparse' | 'exact'
    worst_member_gate: bool = True,
    size_cap: int = DEFAULT_SIZE_CAP,
) -> SparseClusterResult:
    """Cluster nodes over a precomputed similarity edge list.

    ``embeddings``: float32 (n, d), rows unit-norm. ``edges_*``: parallel
    arrays of an undirected edge list already filtered to the edge threshold
    (each pair once; ordering need not be sorted — it is re-sorted here for
    determinism). ``cannot_link_pairs``: iterable of (a, b) node ids.
    """
    n = embeddings.shape[0]
    if not (len(edges_i) == len(edges_j) == len(edges_sim)):
        raise ValueError("edge arrays misaligned")
    if min_gate not in ("sparse", "exact"):
        raise ValueError(f"unknown min_gate {min_gate!r}")

    edges_i = np.asarray(edges_i, dtype=np.int64)
    edges_j = np.asarray(edges_j, dtype=np.int64)
    edges_sim = np.asarray(edges_sim, dtype=np.float32)
    order = np.lexsort((edges_j, edges_i, -edges_sim))
    edges_i, edges_j, edges_sim = edges_i[order], edges_j[order], edges_sim[order]

    union_find = UnionFind(n)
    members: dict[int, list] = {i: [i] for i in range(n)}
    sums = embeddings.astype(np.float32).copy()  # per-root running sums
    version = np.zeros(n, dtype=np.int64)
    enemies: dict[int, set] = {}
    for a, b in cannot_link_pairs:
        a, b = int(a), int(b)
        if a == b:
            continue
        enemies.setdefault(a, set()).add(b)
        enemies.setdefault(b, set()).add(a)

    adjacency = _Adjacency(n, edges_i, edges_j, edges_sim) \
        if min_gate == "sparse" else None
    rejected: set = set()
    stats = {"edges": len(edges_i), "merges": 0, "g0_blocked": 0,
             "g1_blocked": 0, "g2_blocked": 0, "g3_blocked": 0,
             "memo_skipped": 0, "size_cap_skipped_g3": 0}

    def observed_min(small_root: int, large_root: int) -> float:
        worst = np.inf
        for node in members[small_root]:
            dst, sim = adjacency.neighbors(node)
            for neighbor, s in zip(dst.tolist(), sim.tolist()):
                if union_find.find(neighbor) == large_root and s < worst:
                    worst = s
        return worst

    for sim, left, right in zip(edges_sim.tolist(), edges_i.tolist(),
                                edges_j.tolist()):
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue
        memo_key = (left_root, version[left_root], right_root,
                    version[right_root]) if left_root < right_root else \
            (right_root, version[right_root], left_root, version[left_root])
        if memo_key in rejected:
            stats["memo_skipped"] += 1
            continue

        size_l, size_r = len(members[left_root]), len(members[right_root])

        # G0: enemy sets (constraints).
        left_enemies = enemies.get(left_root, ())
        if right_root in left_enemies:
            stats["g0_blocked"] += 1
            continue  # permanent for these roots; no need to memoize

        # G1: exact all-pairs cross mean via running sums.
        cross_mean = float(sums[left_root] @ sums[right_root]) / (size_l * size_r)
        if cross_mean < merge_mean:
            stats["g1_blocked"] += 1
            rejected.add(memo_key)
            continue

        # G2: min gate.
        if min_gate == "exact":
            cross = embeddings[members[left_root]] @ \
                embeddings[members[right_root]].T
            if float(cross.min()) < merge_min:
                stats["g2_blocked"] += 1
                rejected.add(memo_key)
                continue
        else:
            small, large = ((left_root, right_root) if size_l <= size_r
                            else (right_root, left_root))
            if observed_min(small, large) < merge_min:
                stats["g2_blocked"] += 1
                rejected.add(memo_key)
                continue

        # G3: worst member vs opposite centroid (both directions).
        if worst_member_gate and min_gate == "sparse":
            if size_l + size_r > size_cap:
                stats["size_cap_skipped_g3"] += 1
            else:
                centroid_r = sums[right_root] / np.linalg.norm(sums[right_root])
                centroid_l = sums[left_root] / np.linalg.norm(sums[left_root])
                worst_l = float(
                    (embeddings[members[left_root]] @ centroid_r).min())
                worst_r = float(
                    (embeddings[members[right_root]] @ centroid_l).min())
                if min(worst_l, worst_r) < merge_min:
                    stats["g3_blocked"] += 1
                    rejected.add(memo_key)
                    continue

        # Merge.
        new_root = union_find.union(left_root, right_root)
        old_root = right_root if new_root == left_root else left_root
        members[new_root] = members[left_root] + members[right_root]
        if old_root != new_root:
            members.pop(old_root)
        sums[new_root] = sums[left_root] + sums[right_root]
        merged_enemies = (enemies.pop(left_root, set())
                          | enemies.pop(right_root, set()))
        # Re-key: enemies recorded against either old root now point at new_root.
        if merged_enemies:
            enemies[new_root] = merged_enemies
            for enemy in merged_enemies:
                enemy_set = enemies.setdefault(enemy, set())
                enemy_set.discard(left_root)
                enemy_set.discard(right_root)
                enemy_set.add(new_root)
        version[new_root] += 1
        stats["merges"] += 1

    # Emit labels. Deterministic cluster ids: roots ordered by smallest member.
    labels = np.full(n, -1, dtype=np.int64)
    similarity = np.zeros(n, dtype=np.float32)
    n_clusters = 0
    for root in sorted(members, key=lambda r: min(members[r])):
        idx = members[root]
        if len(idx) < min_cluster_size:
            continue
        idx_arr = np.asarray(sorted(idx))
        centroid = sums[root] / np.linalg.norm(sums[root])
        labels[idx_arr] = n_clusters
        similarity[idx_arr] = embeddings[idx_arr] @ centroid
        n_clusters += 1

    stats["assigned"] = int((labels >= 0).sum())
    stats["unassigned"] = int((labels < 0).sum())
    return SparseClusterResult(
        labels=labels,
        similarity_to_cluster=similarity,
        n_clusters=n_clusters,
        stats=stats,
    )
