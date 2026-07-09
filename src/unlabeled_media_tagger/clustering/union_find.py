"""Disjoint-set structure, ported verbatim from pipeline/recluster.py.

Copied rather than imported: recluster.py pulls in pipeline/run.py (Drive,
DeepFace, the whole pipeline import graph), and this package must stay
importable in seconds with numpy only. Keep the two in lockstep — the parity
tests in tests/clustering/ compare cluster assignments against recluster.py.
"""

from __future__ import annotations


class UnionFind:
    """Disjoint-set structure for cluster merges."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root

        keep, drop = sorted((left_root, right_root))
        self.parent[drop] = keep
        return keep
