"""Scalable face clustering over detect-run outputs (FOLDER 2 era).

This package operates on the sharded embedding store written by
scripts/run_embed_batch.py, NOT on the per-file processed JSONs the original
pipeline caches. It exists because pipeline/recluster.py's dense N x N
similarity matrix cannot scale past ~50K faces on this machine, while the
FOLDER 2 good-face set is ~740K. The union-find + merge-gate semantics are
ported from recluster.py (kept as the dense reference implementation for
parity tests); the dense matrix is replaced by per-file matrices (tracks) and
a sparse kNN graph (clusters).
"""
