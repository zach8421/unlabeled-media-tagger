"""ExactGpuIndex parity vs the CPU exact path.

Needs the TF GPU stack, so it is opt-in like the embed parity gate:
RUN_GPU_TESTS=1 ./.venv/bin/python -m pytest tests/clustering/test_knn_graph_gpu.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS"),
    reason="GPU parity test; set RUN_GPU_TESTS=1 to run",
)


def _unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_search_matches_exact_topk():
    from unlabeled_media_tagger.clustering.knn_graph import (
        ExactGpuIndex, exact_gpu_available, exact_topk_sample)
    if not exact_gpu_available():
        pytest.skip("no GPU visible to TensorFlow")

    emb = _unit_rows(500, 32, seed=0)
    index = ExactGpuIndex(emb, chunk=128)  # forces multiple chunks
    sims, ids = index.search(emb, 11)
    assert ids.shape == (500, 11)
    assert sims.dtype == np.float32 and ids.dtype == np.int64
    assert (ids[:, 0] == np.arange(500)).all()  # self is the top hit

    exact = exact_topk_sample(emb, np.arange(500), 10)
    for i, (row_ids, row_exact) in enumerate(zip(ids, exact)):
        gpu_set = set(row_ids.tolist()) - {i}
        cpu_set = set(row_exact.tolist())
        # GPU fp32 GEMM may swap neighbors whose sims are within rounding
        # of each other at the rank boundary; any disagreement must be a
        # near-tie, not a genuinely different neighbor.
        for j in gpu_set ^ cpu_set:
            boundary = float(emb[i] @ emb[row_exact[-1]])
            assert abs(float(emb[i] @ emb[j]) - boundary) < 1e-4

    # k > n clamps instead of raising (mirrors ann_recall's contract).
    tiny = _unit_rows(5, 8, seed=1)
    sims, ids = ExactGpuIndex(tiny).search(tiny, 50)
    assert ids.shape == (5, 5)


def test_knn_edges_parity_with_cpu_exact():
    from unlabeled_media_tagger.clustering.knn_graph import (
        ExactGpuIndex, exact_gpu_available, knn_edges)
    if not exact_gpu_available():
        pytest.skip("no GPU visible to TensorFlow")

    emb = _unit_rows(400, 16, seed=2)
    threshold = 0.2
    gpu = knn_edges(emb, k=8, threshold=threshold,
                    index=ExactGpuIndex(emb, chunk=64), exact=False)
    cpu = knn_edges(emb, k=8, threshold=threshold, exact=True)

    gpu_pairs = dict(zip(zip(gpu[0].tolist(), gpu[1].tolist()),
                         gpu[2].tolist()))
    cpu_pairs = dict(zip(zip(cpu[0].tolist(), cpu[1].tolist()),
                         cpu[2].tolist()))
    # Disagreements are only allowed within float rounding of the threshold.
    for a, b in set(gpu_pairs) ^ set(cpu_pairs):
        assert abs(float(emb[a] @ emb[b]) - threshold) < 1e-4
    for pair in set(gpu_pairs) & set(cpu_pairs):
        assert abs(gpu_pairs[pair] - cpu_pairs[pair]) < 1e-5
