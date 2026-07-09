"""kNN edge-list construction over node embeddings.

Produces the sparse edge list sparse_cluster.py consumes: each undirected
pair once, similarity >= threshold, cosine via inner product on unit vectors.

Backends: exact chunked-GEMM (CPU below EXACT_FALLBACK_N, or GPU via
ExactGpuIndex at any size — recall 1.0 by construction) and faiss-cpu HNSW.
faiss-cpu was chosen over hnswlib because it ships manylinux wheels (hnswlib
is sdist-only -> local compile) and coexists with the pinned TF stack
(verified: faiss 1.14.3 + TF 2.20 + numpy 2.4.6 import together).

Includes the ANN recall check the M4 gate requires: exact chunked-GEMM top-k
on a sample, compared against the HNSW result (target recall@k >= 0.98).
NOTE: measured on FOLDER 2 (353,959 nodes, 512-d), HNSW cannot reach that
gate at any affordable efSearch — the same degenerate hub nodes the
quarantine exists for break graph navigability (efSearch 256 -> recall
0.908; 3072 -> 0.976 at 20 min of query time). ExactGpuIndex does the full
top-51 pass in ~42 s on one TITAN Xp; exact CPU takes ~20 min. Prefer exact.
"""

from __future__ import annotations

import numpy as np

DEFAULT_K = 50
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 256


def build_hnsw_index(embeddings: np.ndarray, *, m: int = DEFAULT_M,
                     ef_construction: int = DEFAULT_EF_CONSTRUCTION,
                     ef_search: int = DEFAULT_EF_SEARCH, threads: int = 0):
    """Build a faiss HNSW inner-product index over unit-norm float32 rows."""
    import faiss

    if threads:
        faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], m,
                                faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    index.hnsw.efSearch = ef_search
    return index


# Below this many nodes, exact chunked-GEMM kNN is cheap (seconds) and HNSW
# recall on small sets is unreliable (measured 0.94 on an 11K pilot subset).
EXACT_FALLBACK_N = 50_000

DEFAULT_GPU_QUERY_CHUNK = 2048


class ExactGpuIndex:
    """Exact top-k via chunked GEMM + top_k on GPU (TensorFlow).

    Duck-types the faiss ``index.search(block, k)`` contract that knn_edges
    and ann_recall rely on: returns (sims, ids) with the self-hit included
    (callers filter it). Exact by construction, so the recall gate is
    satisfied trivially and hub degrees are measured on the true graph.

    chunk bounds the transient sims matrix on device:
    2048 x 354K float32 ~ 2.9 GB — comfortable on a 12 GB card.
    """

    def __init__(self, embeddings: np.ndarray,
                 chunk: int = DEFAULT_GPU_QUERY_CHUNK):
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # TF context already initialized; keep its policy
        self._tf = tf
        self._chunk = chunk
        self._emb = tf.constant(
            np.ascontiguousarray(embeddings, dtype=np.float32))

    def search(self, block: np.ndarray, k: int):
        tf = self._tf
        k = min(k, int(self._emb.shape[0]))
        sims = np.empty((len(block), k), dtype=np.float32)
        ids = np.empty((len(block), k), dtype=np.int64)
        for start in range(0, len(block), self._chunk):
            q = tf.constant(np.ascontiguousarray(
                block[start:start + self._chunk], dtype=np.float32))
            vals, idx = tf.math.top_k(
                tf.matmul(q, self._emb, transpose_b=True), k=k)
            sims[start:start + int(q.shape[0])] = vals.numpy()
            ids[start:start + int(q.shape[0])] = idx.numpy()
        return sims, ids


def exact_gpu_available() -> bool:
    """True when TensorFlow is importable and sees at least one GPU."""
    try:
        import tensorflow as tf
        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def knn_edges(embeddings: np.ndarray, *, k: int = DEFAULT_K,
              threshold: float, index=None, query_batch: int = 65_536,
              exact: bool | None = None, **index_kwargs):
    """Return (edges_i, edges_j, edges_sim): undirected, deduped, >= threshold.

    Self-hits are dropped; each unordered pair appears once (keeping the max
    observed similarity when i->j and j->i disagree, which HNSW allows).
    ``exact=None`` auto-selects exact search below EXACT_FALLBACK_N nodes.
    """
    n = embeddings.shape[0]
    if exact is None:
        exact = index is None and n <= EXACT_FALLBACK_N
    if exact:
        return _exact_edges(embeddings, k=k, threshold=threshold)
    if index is None:
        index = build_hnsw_index(embeddings, **index_kwargs)

    pair_i, pair_j, pair_s = [], [], []
    for start in range(0, n, query_batch):
        block = np.ascontiguousarray(embeddings[start:start + query_batch],
                                     dtype=np.float32)
        sims, neighbors = index.search(block, k + 1)  # +1 absorbs the self-hit
        rows = (np.arange(block.shape[0]) + start)[:, None]
        rows = np.broadcast_to(rows, neighbors.shape)
        valid = (neighbors >= 0) & (neighbors != rows) & (sims >= threshold)
        src, dst, sim = rows[valid], neighbors[valid], sims[valid]
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        pair_i.append(lo)
        pair_j.append(hi)
        pair_s.append(sim)

    if not pair_i:
        empty = np.empty(0)
        return (empty.astype(np.int64), empty.astype(np.int64),
                empty.astype(np.float32))

    lo = np.concatenate(pair_i).astype(np.int64)
    hi = np.concatenate(pair_j).astype(np.int64)
    sim = np.concatenate(pair_s).astype(np.float32)

    # Dedupe unordered pairs keeping the max similarity.
    keys = lo * n + hi
    order = np.lexsort((-sim, keys))
    keys, lo, hi, sim = keys[order], lo[order], hi[order], sim[order]
    first = np.ones(len(keys), dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    return lo[first], hi[first], sim[first]


def _exact_edges(embeddings: np.ndarray, *, k: int, threshold: float):
    """Exact kNN edge list for small node sets (recall 1.0 by construction)."""
    n = embeddings.shape[0]
    if n < 2:
        empty = np.empty(0)
        return (empty.astype(np.int64), empty.astype(np.int64),
                empty.astype(np.float32))
    sample_ids = np.arange(n)
    top = exact_topk_sample(embeddings, sample_ids, k)
    src = np.repeat(sample_ids, top.shape[1])
    dst = top.reshape(-1)
    sim = np.einsum("ij,ij->i", embeddings[src], embeddings[dst])
    keep = sim >= threshold
    src, dst, sim = src[keep], dst[keep], sim[keep]
    lo = np.minimum(src, dst).astype(np.int64)
    hi = np.maximum(src, dst).astype(np.int64)
    keys = lo * n + hi
    order = np.lexsort((-sim, keys))
    keys, lo, hi, sim = keys[order], lo[order], hi[order], sim[order]
    first = np.ones(len(keys), dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    return lo[first], hi[first], sim[first].astype(np.float32)


def exact_topk_sample(embeddings: np.ndarray, sample_ids: np.ndarray,
                      k: int, chunk: int = 512) -> np.ndarray:
    """Exact top-k neighbor ids (self excluded) for the sampled rows.

    k is clamped to n-1 (argpartition needs kth < n). chunk=512 keeps the
    transient (chunk x N sims + the int64 argpartition result) near ~1.5 GB
    at N=260K — the earlier 2048 default peaked ~7.5 GB on this box.
    """
    n = embeddings.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.empty((len(sample_ids), 0), dtype=np.int64)
    result = np.empty((len(sample_ids), k), dtype=np.int64)
    for pos in range(0, len(sample_ids), chunk):
        ids = sample_ids[pos:pos + chunk]
        sims = embeddings[ids] @ embeddings.T
        sims[np.arange(len(ids)), ids] = -np.inf  # drop self
        sims *= -1  # in place: fresh per-chunk array, avoids a full copy
        top = np.argpartition(sims, k - 1, axis=1)[:, :k]
        row_sims = np.take_along_axis(sims, top, axis=1)
        result[pos:pos + len(ids)] = np.take_along_axis(
            top, np.argsort(row_sims, axis=1), axis=1)
    return result


def ann_recall(embeddings: np.ndarray, index, *, k: int = DEFAULT_K,
               sample: int = 5_000, seed: int = 0) -> float:
    """recall@k of the HNSW index vs exact search on a random sample.

    Small active sets (pilot runs can have <= k nodes) clamp k to n-1;
    degenerate sets (< 2 nodes) trivially recall 1.0.
    """
    n = embeddings.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(n, size=min(sample, n), replace=False)
    exact = exact_topk_sample(embeddings, sample_ids, k)

    block = np.ascontiguousarray(embeddings[sample_ids], dtype=np.float32)
    _, approx = index.search(block, k + 1)

    hits = 0
    for row_exact, row_approx, self_id in zip(exact, approx, sample_ids):
        approx_set = set(row_approx.tolist()) - {int(self_id), -1}
        hits += len(set(row_exact.tolist()) & approx_set)
    return hits / (len(sample_ids) * k)
