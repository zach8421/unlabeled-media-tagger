"""Concurrent, size-aware download pool for the detect-only run.

Phase 4 of the FOLDER 2 detect-only run is download-bound: wired single-stream
tops out ~35-41 MB/s, but aggregate throughput saturates ~100 MB/s at ~8
parallel streams (see dev-log 2026-06-12). This module is the concurrency
lever — it runs N download workers in parallel while detection stays on a
single thread (one GPU), so the GPU is kept fed without serializing on the
download pipe.

Two pieces:

- ``ByteBudget`` — a variable-weight counting semaphore measured in BYTES, not
  slots. It is the *size-aware* throttle the dev-log calls for: 8 concurrent
  full downloads of the big masters (single files reach 211 GB) would need
  ~1 TB of scratch, so admission is bounded by total in-flight bytes, not just
  worker count. A file larger than the whole budget is admitted alone (it can
  never fit alongside anything else) rather than deadlocking.

- ``DownloadPool`` — worker threads pull items off an input queue, reserve
  their bytes from the budget, download to scratch, and hand the finished file
  to the main thread one at a time. The main thread runs detection (GPU) and,
  in a guaranteed ``finally``, deletes the item's scratch and releases its
  bytes back to the budget — preserving the runner's invariant that peak local
  disk stays ~(sum of in-flight items), never the folder size.
"""

from __future__ import annotations

import queue
import shutil
import threading
from pathlib import Path
from typing import Callable, Iterable


class ByteBudget:
    """A counting semaphore measured in bytes (variable-weight acquire).

    ``acquire(n)`` blocks until ``n`` bytes fit under the cap, then reserves
    them; ``release(n)`` returns them and wakes waiters. A request larger than
    the cap is clamped to the cap and admitted only when nothing else holds
    budget, so a single oversized file (e.g. a 211 GB master against a 256 GB
    cap) runs alone instead of deadlocking. ``acquire`` returns the amount
    actually reserved — always pass that same value to ``release``.
    """

    def __init__(self, cap_bytes: int):
        if cap_bytes <= 0:
            raise ValueError(f"cap_bytes must be positive, got {cap_bytes}")
        self.cap = int(cap_bytes)
        self._used = 0
        self._cond = threading.Condition()

    def acquire(self, n: int) -> int:
        """Reserve ``min(n, cap)`` bytes, blocking until they fit. Returns the
        amount reserved (pass it back to ``release``)."""
        want = min(max(int(n), 0), self.cap)
        with self._cond:
            # Wait while granting `want` would exceed the cap AND something is
            # already holding budget. Once `_used == 0`, even an at-cap request
            # is granted (the oversized-file-runs-alone case).
            while self._used > 0 and self._used + want > self.cap:
                self._cond.wait()
            self._used += want
        return want

    def release(self, n: int) -> None:
        """Return ``n`` previously-reserved bytes and wake any waiters."""
        with self._cond:
            self._used -= int(n)
            if self._used < 0:
                self._used = 0
            self._cond.notify_all()

    @property
    def used(self) -> int:
        with self._cond:
            return self._used


class DownloadPool:
    """Parallel downloads feeding a single-threaded consumer.

    ``run(items, consume)`` spawns one worker per service. Each worker, for one
    item: reserves its bytes from the budget, calls ``download_fn`` to fetch it
    to scratch, and enqueues the result. The calling thread drains results in
    completion order and invokes ``consume(result)`` on each — that callback is
    where detection runs, so it executes on ONE thread (one GPU) regardless of
    worker count. After ``consume`` (or a download failure), the pool always
    deletes the item's scratch dir and releases its budget.

    ``download_fn(service, item, scratch_root) -> dict`` must download the item
    and return a dict with at least ``local_path``, ``name`` and
    ``download_sec``; it should raise on failure.

    Each result dict passed to ``consume`` has:
      - ``item``         — the original manifest row
      - ``ok``           — True if the download succeeded
      - ``item_scratch`` — Path to this item's scratch dir (always set)
      - on success: the keys returned by ``download_fn``
      - on failure: ``download_error`` (a "Type: message" string)
    """

    def __init__(
        self,
        *,
        download_fn: Callable[[object, dict, Path], dict],
        services: Iterable,
        scratch_root,
        budget: ByteBudget,
        default_reserve_bytes: int = 1 << 30,  # 1 GiB, for rows lacking a size
        ready_buffer: int | None = None,
    ):
        self.download_fn = download_fn
        self.services = list(services)
        if not self.services:
            raise ValueError("DownloadPool needs at least one service")
        self.scratch_root = Path(scratch_root)
        self.budget = budget
        self.default_reserve = int(default_reserve_bytes)
        self.n_workers = len(self.services)
        # Cap downloaded-but-not-yet-consumed results in memory. The byte budget
        # already bounds disk; this just keeps workers from racing far ahead.
        self.ready_buffer = ready_buffer or (self.n_workers + 1)

    def _reserve_for(self, item: dict) -> int:
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return size if size > 0 else self.default_reserve

    def run(self, items: Iterable[dict], consume: Callable[[dict], None]) -> None:
        items = list(items)
        input_q: queue.Queue = queue.Queue()
        for it in items:
            input_q.put(it)
        for _ in range(self.n_workers):
            input_q.put(None)  # one sentinel per worker
        ready_q: queue.Queue = queue.Queue(maxsize=self.ready_buffer)

        def worker(service) -> None:
            while True:
                item = input_q.get()
                try:
                    if item is None:
                        return
                    item_scratch = self.scratch_root / item["file_id"]
                    reserved = self.budget.acquire(self._reserve_for(item))
                    result = {
                        "item": item,
                        "reserved": reserved,
                        "item_scratch": item_scratch,
                        "ok": False,
                    }
                    try:
                        result.update(self.download_fn(service, item, self.scratch_root))
                        result["ok"] = True
                    except Exception as exc:  # noqa: BLE001 - reported, not raised
                        result["download_error"] = f"{type(exc).__name__}: {exc}"
                    ready_q.put(result)
                finally:
                    input_q.task_done()

        threads = [
            threading.Thread(target=worker, args=(svc,), name=f"dl-worker-{i}",
                             daemon=True)
            for i, svc in enumerate(self.services)
        ]
        for t in threads:
            t.start()

        try:
            for _ in range(len(items)):
                result = ready_q.get()
                try:
                    consume(result)
                finally:
                    shutil.rmtree(result["item_scratch"], ignore_errors=True)
                    self.budget.release(result["reserved"])
        finally:
            for t in threads:
                t.join()
