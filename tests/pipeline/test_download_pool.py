"""Tests for the concurrent, size-aware download pool.

No network: DownloadPool is driven with a fake download_fn that just creates a
scratch file and records concurrency, so we can assert the byte-budget bounds
peak in-flight downloads and that scratch + budget are always released.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from unlabeled_media_tagger.pipeline.download_pool import ByteBudget, DownloadPool


# --------------------------------------------------------------------------- #
# ByteBudget
# --------------------------------------------------------------------------- #

def test_acquire_release_accounting():
    b = ByteBudget(100)
    assert b.acquire(30) == 30
    assert b.acquire(40) == 40
    assert b.used == 70
    b.release(30)
    assert b.used == 40


def test_acquire_clamps_to_cap():
    """A request larger than the whole budget is clamped, not rejected."""
    b = ByteBudget(100)
    assert b.acquire(250) == 100  # clamped to cap
    assert b.used == 100


def test_zero_cap_rejected():
    with pytest.raises(ValueError):
        ByteBudget(0)


def test_oversized_request_runs_alone_then_proceeds():
    """An at/over-cap request waits until the budget is empty, then is admitted."""
    b = ByteBudget(100)
    b.acquire(60)  # someone is holding budget

    started = threading.Event()
    done = threading.Event()

    def grab_giant():
        started.set()
        b.acquire(200)  # clamps to 100; can only fit once `used` hits 0
        done.set()

    t = threading.Thread(target=grab_giant)
    t.start()
    assert started.wait(1.0)
    # The giant cannot be admitted while 60 is still held.
    assert not done.wait(0.2)
    b.release(60)
    assert done.wait(1.0)
    t.join()
    assert b.used == 100


def test_acquire_blocks_until_release():
    b = ByteBudget(100)
    b.acquire(80)
    proceeded = threading.Event()

    def grab():
        b.acquire(40)  # 80 + 40 > 100 -> must wait
        proceeded.set()

    t = threading.Thread(target=grab)
    t.start()
    assert not proceeded.wait(0.2)
    b.release(80)
    assert proceeded.wait(1.0)
    t.join()


# --------------------------------------------------------------------------- #
# DownloadPool — fake downloader
# --------------------------------------------------------------------------- #

class FakeDownloader:
    """Records peak concurrent downloads; creates a scratch file per item."""

    def __init__(self, fail_ids=None, dl_sleep=0.05):
        self.fail_ids = set(fail_ids or [])
        self.dl_sleep = dl_sleep
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __call__(self, service, item, scratch_root):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            file_id = item["file_id"]
            item_scratch = Path(scratch_root) / file_id
            local_path = item_scratch / "media.bin"
            item_scratch.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"x")  # something for rmtree to remove
            time.sleep(self.dl_sleep)
            if file_id in self.fail_ids:
                raise RuntimeError("simulated download failure")
            return {
                "local_path": str(local_path), "name": "media.bin",
                "item_scratch": item_scratch, "download_sec": 0.0,
            }
        finally:
            with self._lock:
                self.active -= 1


def _items(sizes):
    return [{"file_id": f"f{i}", "path": f"/p/f{i}.mp4", "mimeType": "video/mp4",
             "size": str(s)} for i, s in enumerate(sizes)]


def _run(pool, items, consume_sleep=0.05):
    consumed = []

    def consume(result):
        consumed.append(result)
        time.sleep(consume_sleep)  # simulate detection holding the item's bytes

    pool.run(items, consume)
    return consumed


def test_all_items_consumed_and_scratch_cleaned(tmp_path):
    scratch = tmp_path / "scratch"
    dl = FakeDownloader()
    budget = ByteBudget(1 << 30)
    pool = DownloadPool(download_fn=dl, services=[1, 2, 3],
                        scratch_root=scratch, budget=budget)
    items = _items([10, 10, 10, 10, 10])
    consumed = _run(pool, items)

    assert len(consumed) == len(items)
    assert {r["item"]["file_id"] for r in consumed} == {it["file_id"] for it in items}
    assert all(r["ok"] for r in consumed)
    # Budget fully returned and every scratch dir removed.
    assert budget.used == 0
    assert not any(scratch.iterdir()) if scratch.exists() else True


def test_budget_bounds_peak_concurrency(tmp_path):
    """cap=100, sizes=40 each, 3 workers: 40+40<=100 but 3x40>100 -> peak 2."""
    dl = FakeDownloader(dl_sleep=0.1)
    budget = ByteBudget(100)
    pool = DownloadPool(download_fn=dl, services=[1, 2, 3],
                        scratch_root=tmp_path / "s", budget=budget)
    _run(pool, _items([40, 40, 40, 40, 40]), consume_sleep=0.1)
    assert dl.peak == 2
    assert budget.used == 0


def test_budget_serializes_when_only_one_fits(tmp_path):
    """cap=100, sizes=60 each: two never fit together -> downloads serialize."""
    dl = FakeDownloader(dl_sleep=0.1)
    budget = ByteBudget(100)
    pool = DownloadPool(download_fn=dl, services=[1, 2, 3],
                        scratch_root=tmp_path / "s", budget=budget)
    _run(pool, _items([60, 60, 60]), consume_sleep=0.1)
    assert dl.peak == 1
    assert budget.used == 0


def test_oversized_item_processed_alone(tmp_path):
    """A single file bigger than the whole budget still gets processed."""
    dl = FakeDownloader()
    budget = ByteBudget(100)
    pool = DownloadPool(download_fn=dl, services=[1, 2],
                        scratch_root=tmp_path / "s", budget=budget)
    consumed = _run(pool, _items([500]))
    assert len(consumed) == 1
    assert consumed[0]["ok"]
    assert budget.used == 0


def test_download_failure_reported_and_cleaned(tmp_path):
    scratch = tmp_path / "scratch"
    dl = FakeDownloader(fail_ids={"f1"})
    budget = ByteBudget(1 << 30)
    pool = DownloadPool(download_fn=dl, services=[1, 2],
                        scratch_root=scratch, budget=budget)
    consumed = _run(pool, _items([10, 10, 10]))

    by_id = {r["item"]["file_id"]: r for r in consumed}
    assert len(consumed) == 3
    assert by_id["f1"]["ok"] is False
    assert "RuntimeError" in by_id["f1"]["download_error"]
    assert by_id["f0"]["ok"] and by_id["f2"]["ok"]
    # Even the failed item's scratch is removed and its budget returned.
    assert budget.used == 0
    assert not (scratch / "f1").exists()


def test_rows_without_size_use_default_reserve(tmp_path):
    """Missing/blank size falls back to default_reserve_bytes (no crash)."""
    dl = FakeDownloader()
    budget = ByteBudget(10 << 30)
    pool = DownloadPool(download_fn=dl, services=[1, 2],
                        scratch_root=tmp_path / "s", budget=budget,
                        default_reserve_bytes=1 << 30)
    items = [{"file_id": "a", "path": "/p/a.mp4", "mimeType": "video/mp4"},
             {"file_id": "b", "path": "/p/b.mp4", "mimeType": "video/mp4", "size": ""}]
    consumed = _run(pool, items)
    assert len(consumed) == 2
    assert budget.used == 0


def test_empty_services_rejected(tmp_path):
    with pytest.raises(ValueError):
        DownloadPool(download_fn=lambda *a: {}, services=[],
                     scratch_root=tmp_path, budget=ByteBudget(100))
