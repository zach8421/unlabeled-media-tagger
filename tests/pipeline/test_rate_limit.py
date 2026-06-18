"""Tests for the time-of-day bandwidth throttle.

The token bucket is driven with a fake clock (time + sleep both advance the same
virtual time) so rate math is asserted deterministically with no real sleeping.
"""

from __future__ import annotations

import threading

from unlabeled_media_tagger.pipeline.rate_limit import (
    BandwidthLimiter,
    tod_rate_provider,
)


class FakeClock:
    """Virtual clock: sleep() advances the same time time() reads."""

    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _limiter(rate, clock):
    return BandwidthLimiter(lambda: rate, time_fn=clock.time, sleep_fn=clock.sleep)


# --------------------------------------------------------------------------- #
# tod_rate_provider
# --------------------------------------------------------------------------- #

def test_full_window_is_unlimited():
    p = tod_rate_provider(60_000_000, full_window=(2, 8),
                          now_fn=lambda: _at(5))
    assert p() is None


def test_outside_window_is_capped():
    p = tod_rate_provider(60_000_000, full_window=(2, 8),
                          now_fn=lambda: _at(13))
    assert p() == 60_000_000


def test_boundaries_inclusive_start_exclusive_end():
    cap = 60_000_000
    assert tod_rate_provider(cap, (2, 8), now_fn=lambda: _at(2))() is None   # 02:00 full
    assert tod_rate_provider(cap, (2, 8), now_fn=lambda: _at(8))() == cap    # 08:00 capped
    assert tod_rate_provider(cap, (2, 8), now_fn=lambda: _at(1))() == cap    # 01:59 capped


def test_window_wrapping_midnight():
    cap = 5
    p = lambda h: tod_rate_provider(cap, (22, 6), now_fn=lambda: _at(h))()
    assert p(23) is None   # inside 22–06
    assert p(3) is None
    assert p(6) == cap     # 06:00 capped
    assert p(12) == cap


class _FakeNow:
    def __init__(self, hour):
        self.hour = hour


def _at(hour):
    return _FakeNow(hour)


# --------------------------------------------------------------------------- #
# BandwidthLimiter
# --------------------------------------------------------------------------- #

def test_unlimited_acquire_is_instant():
    clock = FakeClock()
    lim = _limiter(None, clock)
    lim.acquire(10 ** 12)  # 1 TB
    assert clock.t == 0.0  # no time advanced


def test_acquire_paces_to_rate():
    clock = FakeClock()
    lim = _limiter(1000, clock)  # 1000 B/s
    total = 0
    for _ in range(10):
        lim.acquire(500)
        total += 500
    # 5000 bytes at 1000 B/s ~= 5s of virtual time (allow tiny burst slack).
    assert 4.0 <= clock.t <= 5.5
    assert total == 5000


def test_oversized_request_does_not_deadlock():
    clock = FakeClock()
    lim = _limiter(1000, clock)        # cap/capacity = 1000 B
    lim.acquire(5000)                  # 5x capacity in one shot
    assert clock.t > 0                 # had to wait
    assert clock.t <= 6.0              # ~5s for 5000 B at 1000 B/s


def test_rate_change_takes_effect():
    """Flipping the provider from capped to unlimited unblocks immediately."""
    clock = FakeClock()
    rate = {"v": 1000}
    lim = BandwidthLimiter(lambda: rate["v"], time_fn=clock.time,
                           sleep_fn=clock.sleep)
    lim.acquire(1000)        # drains the bucket
    rate["v"] = None         # now unlimited
    t_before = clock.t
    lim.acquire(10 ** 9)     # would block forever if still capped
    assert clock.t == t_before


def test_thread_safe_under_contention():
    """8 threads hammering one limiter complete without corruption/deadlock.

    Uses the REAL clock at a high rate so it finishes in milliseconds; the point
    is to exercise the shared Lock under contention, not to assert timing.
    """
    lim = BandwidthLimiter(lambda: 50_000_000)  # 50 MB/s, real monotonic clock
    consumed = [0]
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            lim.acquire(1024)  # 1 KiB; 8*20*1KiB = 160 KiB total -> ~3ms
            with lock:
                consumed[0] += 1024

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)  # no deadlock
    assert consumed[0] == 8 * 20 * 1024
