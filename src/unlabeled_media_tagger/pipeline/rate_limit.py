"""Time-of-day bandwidth throttle for the detect-only download pool.

A single ``BandwidthLimiter`` is shared by all download workers; each worker
calls ``acquire(nbytes)`` before reading a chunk, so the SUM of bytes/sec across
workers is capped at the currently-scheduled rate. The rate is *dynamic*: a
provider callable returns the cap in bytes/sec (or ``None`` = unlimited) from the
wall clock, so the cap flips at the schedule boundaries mid-run without a
restart.

Units note (this bit it us once): network specs are in **bits**, this limiter is
in **bytes**. 650 Mbps = 81.25 MB/s; 60 MB/s = 480 Mbps. Configure the cap in
decimal MB/s and multiply by 1_000_000 to get bytes/sec.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime


def tod_rate_provider(capped_bytes_per_sec, full_window=(2, 8), now_fn=None):
    """Build a provider -> current cap in bytes/sec, or ``None`` (unlimited).

    Unlimited during the ``[start, end)`` local-clock hour window (full speed);
    capped at ``capped_bytes_per_sec`` otherwise. A window whose start > end is
    treated as wrapping midnight (e.g. ``(22, 6)`` = 22:00–06:00 full).

    Hour granularity is intentional: the boundaries land exactly on the top of
    the hour (e.g. 02:00 / 08:00).
    """
    start, end = full_window
    now_fn = now_fn or datetime.now

    def provider():
        h = now_fn().hour
        full = (start <= h < end) if start <= end else (h >= start or h < end)
        return None if full else capped_bytes_per_sec

    return provider


class BandwidthLimiter:
    """Thread-safe token bucket shared across download workers.

    ``acquire(n)`` blocks until ``n`` bytes of budget are available, then spends
    them; the long-run rate of all callers combined is bounded by the provider's
    current bytes/sec. When the provider returns ``None`` (unlimited) ``acquire``
    returns immediately. Capacity is one second of the current rate (a small
    burst); a request larger than capacity is allowed once the bucket is full
    (tokens go negative and are repaid by refill), so it never deadlocks.

    ``time_fn`` / ``sleep_fn`` are injectable so the bucket is testable with a
    fake clock (no real sleeping).
    """

    def __init__(self, rate_provider, *, time_fn=_time.monotonic,
                 sleep_fn=_time.sleep, max_wait=0.5):
        self._rate = rate_provider
        self._time = time_fn
        self._sleep = sleep_fn
        self._max_wait = max_wait
        self._lock = threading.Lock()
        self._tokens = 0.0
        self._last = time_fn()

    def current_rate(self):
        """Current cap in bytes/sec, or ``None`` if unlimited right now."""
        return self._rate()

    def acquire(self, n):
        """Block until ``n`` bytes of budget are available, then spend them."""
        n = float(n)
        while True:
            with self._lock:
                rate = self._rate()
                now = self._time()
                if not rate or rate <= 0:  # unlimited window
                    # Reset so a later cap doesn't grant a stale burst.
                    self._tokens = 0.0
                    self._last = now
                    return
                cap = rate  # one-second burst capacity
                self._tokens = min(cap, self._tokens + (now - self._last) * rate)
                self._last = now
                if self._tokens >= n or self._tokens >= cap:
                    # Enough tokens, OR bucket full for an oversized request
                    # (n > cap): proceed; tokens may go negative, repaid by refill.
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / rate
            # Sleep outside the lock; cap the nap so the loop re-checks the
            # schedule promptly when the cap flips on/off.
            self._sleep(min(wait, self._max_wait))
