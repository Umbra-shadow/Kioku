# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1
"""Latency and recall instrumentation — measured, never fabricated.

A tiny in-process recorder: timed spans feed per-name percentile summaries
that the inspector's substrate gauge reads (retrieval p50/p95, pack token
size). Bounded ring buffers, so it never grows without limit.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Iterator

_MAX_SAMPLES = 4096


@dataclass(frozen=True, slots=True)
class Summary:
    name: str
    count: int
    p50: float
    p95: float
    p99: float
    last: float
    unit: str = "ms"


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class Metrics:
    """Thread-safe latency/value recorder with percentile summaries."""

    def __init__(self) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def record(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._samples[name].append(value_ms)

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] += by

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def summary(self, name: str, unit: str = "ms") -> Summary:
        with self._lock:
            vals = sorted(self._samples.get(name, ()))
        return Summary(
            name=name,
            count=len(vals),
            p50=round(_percentile(vals, 0.50), 3),
            p95=round(_percentile(vals, 0.95), 3),
            p99=round(_percentile(vals, 0.99), 3),
            last=round(vals[bisect_left(vals, vals[-1])], 3) if vals else 0.0,
            unit=unit,
        )

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            names = list(self._samples.keys())
            counters = dict(self._counters)
        return {
            "latencies": {n: asdict(self.summary(n)) for n in names},
            "counters": counters,
        }


METRICS = Metrics()
