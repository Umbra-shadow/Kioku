"""Metrics recorder — percentiles, counters, timer span."""

from __future__ import annotations

from engine.metrics import Metrics, _percentile


def test_percentile_interpolates() -> None:
    vals = [float(i) for i in range(1, 101)]  # 1..100
    assert _percentile(vals, 0.50) == 50.5
    assert _percentile(vals, 0.95) == 95.05
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.99) == 7.0


def test_record_and_summary() -> None:
    m = Metrics()
    for v in (10.0, 20.0, 30.0, 40.0, 50.0):
        m.record("retrieve_ms", v)
    s = m.summary("retrieve_ms")
    assert s.count == 5
    assert s.p50 == 30.0
    assert s.p95 == 48.0
    assert s.unit == "ms"


def test_counters() -> None:
    m = Metrics()
    m.incr("recalls")
    m.incr("recalls", 4)
    assert m.counter("recalls") == 5
    assert m.counter("never") == 0


def test_timer_records_a_span() -> None:
    m = Metrics()
    with m.timer("commit_ms"):
        sum(range(1000))
    s = m.summary("commit_ms")
    assert s.count == 1
    assert s.last >= 0.0


def test_ring_buffer_is_bounded() -> None:
    m = Metrics()
    for i in range(5000):
        m.record("x", float(i))
    assert m.summary("x").count == 4096


def test_snapshot_shape() -> None:
    m = Metrics()
    m.record("retrieve_ms", 5.0)
    m.incr("recalls")
    snap = m.snapshot()
    assert "retrieve_ms" in snap["latencies"]
    assert snap["counters"]["recalls"] == 1
