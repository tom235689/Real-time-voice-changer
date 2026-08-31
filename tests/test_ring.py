"""Ring buffer behaviour, including the failure modes it exists to detect."""

from __future__ import annotations

import numpy as np
import pytest

from rtvc.ring import InputRing, OutputRing, Overrun


def test_input_ring_reads_back_what_was_written():
    ring = InputRing(64)
    data = np.arange(20, dtype=np.float32)
    ring.write(data)
    out = np.zeros(20, dtype=np.float32)
    ring.read_window(20, 20, out)
    np.testing.assert_array_equal(out, data)


def test_input_ring_window_survives_wraparound():
    ring = InputRing(64)
    for i in range(10):
        ring.write(np.full(10, i, dtype=np.float32))
    out = np.zeros(30, dtype=np.float32)
    ring.read_window(100, 30, out)
    expected = np.concatenate([np.full(10, i, dtype=np.float32) for i in (7, 8, 9)])
    np.testing.assert_array_equal(out, expected)


def test_input_ring_overlapping_windows_do_not_consume():
    ring = InputRing(128)
    ring.write(np.arange(50, dtype=np.float32))
    a = np.zeros(30, dtype=np.float32)
    b = np.zeros(30, dtype=np.float32)
    ring.read_window(40, 30, a)
    ring.read_window(50, 30, b)
    np.testing.assert_array_equal(a, np.arange(10, 40, dtype=np.float32))
    np.testing.assert_array_equal(b, np.arange(20, 50, dtype=np.float32))


def test_input_ring_reports_overrun_instead_of_returning_garbage():
    ring = InputRing(32)
    block = np.zeros(8, dtype=np.float32)
    for _ in range(12):  # 96 frames through a 32-frame ring
        ring.write(block)
    out = np.zeros(10, dtype=np.float32)
    with pytest.raises(Overrun):
        ring.read_window(20, 10, out)  # long since overwritten


def test_input_ring_rejects_a_window_before_the_stream_start():
    ring = InputRing(32)
    out = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        ring.read_window(5, 10, out)


def test_output_ring_round_trip():
    ring = OutputRing(64)
    assert ring.write(np.arange(10, dtype=np.float32))
    out = np.zeros(10, dtype=np.float32)
    assert ring.read_into(out)
    np.testing.assert_array_equal(out, np.arange(10, dtype=np.float32))
    assert ring.available == 0


def test_output_ring_underrun_yields_silence_not_stale_audio():
    ring = OutputRing(64)
    ring.write(np.full(4, 0.5, dtype=np.float32))
    out = np.full(10, 9.0, dtype=np.float32)
    assert ring.read_into(out) is False
    np.testing.assert_array_equal(out, np.zeros(10, dtype=np.float32))


def test_output_ring_refuses_to_overflow():
    ring = OutputRing(16)
    assert ring.write(np.zeros(16, dtype=np.float32))
    assert ring.write(np.zeros(1, dtype=np.float32)) is False


def test_prefill_adds_exactly_the_requested_latency():
    ring = OutputRing(1000)
    ring.prefill(240)
    assert ring.available == 240


def test_prefill_stops_at_capacity():
    ring = OutputRing(64)
    ring.prefill(1000)
    assert ring.available == 64
