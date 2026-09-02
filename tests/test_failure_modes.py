"""What the engine does when something goes wrong.

The silent failures are the ones worth pinning: a worker that dies without a word looks
exactly like a working engine from the outside, right up until someone notices nobody
can hear them.
"""

from __future__ import annotations

import gc
import struct
import time
import wave

import numpy as np
import pytest

from rtvc import wavio
from rtvc.config import AudioConfig, Config, EngineConfig
from rtvc.convert.base import Converter, Passthrough
from rtvc.engine import Engine


class Exploding(Converter):
    """Fails partway through, the way a model file deleted mid-run would."""

    context_ms = 0.0

    def __init__(self, fail_after: int) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("model went away")
        return window[-tail:]


def pump(engine: Engine, want_chunks: int, seconds: float = 5.0) -> None:
    block = np.zeros(engine.block, dtype=np.float32)
    out = np.zeros(engine.block, dtype=np.float32)
    deadline = time.monotonic() + seconds
    while engine.stats.chunks < want_chunks and time.monotonic() < deadline:
        if engine.stats.worker_error:
            return
        for _ in range(10):
            engine.rin.write(block)
            engine.rout.read_into(out)
        time.sleep(0.005)


def test_a_dying_worker_is_reported_rather_than_silent():
    engine = Engine(
        Exploding(fail_after=8),  # six warmup calls, then two real chunks
        AudioConfig(sample_rate=48000, block=480),
        EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=100.0),
    )
    engine.start()
    try:
        pump(engine, want_chunks=20)
    finally:
        engine.stop()

    telemetry = engine.snapshot()
    assert telemetry.failed, "the worker died and telemetry still looked healthy"
    assert "model went away" in telemetry.worker_error
    assert "ENGINE STOPPED" in engine.report(1.0)


def test_a_healthy_engine_reports_no_error():
    engine = Engine(
        Passthrough(),
        AudioConfig(sample_rate=48000, block=480),
        EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=100.0),
    )
    engine.start()
    try:
        pump(engine, want_chunks=5)
    finally:
        engine.stop()
    assert not engine.snapshot().failed
    assert engine.snapshot().worker_error is None


def test_restarting_clears_a_previous_error():
    converter = Exploding(fail_after=7)
    engine = Engine(
        converter,
        AudioConfig(sample_rate=48000, block=480),
        EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=100.0),
    )
    engine.start()
    try:
        pump(engine, want_chunks=20)
    finally:
        engine.stop()
    assert engine.snapshot().failed

    converter.fail_after = 10_000  # the model is back
    engine.start()
    try:
        pump(engine, want_chunks=3)
    finally:
        engine.stop()
    assert not engine.snapshot().failed


def test_gc_stays_paused_until_the_last_engine_stops():
    """Stopping one engine must not hand a running one a collector it was promised was off."""
    was_enabled = gc.isenabled()
    first = Engine(Passthrough(), AudioConfig(), EngineConfig(prefill_ms=50.0))
    second = Engine(Passthrough(), AudioConfig(), EngineConfig(prefill_ms=50.0))
    try:
        first.start()
        second.start()
        assert not gc.isenabled()
        first.stop()
        assert not gc.isenabled(), "gc resumed while the second engine was still running"
    finally:
        second.stop()
    assert gc.isenabled() == was_enabled


def test_stop_without_start_does_not_disturb_gc():
    was_enabled = gc.isenabled()
    Engine(Passthrough(), AudioConfig(), EngineConfig(prefill_ms=50.0)).stop()
    assert gc.isenabled() == was_enabled


def test_reads_24_bit_wav(tmp_path):
    """24-bit is what most recording gear writes, and calibration wants real recordings."""
    path = tmp_path / "24bit.wav"
    values = [0, 1 << 22, -(1 << 22), (1 << 23) - 1, -(1 << 23)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(48000)
        w.writeframes(b"".join(struct.pack("<i", v << 8)[1:] for v in values))

    data, rate = wavio.read(path)
    assert rate == 48000
    assert data.shape == (5,)
    np.testing.assert_allclose(data, [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-6)


def test_preset_with_an_unknown_key_names_it(tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"params": {"key_shift": 1.0, "reverb_amount": 3}}', encoding="utf-8")
    with pytest.raises(ValueError, match="reverb_amount"):
        Config.load(path)
