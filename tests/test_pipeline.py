"""Engine and offline geometry, exercised with a converter that needs no model files.

The point of these is the plumbing, not the audio: if the crossfade arithmetic or the
window alignment is wrong, the model cannot save it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rtvc import wavio
from rtvc.config import AudioConfig, EngineConfig, RuntimeParams
from rtvc.convert.base import Converter, Passthrough
from rtvc.engine import Engine
from rtvc.offline import convert_array


class Delayed(Converter):
    """Passthrough with context, so alignment errors show up as an offset."""

    def __init__(self, context_ms: float = 100.0) -> None:
        self.context_ms = context_ms
        self.calls = 0

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        self.calls += 1
        return window[-tail:].copy()


def test_offline_conversion_preserves_a_passthrough_signal():
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    out, timings = convert_array(
        Delayed(100.0), audio, chunk=9600, fade=960, context=4800
    )

    assert out.shape == audio.shape
    assert len(timings) > 0
    # Equal-power crossfade of a signal against itself is the identity, so passthrough
    # must survive the chunking bit for bit apart from float rounding.
    np.testing.assert_allclose(out, audio, atol=1e-5)


def test_offline_output_length_matches_input_for_every_chunk_size():
    audio = np.random.default_rng(0).standard_normal(50000).astype(np.float32) * 0.1
    for chunk in (4800, 7200, 9600, 12000):
        out, _ = convert_array(Passthrough(), audio, chunk=chunk, fade=960, context=0)
        assert out.shape == audio.shape


def test_engine_window_geometry():
    engine = Engine(
        Delayed(500.0),
        audio=AudioConfig(sample_rate=48000, block=480),
        settings=EngineConfig(chunk_ms=200.0, fade_ms=20.0, prefill_ms=100.0),
    )
    assert engine.C == 9600
    assert engine.F == 960
    assert engine.ctx == 24000
    assert engine.window == 24000 + 9600 + 960
    assert engine.chunk_budget_ms == 200.0
    assert engine.prefill_ms == 100.0


def _pump(engine: Engine, want_chunks: int, seconds: float = 5.0) -> None:
    """Feed blocks and drain output until the worker has produced `want_chunks`.

    The worker is a separate thread, so the test has to release the GIL while it waits;
    a tight spin would starve exactly the thread it is waiting for.
    """
    block = np.zeros(engine.block, dtype=np.float32)
    out = np.zeros(engine.block, dtype=np.float32)
    deadline = time.monotonic() + seconds
    while engine.stats.chunks < want_chunks and time.monotonic() < deadline:
        for _ in range(10):
            engine.rin.write(block)
            engine.rout.read_into(out)
        time.sleep(0.005)


def test_engine_emits_chunks_and_reports_latency():
    engine = Engine(
        Delayed(100.0),
        audio=AudioConfig(sample_rate=48000, block=480),
        settings=EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=200.0),
    )
    engine.start()
    try:
        _pump(engine, want_chunks=5)
    finally:
        engine.stop()

    telemetry = engine.snapshot()
    assert telemetry.chunks >= 5
    assert telemetry.prefill_ms == 200.0
    assert telemetry.budget_ms == 100.0


def test_bypass_skips_inference():
    converter = Delayed(50.0)
    engine = Engine(
        converter,
        audio=AudioConfig(sample_rate=48000, block=480),
        settings=EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=100.0),
        params=RuntimeParams(bypass=True),
    )
    engine.start()
    calls_after_warmup = converter.calls  # start() probes six times
    try:
        _pump(engine, want_chunks=3)
    finally:
        engine.stop()

    assert engine.stats.chunks >= 3
    assert converter.calls == calls_after_warmup  # bypassed chunks never reach the model


@pytest.mark.parametrize(
    ("p50", "p95", "expected"),
    [
        (60.0, 100.0, "comfortable"),
        (120.0, 150.0, "workable"),
        # Measured for the fp32 generator: median inside budget, p95 outside, and in
        # practice zero underruns. It must not be reported as a failure.
        (191.4, 210.5, "no slack"),
        # Measured for the fp32 encoder: median past budget, and it underran 92 times.
        (219.9, 249.5, "CANNOT KEEP UP"),
    ],
)
def test_verdict_is_decided_by_the_median_not_p95(p50, p95, expected):
    from rtvc.engine import Telemetry

    assert Telemetry(infer_p50_ms=p50, infer_p95_ms=p95, budget_ms=200.0).verdict == expected


def test_verdict_without_measurements_is_unknown():
    from rtvc.engine import Telemetry

    assert Telemetry().verdict == "unknown"


def test_wav_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(4800) * 0.2).astype(np.float32)
    path = tmp_path / "a.wav"
    wavio.write(path, audio, 48000)
    back, rate = wavio.read(path)
    assert rate == 48000
    assert back.shape == audio.shape
    np.testing.assert_allclose(back, audio, atol=1e-4)  # 16-bit quantisation


def test_wav_write_clips_rather_than_wrapping(tmp_path):
    path = tmp_path / "loud.wav"
    wavio.write(path, np.array([2.0, -2.0, 0.0], dtype=np.float32), 48000)
    back, _ = wavio.read(path)
    assert back[0] > 0.99 and back[1] < -0.99
