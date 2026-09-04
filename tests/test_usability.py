"""Level meters, the pre-flight check, and gain being applied exactly once."""

from __future__ import annotations

import time

import numpy as np

from rtvc.config import AudioConfig, Config, EngineConfig, RuntimeParams
from rtvc.convert.base import Passthrough
from rtvc.doctor import FAIL, OK, format_report, run_checks, worst_status
from rtvc.engine import Engine, to_dbfs


def pump(engine: Engine, want_chunks: int, level: float, seconds: float = 5.0) -> None:
    block = np.full(engine.block, level, dtype=np.float32)
    out = np.zeros(engine.block, dtype=np.float32)
    deadline = time.monotonic() + seconds
    while engine.stats.chunks < want_chunks and time.monotonic() < deadline:
        for _ in range(10):
            engine.rin.write(block)
            engine.rout.read_into(out)
        time.sleep(0.005)


def make_engine(params: RuntimeParams | None = None) -> Engine:
    return Engine(
        Passthrough(),
        AudioConfig(sample_rate=48000, block=480),
        EngineConfig(chunk_ms=100.0, fade_ms=20.0, prefill_ms=100.0),
        params,
    )


# ------------------------------------------------------------------ meters
def test_dbfs_scale():
    assert to_dbfs(1.0) == 0.0
    assert to_dbfs(0.5) == -6.020599913279624
    assert to_dbfs(0.0) == -90.0  # floored rather than -inf


def test_meters_follow_the_signal():
    engine = make_engine()
    engine.start()
    try:
        pump(engine, want_chunks=5, level=0.5)
    finally:
        engine.stop()
    telemetry = engine.snapshot()
    assert telemetry.input_peak == pytest_approx(0.5)
    assert telemetry.output_peak == pytest_approx(0.5)
    assert -7 < telemetry.input_dbfs < -5


def test_meters_stay_at_the_floor_for_silence():
    engine = make_engine()
    engine.start()
    try:
        pump(engine, want_chunks=5, level=0.0)
    finally:
        engine.stop()
    telemetry = engine.snapshot()
    assert telemetry.input_peak == 0.0
    assert telemetry.input_dbfs == -90.0


def test_a_live_input_with_a_silent_output_is_distinguishable():
    """This is the reading that separates a dead model from a dead microphone."""
    engine = make_engine(RuntimeParams(output_gain=0.0))
    engine.start()
    try:
        pump(engine, want_chunks=5, level=0.5)
    finally:
        engine.stop()
    telemetry = engine.snapshot()
    assert telemetry.input_peak > 0.4
    assert telemetry.output_peak == 0.0


def pytest_approx(value: float, tol: float = 1e-6):
    import pytest

    return pytest.approx(value, abs=tol)


# ------------------------------------------------------------------ gain
def test_output_gain_is_applied_exactly_once():
    """A converter that scaled as well would square the gain: 0.5 became 0.25."""
    engine = make_engine(RuntimeParams(output_gain=0.5))
    engine.start()
    try:
        pump(engine, want_chunks=5, level=1.0)
    finally:
        engine.stop()
    assert engine.snapshot().output_peak == pytest_approx(0.5)


# ------------------------------------------------------------------ doctor
def test_doctor_passes_on_a_working_install():
    checks = run_checks(Config())
    assert checks
    assert worst_status(checks) == 0
    report = format_report(checks)
    assert "onnxruntime" in report


def test_doctor_fails_and_says_how_to_fix_a_missing_model_root(tmp_path):
    cfg = Config()
    cfg.model.root = tmp_path / "absent"
    checks = run_checks(cfg)
    assert worst_status(checks) == 1
    failed = [c for c in checks if c.status == FAIL]
    assert any("model root" in c.title for c in failed)
    assert any("tools.fetch_models" in c.fix for c in failed)


def test_doctor_reports_a_missing_voice(tmp_path):
    onnx = tmp_path / "rvc" / "onnx"
    onnx.mkdir(parents=True)
    (tmp_path / "rvc" / "rmvpe.onnx").touch()
    (onnx / "encoder_contentvec_qdq.onnx").touch()

    cfg = Config()
    cfg.model.root = tmp_path
    checks = run_checks(cfg)
    assert worst_status(checks) == 1
    assert any(c.status == FAIL and "no exported voices" in c.title for c in checks)
    assert any(c.status == OK and "pitch tracker" in c.title for c in checks)


def test_doctor_report_is_readable():
    report = format_report(run_checks(Config()))
    assert report.splitlines()[-1]
    assert "[" in report
