"""Command line argument handling.

Settings that cannot work should be refused at the command line, not deep inside a
worker thread where the message no longer points at the cause.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from rtvc.cli import build_parser, main, parse_backend, validate


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["simulate", "--chunk", "0"], "greater than 0"),
        (["simulate", "--chunk", "-100"], "greater than 0"),
        (["simulate", "--chunk", "137"], "multiple of 10ms"),
        (["simulate", "--chunk", "100", "--fade", "100"], "shorter than --chunk"),
        (["simulate", "--chunk", "100", "--fade", "-5"], "cannot be negative"),
        (["simulate", "--context", "-1"], "cannot be negative"),
        (["simulate", "--threads", "0"], "at least 1"),
        (["simulate", "--block", "0"], "at least 1"),
        (["simulate", "--rate", "100"], "at least 8000"),
        (["simulate", "--gain", "-1"], "cannot be negative"),
    ],
)
def test_impossible_settings_are_refused(argv, expected):
    problem = validate(parse(argv))
    assert problem is not None and expected in problem


def test_workable_settings_pass():
    assert validate(parse(["simulate", "--chunk", "200", "--fade", "20"])) is None


def test_devices_has_nothing_to_validate():
    assert validate(parse(["devices"])) is None


def test_convert_reports_a_missing_file_without_a_traceback(capsys):
    assert main(["convert", "--in", "no_such_file.wav", "--out", "out.wav"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_convert_reports_an_unreadable_file_without_a_traceback(tmp_path, capsys):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is not a wav file at all")
    assert main(["convert", "--in", str(broken), "--out", str(tmp_path / "o.wav")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_convert_passthrough_round_trips_a_real_file(tmp_path):
    source = tmp_path / "in.wav"
    with wave.open(str(source), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        t = np.arange(48000, dtype=np.float32) / 48000
        w.writeframes((np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes())

    out = tmp_path / "out.wav"
    code = main(
        ["convert", "--in", str(source), "--out", str(out), "--converter", "passthrough",
         "--chunk", "100"]
    )
    assert code == 0
    assert out.exists()

    from rtvc import wavio

    data, rate = wavio.read(out)
    assert rate == 48000
    assert data.shape == (48000,)


def test_backend_spec_parsing():
    assert parse_backend("ort") == {"enc": "ort", "pit": "ort", "gen": "ort"}
    assert parse_backend("enc=ov,pit=ort,gen=ort")["enc"] == "ov"


def test_malformed_backend_spec_is_rejected():
    with pytest.raises(ValueError):
        parse_backend("enc=ov,garbage")
