"""Offline tooling.

These exercise the seams that broke silently before: the vendored RVC code imports
itself by its upstream path, and fetch_models used to start a 780 MB download when
asked for --help.
"""

from __future__ import annotations

import pytest

from rtvc.constants import generator_frames
from tools import export_onnx, fetch_models, inspect_voice, quantize
from tools.paths import enable_vendored_rvc


def test_vendored_rvc_resolves_its_upstream_imports():
    """third_party/rvc is kept verbatim, so it imports `infer.module.*`.

    Patching those imports would work until the next fetch_models overwrote them, so the
    names are aliased at import time instead. If this breaks, exporting stops working
    with a ModuleNotFoundError that points at upstream rather than at us.
    """
    pytest.importorskip("torch")
    enable_vendored_rvc()
    from rvc.models import SynthesizerTrnMs768NSFsid

    assert SynthesizerTrnMs768NSFsid is not None


def test_enable_vendored_rvc_is_idempotent():
    pytest.importorskip("torch")
    enable_vendored_rvc()
    enable_vendored_rvc()  # a second call must not re-alias or raise


@pytest.mark.parametrize(
    "module", [fetch_models, inspect_voice, export_onnx, quantize], ids=lambda m: m.__name__
)
def test_help_never_does_work(module, capsys):
    """--help must not download, load a checkpoint, or quantise anything."""
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    # These are the banners each tool prints once it starts doing work. The word
    # "download" itself appears in fetch_models' own description, so it proves nothing.
    for banner in ("Base assets", "== encoder", "== generator", "Generators", "file:"):
        assert banner not in out


def test_quantize_refuses_to_guess_calibration_data(capsys):
    """Neither --audio nor --random means the user has not chosen; do not pick for them.

    Defaulting to random calibration is what produced measurably broken audio before.
    """
    assert quantize.main([]) == 2
    assert "--audio" in capsys.readouterr().out


def test_quantize_rejects_a_missing_audio_file(tmp_path, capsys):
    assert quantize.main(["--audio", str(tmp_path / "nope.wav")]) == 2
    assert "does not exist" in capsys.readouterr().out


def test_export_chunk_sizes_map_to_the_expected_frame_counts():
    """The tool and the runtime must agree, or the export lands under the wrong name."""
    for chunk_ms, frames in [(100.0, 36), (200.0, 46), (300.0, 56)]:
        assert generator_frames(chunk_ms, 20.0) == frames
    assert export_onnx.DEFAULT_CHUNKS == [100.0, 150.0, 200.0, 250.0]


def test_inspect_voice_reports_a_missing_file_without_traceback(tmp_path, capsys):
    assert inspect_voice.main([str(tmp_path / "absent.pth")]) == 2
    assert "does not exist" in capsys.readouterr().out
