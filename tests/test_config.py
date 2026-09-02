"""Config geometry and model-file discovery.

Chunk size, generator frame count and file name have to agree exactly: a mismatch loads
a generator built for a different length and produces plausible-sounding garbage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rtvc.catalog import chunk_ms_for_frames, exported_voices
from rtvc.config import Config
from rtvc.constants import generator_frames


@pytest.mark.parametrize(
    ("chunk_ms", "frames"),
    [(100.0, 36), (150.0, 41), (200.0, 46), (250.0, 51)],
)
def test_generator_frames_match_the_exported_files(chunk_ms, frames):
    assert generator_frames(chunk_ms, 20.0) == frames


@pytest.mark.parametrize("chunk_ms", [100.0, 150.0, 200.0, 250.0])
def test_chunk_and_frame_count_round_trip(chunk_ms):
    assert chunk_ms_for_frames(generator_frames(chunk_ms, 20.0), 20.0) == pytest.approx(chunk_ms)


def test_generator_path_follows_chunk_size():
    cfg = Config()
    cfg.model.root = Path("models")
    cfg.engine.chunk_ms = 200.0
    assert cfg.generator_path().name == "generator_my_voice_f46_qdqc.onnx"
    cfg.engine.chunk_ms = 100.0
    assert cfg.generator_path().name == "generator_my_voice_f36_qdqc.onnx"


def test_generator_and_encoder_precision_are_independent():
    """The two are separate knobs: fp32 generator fits the budget, fp32 encoder does not."""
    cfg = Config()
    cfg.model.int8_generator = False
    assert cfg.generator_path().name == "generator_my_voice_f46.onnx"
    assert cfg.encoder_path().name == "encoder_contentvec_qdq.onnx"  # unchanged

    cfg.model.int8_encoder = False
    assert cfg.encoder_path().name == "encoder_contentvec.onnx"


def test_config_round_trips_through_json(tmp_path):
    cfg = Config()
    cfg.params.key_shift = -3.0
    cfg.params.vad_db = -45.0
    cfg.engine.chunk_ms = 150.0
    path = tmp_path / "preset.json"
    cfg.save(path)

    loaded = Config.load(path)
    assert loaded.params.key_shift == -3.0
    assert loaded.params.vad_db == -45.0
    assert loaded.engine.chunk_ms == 150.0
    assert loaded.model.root == cfg.model.root
    assert loaded.generator_path() == cfg.generator_path()


def test_presets_accept_plain_strings(tmp_path):
    """Qt file dialogs return str, not Path; both have to work."""
    cfg = Config()
    cfg.params.key_shift = 5.0
    path = str(tmp_path / "preset.json")
    cfg.save(path)
    assert Config.load(path).params.key_shift == 5.0


def test_missing_model_file_names_the_available_chunk_sizes(tmp_path):
    from rtvc.convert.rvc import ModelFileMissing, RealRVC

    onnx = tmp_path / "rvc" / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "generator_my_voice_f46_qdqc.onnx").touch()  # only 200 ms exists

    cfg = Config()
    cfg.model.root = tmp_path
    cfg.engine.chunk_ms = 100.0  # asks for f36, which is absent

    with pytest.raises(ModelFileMissing) as excinfo:
        RealRVC(
            model=cfg.model,
            generator_path=cfg.generator_path(),
            encoder_path=cfg.encoder_path(),
            frames=cfg.generator_frames,
        )
    assert "200ms" in str(excinfo.value)


def test_missing_encoder_is_reported_before_inference(tmp_path):
    from rtvc.convert.rvc import ModelFileMissing, RealRVC

    onnx = tmp_path / "rvc" / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "generator_my_voice_f46_qdqc.onnx").touch()  # generator present, encoder not

    cfg = Config()
    cfg.model.root = tmp_path

    with pytest.raises(ModelFileMissing, match="encoder_contentvec"):
        RealRVC(
            model=cfg.model,
            generator_path=cfg.generator_path(),
            encoder_path=cfg.encoder_path(),
            frames=cfg.generator_frames,
        )


def test_gui_command_reports_a_missing_pyside_instead_of_a_traceback(monkeypatch, capsys):
    """app.py imports PySide6 inside run(), so guarding the module import catches nothing."""
    import builtins

    from rtvc.cli import main

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "PySide6":
            raise ImportError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert main(["gui"]) == 2
    assert "uv sync --extra gui" in capsys.readouterr().err


def test_exported_voices_is_empty_for_a_missing_directory(tmp_path):
    assert exported_voices(tmp_path / "nope") == {}


def test_exported_voices_groups_by_voice_and_variant(tmp_path):
    for name in (
        "generator_my_voice_f46.onnx",
        "generator_my_voice_f46_qdq.onnx",
        "generator_my_voice_f36_qdqc.onnx",
        "generator_other_f46_qdqc.onnx",
        "encoder_contentvec_qdq.onnx",  # not a generator, must be ignored
    ):
        (tmp_path / name).touch()

    voices = exported_voices(tmp_path)
    assert set(voices) == {"my_voice", "other"}
    assert voices["my_voice"].frames[46] == {"", "_qdq"}
    assert voices["my_voice"].frames[36] == {"_qdqc"}
    assert voices["my_voice"].chunk_sizes(20.0, "_qdqc") == [100.0]
    assert voices["my_voice"].chunk_sizes(20.0, "_qdq") == [200.0]
