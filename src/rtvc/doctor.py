"""Pre-flight check.

Everything that has to be true before a conversion can run, checked in one place and
reported with the fix rather than the symptom. Most first-run failures are a missing
model file or a missing virtual cable, and both of those otherwise surface much later
as something that reads like a bug.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .catalog import exported_voices
from .config import Config
from .constants import generator_frames

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    status: str
    title: str
    detail: str = ""
    fix: str = ""


def _runtime_checks() -> list[Check]:
    out: list[Check] = []
    version = ".".join(str(v) for v in sys.version_info[:3])
    ok = (3, 11) <= sys.version_info < (3, 12)
    out.append(
        Check(
            OK if ok else WARN,
            f"Python {version}",
            "" if ok else "this project is pinned to 3.11",
            "" if ok else "install Python 3.11 and re-sync",
        )
    )

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        out.append(Check(OK, f"onnxruntime {ort.__version__}", ", ".join(providers)))
    except ImportError:
        out.append(
            Check(FAIL, "onnxruntime missing", fix="uv sync")
        )

    try:
        import openvino as ov

        devices = ov.Core().available_devices
        has_cpu = "CPU" in devices
        out.append(
            Check(
                OK if has_cpu else WARN,
                f"openvino {ov.__version__}",
                ", ".join(devices) or "no devices",
                "" if has_cpu else "the encoder falls back to onnxruntime and runs slower",
            )
        )
    except ImportError:
        out.append(
            Check(
                WARN,
                "openvino missing",
                "the encoder falls back to onnxruntime, several times slower",
                "uv sync",
            )
        )
    return out


def _model_checks(cfg: Config) -> list[Check]:
    out: list[Check] = []
    root = cfg.model.root
    if not root.is_dir():
        return [
            Check(
                FAIL,
                f"model root {root} does not exist",
                fix="python -m tools.fetch_models, then python -m tools.export_onnx",
            )
        ]

    for label, path in (("pitch tracker", cfg.model.rmvpe_path), ("encoder", cfg.encoder_path())):
        if path.is_file():
            out.append(Check(OK, f"{label} present", path.name))
        else:
            out.append(
                Check(
                    FAIL,
                    f"{label} missing",
                    str(path),
                    "python -m tools.fetch_models && python -m tools.export_onnx",
                )
            )

    voices = exported_voices(cfg.model.onnx_dir)
    if not voices:
        out.append(
            Check(
                FAIL,
                "no exported voices",
                str(cfg.model.onnx_dir),
                "python -m tools.export_onnx --voice voice/<name>.pth",
            )
        )
        return out

    out.append(Check(OK, f"{len(voices)} voice(s) exported", ", ".join(voices)))

    variant = cfg.model.variant if cfg.model.int8_generator else ""
    for name, entry in voices.items():
        sizes = entry.chunk_sizes(cfg.engine.fade_ms, variant)
        label = f"chunk sizes for {name!r}"
        if sizes:
            out.append(Check(OK, label, ", ".join(f"{ms:.0f}ms" for ms in sizes)))
        else:
            out.append(
                Check(
                    WARN,
                    label,
                    "none at the selected precision",
                    f"python -m tools.quantize --audio <recording>.wav --voice {name}",
                )
            )

    wanted = cfg.generator_path()
    frames = generator_frames(cfg.engine.chunk_ms, cfg.engine.fade_ms)
    if wanted.is_file():
        out.append(Check(OK, f"generator for chunk {cfg.engine.chunk_ms:.0f}ms", wanted.name))
    else:
        out.append(
            Check(
                WARN,
                f"no generator for chunk {cfg.engine.chunk_ms:.0f}ms (f{frames})",
                wanted.name,
                f"python -m tools.export_onnx --voice voice/{cfg.model.voice}.pth "
                f"--skip-encoder --chunk {cfg.engine.chunk_ms:.0f}",
            )
        )
    return out


def _audio_checks() -> list[Check]:
    try:
        from .devices import find_cable_output, list_devices
    except ImportError:
        return [Check(FAIL, "sounddevice missing", fix="uv sync")]

    try:
        devices = list_devices()
    except Exception as exc:  # noqa: BLE001 -- a broken audio stack is a finding, not a crash
        return [Check(FAIL, "cannot enumerate audio devices", str(exc))]

    inputs = [d for d in devices if d.is_input]
    outputs = [d for d in devices if d.is_output]
    out = [
        Check(
            OK if inputs else FAIL,
            f"{len(inputs)} input device(s)",
            inputs[0].name if inputs else "",
            "" if inputs else "connect a microphone",
        ),
        Check(
            OK if outputs else FAIL,
            f"{len(outputs)} output device(s)",
            outputs[0].name if outputs else "",
        ),
    ]

    cable = find_cable_output()
    if cable:
        out.append(Check(OK, "virtual cable found", f"[{cable.index}] {cable.name}"))
    else:
        out.append(
            Check(
                WARN,
                "no virtual cable",
                "conversion works, but a meeting app cannot pick it up",
                "install VB-CABLE from https://vb-audio.com/Cable/ and reboot",
            )
        )
    return out


def run_checks(cfg: Config) -> list[Check]:
    return _runtime_checks() + _model_checks(cfg) + _audio_checks()


def format_report(checks: list[Check]) -> str:
    marks = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    lines = []
    for check in checks:
        lines.append(f"[{marks[check.status]}] {check.title}")
        if check.detail:
            lines.append(f"           {check.detail}")
        if check.fix:
            lines.append(f"           fix: {check.fix}")

    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)
    lines.append("")
    if failures:
        lines.append(f"{failures} blocking problem(s), {warnings} warning(s).")
    elif warnings:
        lines.append(f"Ready to convert. {warnings} warning(s) above.")
    else:
        lines.append("Everything checks out.")
    return "\n".join(lines)


def worst_status(checks: list[Check]) -> int:
    if any(c.status == FAIL for c in checks):
        return 1
    return 0


def default_config(models: Path | None = None) -> Config:
    cfg = Config()
    if models is not None:
        cfg.model.root = models
    return cfg
