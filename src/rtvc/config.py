"""Configuration objects shared by the CLI, the engine and the GUI.

Two kinds of settings live here, and the split matters:

    *Startup* settings (sample rate, chunk size, model variant) are baked into
    allocated buffers and compiled inference graphs. Changing one means rebuilding
    the engine.

    *Runtime* settings (pitch shift, VAD threshold, bypass, gain) are read fresh on
    every chunk by the worker. A control surface may write them at any time; under
    CPython a plain attribute store is atomic, so no lock is needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import SR, generator_frames

DEFAULT_BACKEND: dict[str, str] = {"enc": "ov", "pit": "ort", "gen": "ort"}


@dataclass
class RuntimeParams:
    """Mutable while the engine runs. Written by a control surface, read by the worker."""

    key_shift: float = 0.0
    """Pitch shift in semitones applied to the extracted f0."""

    vad_db: float | None = None
    """Chunk RMS gate in dBFS; None disables it. Below the gate inference is skipped.

    This is thermal protection for a 35 W part rather than a latency optimisation:
    real speech occupies a fraction of a meeting, and skipping the rest keeps the
    package off its throttle point.
    """

    vad_hang_chunks: int = 3
    """Chunks of hangover after speech drops below the gate, so word tails survive."""

    bypass: bool = False
    """Route input straight to output without inference. Useful for A/B and plumbing checks."""

    output_gain: float = 1.0


@dataclass
class AudioConfig:
    sample_rate: int = SR
    block: int = 480
    input_device: int | None = None
    output_device: int | None = None

    @property
    def block_ms(self) -> float:
        return self.block * 1000.0 / self.sample_rate


@dataclass
class EngineConfig:
    chunk_ms: float = 200.0
    """Audio emitted per inference pass. Lower means less latency but a tighter budget."""

    fade_ms: float = 20.0
    """Equal-power crossfade between consecutive chunks, hiding the seam."""

    context_ms: float = 500.0
    """Leading audio the encoder sees but that never reaches the output."""

    prefill_ms: float | None = None
    """Output silence pushed before playback starts. None measures it during warmup."""

    adaptive_prefill: bool = True
    """Grow the prefill margin after a steady-state underrun until underruns stop."""


@dataclass
class ModelConfig:
    root: Path = Path("models")
    voice: str = "my_voice"

    int8_encoder: bool = True
    """Quantise the content encoder. Turning this off costs far more than it sounds:
    the int8 encoder under OpenVINO is several times faster than fp32, and fp32 here is
    on its own enough to push the pipeline past its real-time budget."""

    int8_generator: bool = True
    """Quantise the generator. fp32 is the quality option worth reaching for; it is
    slower but, with the encoder left at int8, still fits."""

    variant: str = "_qdqc"
    """int8 generator suffix. '_qdqc' is calibrated on real speech, '_qdq' is not."""

    threads: int = 12
    backend: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BACKEND))
    """Inference backend per stage. The winner differs by stage on this hardware:
    the int8 encoder is much faster under OpenVINO, while the int8 generator only
    runs correctly under ONNX Runtime."""

    @property
    def onnx_dir(self) -> Path:
        return self.root / "rvc" / "onnx"

    @property
    def rmvpe_path(self) -> Path:
        # RMVPE is always fp32: its int8 graph does not survive quantisation.
        return self.root / "rvc" / "rmvpe.onnx"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    params: RuntimeParams = field(default_factory=RuntimeParams)

    @property
    def generator_frames(self) -> int:
        return generator_frames(self.engine.chunk_ms, self.engine.fade_ms)

    def generator_path(self) -> Path:
        suffix = self.model.variant if self.model.int8_generator else ""
        return self.model.onnx_dir / f"generator_{self.model.voice}_f{self.generator_frames}{suffix}.onnx"

    def encoder_path(self) -> Path:
        suffix = "_qdq" if self.model.int8_encoder else ""
        return self.model.onnx_dir / f"encoder_contentvec{suffix}.onnx"

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"]["root"] = str(self.model.root)
        return d

    def save(self, path: str | Path) -> None:
        # Qt's file dialogs hand back plain strings, so accept both rather than making
        # every caller remember to wrap.
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        model = dict(d.get("model", {}))
        if "root" in model:
            model["root"] = Path(model["root"])
        return cls(
            audio=AudioConfig(**d.get("audio", {})),
            engine=EngineConfig(**d.get("engine", {})),
            model=ModelConfig(**model),
            params=RuntimeParams(**d.get("params", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
