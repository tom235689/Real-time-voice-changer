"""Wiring a Config into a running engine.

Both the CLI and the GUI go through here, so there is one place that knows how a
converter is built, how the audio stream is opened, and in what order things are torn
down. A control surface only needs start(), stop() and snapshot().
"""

from __future__ import annotations

from .config import Config
from .convert.base import Converter, Gain, Passthrough
from .convert.rvc import RealRVC
from .engine import Engine, Simulator, Telemetry


def build_converter(cfg: Config, kind: str = "rvc") -> Converter:
    """Create the converter named by `kind`.

    'passthrough' and 'gain' exist to prove the audio path independently of the model:
    if they are clean and 'rvc' is not, the fault is in inference, not in the plumbing.
    """
    if kind == "passthrough":
        return Passthrough()
    if kind == "gain":
        return Gain(cfg.params.output_gain)
    if kind != "rvc":
        raise ValueError(f"unknown converter: {kind}")
    return RealRVC(
        model=cfg.model,
        generator_path=cfg.generator_path(),
        encoder_path=cfg.encoder_path(),
        frames=cfg.generator_frames,
        context_ms=cfg.engine.context_ms,
        params=cfg.params,
    )


class Session:
    """Owns a converter, an engine and (outside simulation) an audio stream."""

    def __init__(self, cfg: Config, kind: str = "rvc") -> None:
        self.cfg = cfg
        self.kind = kind
        self.converter = build_converter(cfg, kind)
        self.engine = Engine(
            self.converter, audio=cfg.audio, settings=cfg.engine, params=cfg.params
        )
        self._stream = None

    @property
    def params(self):
        """Runtime controls. Writable while the session runs."""
        return self.cfg.params

    def start(self) -> None:
        """Open the audio stream and start converting. Warmup happens inside Engine.start."""
        import sounddevice as sd

        audio = self.cfg.audio
        if audio.input_device is None or audio.output_device is None:
            raise ValueError("input_device and output_device must be set before start()")

        self.engine.start()
        try:
            self._stream = sd.Stream(
                device=(audio.input_device, audio.output_device),
                samplerate=audio.sample_rate,
                blocksize=audio.block,
                channels=1,
                dtype="float32",
                latency="low",
                callback=self.engine.callback,
            )
            self._stream.start()
            self.engine.device_latency_ms = sum(self._stream.latency) * 1000.0
        except Exception:
            # Never leave a worker thread running behind a stream that failed to open.
            self.engine.stop()
            raise

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.engine.stop()

    def close(self) -> None:
        self.stop()
        self.converter.close()

    def snapshot(self) -> Telemetry:
        return self.engine.snapshot()

    def simulate(self, seconds: float, report_every: float = 5.0) -> None:
        """Run at real-time pace with no device, to measure timing and glitches."""
        Simulator(self.engine).run(seconds, report_every)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
