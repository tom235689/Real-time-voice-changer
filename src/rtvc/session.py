"""Wiring a Config into a running engine.

Both the CLI and the GUI go through here, so there is one place that knows how a
converter is built, how the audio stream is opened, and in what order things are torn
down. A control surface only needs start(), stop() and snapshot().
"""

from __future__ import annotations

import sys

from .config import Config
from .convert.base import Converter, Passthrough
from .convert.rvc import RealRVC
from .engine import Engine, Simulator, Telemetry


def ensure_com_apartment() -> None:
    """Give the calling thread a COM apartment, on Windows.

    Windows audio is COM based, and PortAudio cannot open a device from a thread that
    has none. It does not say so: the failure surfaces as an opaque
    "Unanticipated host error [PaErrorCode -9999]" naming a host API that may not even
    be the one in use.

    This matters because anything that keeps a UI responsive opens its device off the
    main thread. Measured on this machine: main thread opens fine, a worker thread fails
    every time, and the same worker succeeds once it has an apartment. A stream opened
    that way keeps running and still closes cleanly after the opening thread has exited,
    so the short-lived-thread pattern is safe.

    Safe to call repeatedly. A thread that already has an apartment keeps it, and the
    differing-mode result that comes back is not a problem here -- an apartment of
    either kind is all PortAudio needs.
    """
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED


def build_converter(cfg: Config, kind: str = "rvc") -> Converter:
    """Create the converter named by `kind`.

    'passthrough' exists to prove the audio path independently of the model: if it is
    clean and 'rvc' is not, the fault is in inference, not in the plumbing. Combine it
    with the output gain to make the path audible.
    """
    if kind == "passthrough":
        return Passthrough()
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
        """Claim the audio device, warm the model up, then let audio flow.

        The device is opened before the warmup rather than after, for two reasons. A bad
        device index or a busy endpoint then fails in under a second instead of after
        the model has spent the better part of a minute loading. And holding the device
        across the warmup stops it from being suspended in the meantime -- display-audio
        and Bluetooth endpoints in particular power down quickly when idle, and a device
        that enumerated cleanly a moment ago will refuse to open.

        Opening is not streaming: the callback does not run until start() on the stream,
        so the engine is ready before a single block arrives.
        """
        import sounddevice as sd

        audio = self.cfg.audio
        if audio.input_device is None or audio.output_device is None:
            raise ValueError("input_device and output_device must be set before start()")

        ensure_com_apartment()
        self._stream = sd.Stream(
            device=(audio.input_device, audio.output_device),
            samplerate=audio.sample_rate,
            blocksize=audio.block,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self.engine.callback,
        )
        try:
            self.engine.start()
            self._stream.start()
            self.engine.device_latency_ms = sum(self._stream.latency) * 1000.0
        except Exception:
            # Never leave a held device or a running worker behind a failed start.
            self.engine.stop()
            self._stream.close()
            self._stream = None
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
