"""The real RVC v2 inference path.

    audio(48k) -> 16k
      |- mel(128 bins, hop 160) -> RMVPE -> cent decode -> f0 at 100 Hz
      |                                       |- coarse pitch (integer bins 1..255)
      |                                       |- nsff0 (continuous f0)
      |- ContentVec -> (F50, 768) -> repeat 2x -> (F100, 768)
    generator(phone, phone_lengths, pitch, nsff0, sid) -> 40k -> device rate

Both the encoder and the pitch tracker need the full window: neither can be given a
bare chunk. Only the generator is restricted to the tail, which is what keeps the
per-chunk cost inside the real-time budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soxr

from ..catalog import exported_voices
from ..config import ModelConfig, RuntimeParams
from ..constants import ENC_SR, GEN_SR, SR
from .backends import GeneratorRunner, make_runner
from .base import Converter
from .features import MelExtractor, coarse_pitch, decode_f0


class ModelFileMissing(FileNotFoundError):
    """A required ONNX file is not on disk for the requested configuration."""


def _require(path: Path, model: ModelConfig) -> None:
    """Fail with what is actually available, not just what is missing.

    A generator is bound to one chunk size, so "file not found" on its own sends people
    looking for the wrong problem. Listing the chunk sizes that do exist usually answers
    the question outright.
    """
    if path.exists():
        return
    message = [f"{path} does not exist."]
    entry = exported_voices(model.onnx_dir).get(model.voice)
    if entry is not None:
        variant = model.variant if model.int8_generator else ""
        usable = entry.chunk_sizes(variant=variant)
        if usable:
            sizes = ", ".join(f"{ms:.0f}ms" for ms in usable)
            message.append(f"Chunk sizes exported for {model.voice!r} at this precision: {sizes}.")
        else:
            message.append(f"No generator is exported for {model.voice!r} at this precision.")
    else:
        message.append(f"No exported voice named {model.voice!r} under {model.onnx_dir}.")
    message.append(
        f"Export it with: python -m tools.export_onnx --voice voice/{model.voice}.pth "
        "--skip-encoder --chunk <ms>, then python -m tools.quantize --audio <recording>.wav"
    )
    raise ModelFileMissing(" ".join(message))


class RealRVC(Converter):
    """RVC conversion against an exported voice model.

    `frames` is fixed at construction because NSF bakes the sequence length into the
    graph as a constant: one generator file serves exactly one chunk size.
    """

    def __init__(
        self,
        model: ModelConfig,
        generator_path: Path,
        encoder_path: Path,
        frames: int,
        context_ms: float = 500.0,
        params: RuntimeParams | None = None,
    ) -> None:
        self.context_ms = context_ms
        self.frames = frames
        self.params = params if params is not None else RuntimeParams()

        _require(generator_path, model)
        # The encoder is easy to forget: fp32 and int8 are separate files, and only one
        # of them is usually present. Fail here with the path rather than inside the
        # runtime's own loader, which reports it far less clearly.
        _require(encoder_path, model)

        threads = model.threads
        backend = model.backend
        self.encoder = make_runner(encoder_path, threads, backend.get("enc", "ort"))
        self.pitch = make_runner(model.rmvpe_path, threads, backend.get("pit", "ort"))
        self.generator = GeneratorRunner(generator_path, threads, backend.get("gen", "ort"))
        self.mel = MelExtractor()

    def features(self, wav16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Content features and f0 over the whole window; neither can be cropped first."""
        feats = self.encoder(wav16[None, :])[0]  # (F50, 768)
        feats = np.repeat(feats, 2, axis=0)  # -> 100 Hz, as upstream RVC does

        mel = self.mel(wav16)  # (128, T)
        t = mel.shape[1]
        # The RMVPE U-Net downsamples five times, so the frame count must be a multiple of 32.
        pad = 32 * ((t - 1) // 32 + 1) - t
        if pad:
            mel = np.pad(mel, ((0, 0), (0, pad)))
        hidden = self.pitch(mel[None, :, :])[0][:t]  # (T, 360)

        f0 = decode_f0(hidden)
        shift = self.params.key_shift
        if shift:
            f0 = f0 * (2.0 ** (shift / 12.0))
        return feats, f0

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        wav16 = soxr.resample(window, SR, ENC_SR, quality="QQ").astype(np.float32)
        feats, f0 = self.features(wav16)

        n = min(feats.shape[0], f0.shape[0])
        feats, f0 = feats[:n], f0[:n]
        take = self.frames
        if n < take:
            # Short window (startup). Repeat the leading frame rather than pad with silence,
            # which would inject an audible attack at the head of the output.
            feats = np.pad(feats, ((take - n, 0), (0, 0)), mode="edge")
            f0 = np.pad(f0, (take - n, 0), mode="edge")
        feats, f0 = feats[-take:], f0[-take:]

        audio = self.generator(
            np.ascontiguousarray(feats[None, :, :], dtype=np.float32),
            np.array([take], dtype=np.int64),
            np.ascontiguousarray(coarse_pitch(f0)[None, :]),
            np.ascontiguousarray(f0[None, :], dtype=np.float32),
        )[0, 0]

        out = soxr.resample(audio, GEN_SR, SR, quality="QQ").astype(np.float32)
        if out.shape[0] < tail:
            out = np.pad(out, (tail - out.shape[0], 0))
        return out[-tail:]

    def close(self) -> None:
        for runner in (self.encoder, self.pitch, self.generator):
            runner.close()
