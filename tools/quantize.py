"""Quantise the exported ONNX to int8.

    python -m tools.quantize --audio voice/sample.wav          # generators, calibrated
    python -m tools.quantize --audio voice/sample.wav --encoder  # the encoder too
    python -m tools.quantize --random                           # speed testing only

Calibration data decides whether the result is usable. Random tensors put the
activation ranges nowhere near where real speech puts them, and the generator comes out
broken -- measured at 2.4x the baseline log-spectral distance with output RMS down 42%.
It still runs, and it still benchmarks fast, which is exactly why it is easy to ship by
accident.

So real audio is the default and produces the `_qdqc` files the runtime prefers.
`--random` is available for timing work and writes `_qdq`, which the runtime will load
only if asked for explicitly.

The best calibration audio is a real recording of the person whose voice is being
converted, run through the same pipeline that will run live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError as exc:  # onnx ships with the tools extra, not with the runtime
    raise SystemExit(
        f"quantisation needs the 'tools' extra ({exc.name} is missing). "
        "Install it with: uv sync --extra tools"
    ) from exc

from rtvc.config import Config
from rtvc.constants import ENC_SR, SR, generator_frames

from .paths import add_model_root, onnx_dir

CALIBRATED_SUFFIX = "_qdqc"
RANDOM_SUFFIX = "_qdq"


class ListReader(CalibrationDataReader):
    def __init__(self, samples: list[dict]) -> None:
        self._it = iter(samples)

    def get_next(self):
        return next(self._it, None)


def quantise(src: Path, dst: Path, reader: CalibrationDataReader) -> bool:
    if not src.exists():
        print(f"  skip {src.name} (not exported)")
        return False
    quantize_static(
        src,
        dst,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(
        f"  {src.name}  ->  {dst.name}  "
        f"({src.stat().st_size / 1e6:.0f} -> {dst.stat().st_size / 1e6:.0f} MB)"
    )
    return True


def collect_from_audio(
    audio_path: Path, cfg: Config, frames: int, limit: int
) -> list[dict]:
    """Push real audio through the fp32 pipeline and keep the generator's inputs.

    This is the whole point of calibrating properly: the tensors handed to the quantiser
    are the tensors the generator will actually see.
    """
    import soxr

    from rtvc import wavio
    from rtvc.convert.features import coarse_pitch
    from rtvc.convert.rvc import RealRVC

    fp32 = Config.from_dict(cfg.to_dict())
    fp32.model.int8_encoder = False
    fp32.model.int8_generator = False
    converter = RealRVC(
        model=fp32.model,
        generator_path=fp32.generator_path(),
        encoder_path=fp32.encoder_path(),
        frames=frames,
        context_ms=fp32.engine.context_ms,
    )

    audio, rate = wavio.read(audio_path)
    if rate != SR:
        audio = soxr.resample(audio, rate, SR).astype(np.float32)

    per_ms = SR / 1000.0
    chunk = int(cfg.engine.chunk_ms * per_ms)
    fade = int(cfg.engine.fade_ms * per_ms)
    context = int(cfg.engine.context_ms * per_ms)
    padded = np.concatenate([np.zeros(context + fade, dtype=np.float32), audio])

    samples: list[dict] = []
    pos = context + fade
    window_len = context + chunk + fade
    try:
        while pos + chunk <= padded.shape[0] and len(samples) < limit:
            window = padded[pos + chunk - window_len : pos + chunk]
            wav16 = soxr.resample(window, SR, ENC_SR, quality="QQ").astype(np.float32)
            feats, f0 = converter.features(wav16)
            n = min(feats.shape[0], f0.shape[0])
            feats, f0 = feats[:n], f0[:n]
            if n < frames:
                feats = np.pad(feats, ((frames - n, 0), (0, 0)), mode="edge")
                f0 = np.pad(f0, (frames - n, 0), mode="edge")
            feats, f0 = feats[-frames:], f0[-frames:]
            samples.append(
                {
                    "phone": np.ascontiguousarray(feats[None, :, :], dtype=np.float32),
                    "phone_lengths": np.array([frames], dtype=np.int64),
                    "pitch": np.ascontiguousarray(coarse_pitch(f0)[None, :]),
                    "nsff0": np.ascontiguousarray(f0[None, :], dtype=np.float32),
                    "sid": np.array([0], dtype=np.int64),
                }
            )
            pos += chunk
    finally:
        converter.close()
    return samples


def random_samples(frames: int, count: int, rng: np.random.Generator) -> list[dict]:
    return [
        {
            "phone": rng.standard_normal((1, frames, 768)).astype(np.float32),
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": rng.integers(0, 255, (1, frames)).astype(np.int64),
            "nsff0": (rng.random((1, frames)) * 200 + 100).astype(np.float32),
            "sid": np.array([0], dtype=np.int64),
        }
        for _ in range(count)
    ]


def report_coverage(samples: list[dict]) -> None:
    phone = np.concatenate([s["phone"].ravel() for s in samples])
    f0 = np.concatenate([s["nsff0"].ravel() for s in samples])
    print(f"  phone range [{phone.min():+.2f}, {phone.max():+.2f}]  sd {phone.std():.3f}")
    print(
        f"  nsff0 range [{f0.min():.1f}, {f0.max():.1f}] Hz  "
        f"voiced {(f0 > 0).mean() * 100:.0f}% of frames"
    )
    if (f0 > 0).mean() < 0.2:
        print("  WARNING: mostly unvoiced. Use audio with more speech in it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, help="wav to calibrate with (recommended)")
    parser.add_argument(
        "--random",
        action="store_true",
        help="calibrate on random tensors; produces measurably broken audio, timing use only",
    )
    parser.add_argument("--voice", default="my_voice")
    parser.add_argument(
        "--chunk", type=float, nargs="+", default=[100.0, 150.0, 200.0, 250.0],
        help="chunk sizes in ms to quantise generators for",
    )
    parser.add_argument("--fade", type=float, default=20.0)
    parser.add_argument("--context", type=float, default=500.0)
    parser.add_argument("--limit", type=int, default=32, help="calibration chunks per generator")
    parser.add_argument("--encoder", action="store_true", help="also quantise the content encoder")
    add_model_root(parser)
    args = parser.parse_args(argv)

    if not args.random and args.audio is None:
        print("Pass --audio <wav> to calibrate properly, or --random for timing work only.")
        return 2
    if args.audio is not None and not args.audio.exists():
        print(f"{args.audio} does not exist.")
        return 2

    out = onnx_dir(args.models)
    suffix = RANDOM_SUFFIX if args.random else CALIBRATED_SUFFIX
    rng = np.random.default_rng(0)

    if args.random:
        print("Calibrating on RANDOM tensors. The result is for timing only; it will")
        print("sound wrong. Re-run with --audio before using it to talk to anyone.\n")

    cfg = Config()
    cfg.model.root = args.models
    cfg.model.voice = args.voice
    cfg.engine.fade_ms = args.fade
    cfg.engine.context_ms = args.context

    if args.encoder:
        print("Encoder")
        # The encoder is length-agnostic, so one calibration window covers every chunk.
        window = int(ENC_SR * (args.context + max(args.chunk) + args.fade) / 1000)
        reader = ListReader(
            [{"wav": rng.standard_normal((1, window)).astype(np.float32)} for _ in range(8)]
        )
        quantise(out / "encoder_contentvec.onnx", out / "encoder_contentvec_qdq.onnx", reader)
        print()

    print("Generators")
    written = 0
    for chunk_ms in args.chunk:
        frames = generator_frames(chunk_ms, args.fade)
        src = out / f"generator_{args.voice}_f{frames}.onnx"
        if not src.exists():
            print(f"  skip chunk {chunk_ms:.0f}ms: {src.name} not exported")
            continue

        cfg.engine.chunk_ms = chunk_ms
        if args.random:
            samples = random_samples(frames, 8, rng)
        else:
            samples = collect_from_audio(args.audio, cfg, frames, args.limit)
            if not samples:
                print(
                    f"  skip chunk {chunk_ms:.0f}ms: audio is shorter than "
                    f"context + chunk ({args.context + chunk_ms:.0f}ms)"
                )
                continue
            print(f"  chunk {chunk_ms:.0f}ms: {len(samples)} calibration chunks")
            report_coverage(samples)

        if quantise(src, out / f"generator_{args.voice}_f{frames}{suffix}.onnx", ListReader(samples)):
            written += 1

    if not written:
        print("\nNothing was quantised.")
        return 1
    print(f"\nwrote {written} generator(s) with suffix {suffix!r} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
