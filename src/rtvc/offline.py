"""Offline file conversion.

Runs the same window/tail/crossfade geometry as the engine, but as fast as the machine
allows and without rings or threads. Use it to judge audio quality by ear, separately
from any question about whether the machine keeps up in real time.
"""

from __future__ import annotations

import time

import numpy as np
import soxr

from .config import Config
from .convert.base import Converter


def convert_array(
    converter: Converter,
    audio: np.ndarray,
    chunk: int,
    fade: int,
    context: int,
    progress: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Convert a whole array. Returns the output and the per-chunk inference times."""
    # Lead the signal with silence so the first real chunk already has its context, and
    # trail it so the final chunk is emitted rather than left in the pipeline.
    padded = np.concatenate(
        [
            np.zeros(context + fade, dtype=np.float32),
            audio.astype(np.float32),
            np.zeros(chunk * 2, dtype=np.float32),
        ]
    )

    window_len = context + chunk + fade
    prev_tail = np.zeros(fade, dtype=np.float32)
    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    fade_in = np.sin(t * np.pi / 2) ** 2
    fade_out = np.cos(t * np.pi / 2) ** 2

    out_blocks: list[np.ndarray] = []
    timings: list[float] = []
    pos = context + fade
    total = max(1, (padded.shape[0] - pos) // chunk)
    done = 0

    while pos + chunk <= padded.shape[0]:
        window = padded[pos + chunk - window_len : pos + chunk]
        t0 = time.perf_counter()
        tail = converter.process(window, chunk + fade)
        timings.append((time.perf_counter() - t0) * 1000.0)

        emit = np.empty(chunk, dtype=np.float32)
        emit[:fade] = prev_tail * fade_out + tail[:fade] * fade_in
        emit[fade:] = tail[fade:chunk]
        prev_tail = tail[chunk:].copy()
        out_blocks.append(emit)

        pos += chunk
        done += 1
        if progress and done % 10 == 0:
            print(f"\r  {done}/{total} chunks", end="", flush=True)

    if progress:
        print(f"\r  {done}/{total} chunks")
    if not out_blocks:
        return np.zeros(0, dtype=np.float32), timings

    # Each pass emits the block starting at pos - fade, not pos: the crossfade region
    # belongs to the previous chunk's span. Live that is simply part of the latency, but
    # a converted file has to line up with its input, so drop the leading fade here.
    result = np.concatenate(out_blocks)[fade:]
    return result[: audio.shape[0]], timings


def convert_file_audio(
    converter: Converter, audio: np.ndarray, rate: int, cfg: Config, progress: bool = False
) -> tuple[np.ndarray, list[float]]:
    """Resample to the engine rate if needed, convert, and resample back."""
    sr = cfg.audio.sample_rate
    source = audio if rate == sr else soxr.resample(audio, rate, sr).astype(np.float32)

    per_ms = sr / 1000.0
    converted, timings = convert_array(
        converter,
        source,
        chunk=int(cfg.engine.chunk_ms * per_ms),
        fade=int(cfg.engine.fade_ms * per_ms),
        context=int(cfg.engine.context_ms * per_ms),
        progress=progress,
    )
    if rate != sr:
        converted = soxr.resample(converted, sr, rate).astype(np.float32)
    return converted, timings


def summarise(timings: list[float], chunk_ms: float) -> str:
    if not timings:
        return "  no chunks processed"
    arr = np.asarray(timings)
    rtf = float(arr.mean()) / chunk_ms
    return (
        f"  chunks {arr.size}   inference p50 {np.percentile(arr, 50):.1f}ms   "
        f"p95 {np.percentile(arr, 95):.1f}ms   max {arr.max():.1f}ms\n"
        f"  budget {chunk_ms:.0f}ms per chunk   mean real-time factor {rtf:.3f}   "
        + ("(real-time capable)" if rtf < 0.7 else "(too slow for real time)")
    )
