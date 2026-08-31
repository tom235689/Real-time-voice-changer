"""Minimal WAV read/write on the standard library.

Offline conversion only needs mono PCM in and out. Pulling in soundfile or scipy for
that would add a dependency the real-time path never touches.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1]. Extra channels are averaged down."""
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        # 8-bit WAV is unsigned, centred on 128.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), rate


def write(path: Path, data: np.ndarray, rate: int) -> None:
    """Write mono float32 as 16-bit PCM, clipping rather than wrapping on overflow."""
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
