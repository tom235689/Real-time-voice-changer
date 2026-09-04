"""Discovering which voices and chunk sizes have actually been exported.

A generator ONNX is valid for exactly one frame count, so the set of usable chunk sizes
is a property of the files on disk rather than something the user may freely choose.
Both the CLI and the GUI ask here instead of guessing and failing at load time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .constants import FRAME_HZ, GUARD

# generator_<voice>_f<frames>[<variant>].onnx
_GENERATOR = re.compile(r"^generator_(?P<voice>.+?)_f(?P<frames>\d+)(?P<variant>_[a-z0-9]+)?\.onnx$")

_MS_PER_FRAME = 1000.0 / FRAME_HZ


def chunk_ms_for_frames(frames: int, fade_ms: float) -> float:
    """Inverse of constants.generator_frames."""
    return (frames - GUARD) * _MS_PER_FRAME - fade_ms


@dataclass(frozen=True)
class VoiceEntry:
    name: str
    frames: dict[int, set[str]]
    """Frame count -> the variant suffixes exported for it ('' means fp32)."""

    def chunk_sizes(self, fade_ms: float = 20.0, variant: str | None = None) -> list[float]:
        """Chunk sizes this voice can run at, optionally restricted to one variant."""
        out = []
        for frames, variants in sorted(self.frames.items()):
            if variant is not None and variant not in variants:
                continue
            ms = chunk_ms_for_frames(frames, fade_ms)
            if ms > 0:
                out.append(ms)
        return out


def exported_voices(onnx_dir: Path) -> dict[str, VoiceEntry]:
    """Scan a directory of exported generators and group them by voice."""
    found: dict[str, dict[int, set[str]]] = {}
    if not onnx_dir.is_dir():
        return {}
    for path in onnx_dir.glob("generator_*.onnx"):
        m = _GENERATOR.match(path.name)
        if not m:
            continue
        voice = m.group("voice")
        frames = int(m.group("frames"))
        variant = m.group("variant") or ""
        found.setdefault(voice, {}).setdefault(frames, set()).add(variant)
    return {name: VoiceEntry(name, frames) for name, frames in sorted(found.items())}
