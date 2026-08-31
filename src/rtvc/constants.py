"""Fixed rates and model geometry shared across the package."""

from __future__ import annotations

# Device-side sample rate. Everything outside the model runs at this rate.
SR = 48000

# ContentVec and RMVPE both expect 16 kHz.
ENC_SR = 16000

# Generator input frame rate. 16000 / 160 (mel hop) = 100 Hz.
FRAME_HZ = 100

# Extra frames handed to the generator ahead of the emitted tail so its transposed
# convolutions have a filled receptive field. Matches the upstream RVC flow_head margin.
GUARD = 24

# The RVC generator synthesises at the rate the voice model was trained at.
GEN_SR = 40000


def generator_frames(chunk_ms: float, fade_ms: float) -> int:
    """Frame count baked into the generator ONNX for a given chunk/fade pair.

    NSF writes the sequence length into the graph as a constant, so a generator file
    is valid for exactly one frame count and each chunk size needs its own export.
    """
    return int(round((chunk_ms + fade_ms) / (1000.0 / FRAME_HZ))) + GUARD
