"""The converter contract.

    process(window, tail) -> the converted audio for the last `tail` samples of window

Window layout is [context | chunk | fade]; tail is [chunk + fade]. The context is
consumed by the encoder and the pitch tracker, which cannot see a chunk in isolation,
but it contributes nothing to the output.

Splitting window from tail is where the compute budget is won. The encoder has to run
over the whole window, but the generator is convolutional and only needs to synthesise
the tail. At a 1220 ms window and a 220 ms tail that is 5.5x less generator work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Converter(ABC):
    context_ms: float = 0.0
    """Leading audio process() requires before the tail it is asked to produce."""

    @abstractmethod
    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        """Return exactly `tail` samples, aligned to the end of `window`."""

    def warmup(self, window_frames: int, tail: int) -> None:
        """Force lazy graph compilation and allocation before the clock starts."""
        for _ in range(2):
            self.process(np.zeros(window_frames, dtype=np.float32), tail)

    def close(self) -> None:  # noqa: B027 -- most converters hold nothing to release
        """Release inference sessions. Safe to call more than once."""


class Passthrough(Converter):
    """No conversion. Verifies the audio path end to end."""

    context_ms = 0.0

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        return window[-tail:]


class Gain(Converter):
    """Passthrough with a level change, so it is audible that the path is live."""

    context_ms = 0.0

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        return window[-tail:] * self.gain
