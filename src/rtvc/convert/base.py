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

    def close(self) -> None:  # noqa: B027 -- most converters hold nothing to release
        """Release inference sessions. Safe to call more than once."""


class Passthrough(Converter):
    """No conversion. Verifies the audio path end to end.

    Level changes belong to the engine's output gain, which applies to every converter;
    a converter that scaled as well would have its gain applied twice.
    """

    context_ms = 0.0

    def process(self, window: np.ndarray, tail: int) -> np.ndarray:
        return window[-tail:]
