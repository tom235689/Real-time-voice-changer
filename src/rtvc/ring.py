"""Single-producer / single-consumer ring buffers.

These sit between the audio callback and the inference worker. Indices are kept as
cumulative frame counts rather than wrapped offsets: wrap-around arithmetic stays
trivial and an overrun (the worker fell so far behind that unread data was
overwritten) becomes detectable instead of silently corrupting audio.

Under CPython the GIL makes an int store/load atomic, so SPSC access needs no lock.
Buffers are allocated once at construction; no code path below allocates again.
"""

from __future__ import annotations

import numpy as np


class Overrun(Exception):
    """The worker fell behind and the requested window was already overwritten."""


class InputRing:
    """Written by the audio callback, peeked at arbitrary positions by the worker.

    There is no read pointer: the worker tracks its own position, because it needs a
    sliding window that overlaps previous reads rather than a consuming read.
    """

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._w = 0  # cumulative frames written

    @property
    def written(self) -> int:
        return self._w

    @property
    def capacity(self) -> int:
        return self._cap

    def write(self, data: np.ndarray) -> None:
        """Called from the audio callback. Two memcpys and an int store, nothing else."""
        n = data.shape[0]
        start = self._w % self._cap
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = data
        else:
            k = self._cap - start
            self._buf[start:] = data[:k]
            self._buf[: end - self._cap] = data[k:]
        # Publish last, so the worker never observes a partially written region.
        self._w += n

    def read_window(self, end_pos: int, length: int, out: np.ndarray) -> None:
        """Copy [end_pos - length, end_pos) into out. Does not consume."""
        start_pos = end_pos - length
        if start_pos < 0:
            raise ValueError("window starts before the beginning of the stream")
        if self._w - start_pos > self._cap:
            raise Overrun(f"overwritten: lag {self._w - start_pos} > capacity {self._cap}")
        start = start_pos % self._cap
        end = start + length
        if end <= self._cap:
            out[:] = self._buf[start:end]
        else:
            k = self._cap - start
            out[:k] = self._buf[start:]
            out[k:] = self._buf[: end - self._cap]


class OutputRing:
    """Written by the inference worker, consumed by the audio callback."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._w = 0
        self._r = 0

    @property
    def available(self) -> int:
        return self._w - self._r

    @property
    def space(self) -> int:
        return self._cap - self.available

    def write(self, data: np.ndarray) -> bool:
        """Called from the worker. Returns False when the ring is full (output backed up)."""
        n = data.shape[0]
        if n > self.space:
            return False
        start = self._w % self._cap
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = data
        else:
            k = self._cap - start
            self._buf[start:] = data[:k]
            self._buf[: end - self._cap] = data[k:]
        self._w += n
        return True

    def read_into(self, out: np.ndarray) -> bool:
        """Called from the audio callback. Fills silence and returns False on underrun."""
        n = out.shape[0]
        if self.available < n:
            out[:] = 0.0
            return False
        start = self._r % self._cap
        end = start + n
        if end <= self._cap:
            out[:] = self._buf[start:end]
        else:
            k = self._cap - start
            out[:k] = self._buf[start:]
            out[k:] = self._buf[: end - self._cap]
        self._r += n
        return True

    def prefill(self, frames: int) -> None:
        """Push silence to build a safety margin against underruns.

        Whatever is pushed here becomes end-to-end latency, one frame for one frame.
        Call from the producer side only, or the SPSC assumption breaks.
        """
        if frames <= 0:
            return
        chunk = np.zeros(min(frames, self._cap), dtype=np.float32)
        left = frames
        while left > 0:
            n = min(left, chunk.shape[0], self.space)
            if n <= 0:
                return  # ring is full; the margin cannot grow any further
            self.write(chunk[:n])
            left -= n
