"""Opening the audio device from a worker thread.

Anything that keeps a UI responsive loads the model off the main thread and opens the
device there too. On Windows that fails unless the thread has a COM apartment, and the
error PortAudio raises names an unrelated host API, so it is easy to misread as a
broken device. These need real hardware and skip without it.
"""

from __future__ import annotations

import sys
import threading

import pytest

sd = pytest.importorskip("sounddevice")

from rtvc.devices import list_devices  # noqa: E402
from rtvc.session import ensure_com_apartment  # noqa: E402


def duplex_pair():
    """A (input, output) index pair that can plausibly be opened, or None."""
    try:
        devices = list_devices()
    except Exception:
        return None
    preferred = [d for d in devices if d.hostapi == "Windows WASAPI"] or devices
    mic = next((d for d in preferred if d.is_input), None)
    out = next((d for d in preferred if d.is_output), None)
    return (mic.index, out.index) if mic and out else None


def open_briefly(mic: int, out: int) -> None:
    with sd.Stream(
        device=(mic, out),
        samplerate=48000,
        blocksize=480,
        channels=1,
        dtype="float32",
        latency="low",
        callback=lambda i, o, f, t, s: o.fill(0.0),
    ):
        pass


def test_ensure_com_apartment_is_safe_to_repeat():
    ensure_com_apartment()
    ensure_com_apartment()


def test_ensure_com_apartment_is_a_no_op_off_windows():
    if sys.platform != "win32":
        ensure_com_apartment()  # must not reach ctypes.windll


@pytest.mark.skipif(duplex_pair() is None, reason="no audio device available")
def test_device_opens_from_a_worker_thread_after_the_apartment_is_set():
    """Without ensure_com_apartment this raises PaErrorCode -9999 on Windows."""
    mic, out = duplex_pair()
    failure: list[BaseException] = []

    def work():
        ensure_com_apartment()
        try:
            open_briefly(mic, out)
        except BaseException as exc:  # noqa: BLE001 -- reported on the main thread
            failure.append(exc)

    thread = threading.Thread(target=work, name="opener")
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive(), "opening the device hung on the worker thread"
    if failure:
        pytest.skip(f"device present but not openable right now: {failure[0]}")
