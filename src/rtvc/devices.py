"""Audio device discovery.

The virtual cable is the whole point of the output side: a meeting app cannot be told
to read from this process, so the converted audio is written to a loopback device that
the app sees as a microphone.
"""

from __future__ import annotations

from dataclasses import dataclass

# Substrings that identify the loopback devices installed by the common virtual cable
# drivers. Matched case-insensitively against the device name.
CABLE_HINTS = ("cable input", "cable output", "voicemeeter", "vb-audio")


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def is_virtual_cable(self) -> bool:
        lowered = self.name.lower()
        return any(hint in lowered for hint in CABLE_HINTS)


def list_devices() -> list[Device]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            Device(
                index=i,
                name=d["name"],
                hostapi=sd.query_hostapis(d["hostapi"])["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
            )
        )
    return out


def find_cable_output() -> Device | None:
    """The device to send converted audio to, i.e. the cable's playback side.

    Meeting apps then select the matching capture side ("CABLE Output") as their mic.
    """
    for d in list_devices():
        if d.is_output and "cable input" in d.name.lower():
            return d
    return None


def format_table(devices: list[Device]) -> str:
    lines = [
        f"{'idx':>4}  {'hostapi':<12} {'in':>3} {'out':>3}  {'rate':>7}  name",
        "-" * 88,
    ]
    for d in devices:
        mark = " *" if d.is_virtual_cable else "  "
        lines.append(
            f"{d.index:>4}{mark}{d.hostapi:<12} {d.max_input_channels:>3} "
            f"{d.max_output_channels:>3}  {d.default_samplerate:>7.0f}  {d.name}"
        )
    lines.append("")
    lines.append("  * = virtual cable")
    lines.append("  Pick: a real microphone for --in (prefer the WASAPI entry),")
    lines.append("        'CABLE Input' as an OUTPUT index for --out.")
    return "\n".join(lines)
