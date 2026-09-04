# rtvc — real-time voice changer

RVC voice conversion, live, on a machine with no discrete GPU
(i5-12400T / Intel UHD 730). Output is written to a virtual audio cable so that Zoom,
Google Meet, Discord or anything else can pick it up as a microphone.

## Targets

| Metric | Goal | Acceptable |
|---|---|---|
| Processing latency | 300 ms | 500 ms |
| Audio glitches | 0 in 10 minutes | 2 in 10 minutes |
| CPU load | under 60% | under 80% |

## Why this is not obvious

Real-time conversion lives or dies on one inequality:

```
real-time factor = inference time / chunk duration  <  0.7
```

200 ms of audio has to be converted in well under 200 ms, forever, on a 35 W CPU with
no CUDA. Three things make it fit:

- **Only the tail is synthesised.** The encoder and the pitch tracker need the whole
  window for context, but the generator is convolutional and only has to produce the
  part that is actually emitted. At a 1220 ms window and a 220 ms tail that is 5.5x
  less generator work.
- **Each stage runs on whichever runtime is faster for it.** The int8 encoder is much
  faster under OpenVINO; the int8 generator only produces correct output under ONNX
  Runtime. The two run side by side in one process.
- **The prefill is the latency.** A sample entering at `t` leaves at `t + prefill`.
  Chunk size and inference time do not add to latency — they set the floor the prefill
  cannot go below: `prefill >= chunk + worst-case inference`. The engine measures that
  floor during warmup and grows the margin if a steady-state underrun ever happens.

## Status

The engine, the CLI and the desktop panel are implemented. Everything except the live
device path can be exercised today; that last step needs a virtual cable driver, which
has to be installed by hand.

| Area | State |
|---|---|
| Audio engine, rings, crossfade | implemented |
| RVC inference path (encoder / RMVPE / generator) | implemented |
| Offline file conversion | implemented |
| Real-time simulation (no device) | implemented |
| CLI | implemented |
| Desktop GUI | implemented |
| Model export and quantisation tools | implemented |
| Live device path into a meeting app | **needs VB-CABLE installed** |

Operating points inherited from the earlier prototype of this pipeline, on the same
hardware and the same exported models: real-time factor 0.287, chunk 200 ms,
processing latency around 260–376 ms, zero underruns over a 30 s run. The rewrite keeps
the same geometry and the same model files, but those numbers should be re-measured
here with `rtvc simulate` rather than assumed.

## Install

```powershell
cd D:\Project\Real-time-voice-changer
. .\dev.ps1                 # puts uv on PATH; once per terminal
uv sync --extra gui --group dev
```

Model weights are not in git. `models/` must contain:

```
models/rvc/rmvpe.onnx                              pitch tracker (fp32; int8 breaks it)
models/rvc/onnx/encoder_contentvec_qdq.onnx        content encoder (int8)
models/rvc/onnx/generator_<voice>_f<frames>*.onnx  one generator per chunk size
```

A generator ONNX has its sequence length baked in by NSF, so each chunk size needs its
own export. The GUI only offers chunk sizes that actually exist on disk.

## Preparing models

Only needed to add a voice or a chunk size; the runtime never touches these.

```powershell
uv sync --extra tools                                    # torch, onnx, transformers

python -m tools.fetch_models                             # base assets, ~780 MB
python -m tools.inspect_voice voice\my_voice.pth         # check sr / f0 / version first
python -m tools.export_onnx --voice voice\my_voice.pth   # -> fp32 ONNX
python -m tools.quantize --audio voice\sample.wav        # -> int8, calibrated
```

`inspect_voice` is worth running before anything else: a v1 model or one trained with
`f0=0` needs a different pipeline than this repo implements, and it says so immediately
rather than failing later in the export.

**Calibrate on real speech.** `tools.quantize` defaults to it and writes `_qdqc`.
Calibrating on random tensors instead — `--random`, which writes `_qdq` — measures 2.4x
the baseline log-spectral distance with output RMS down 42%. It still runs and still
benchmarks fast, which is what makes it easy to ship by mistake. Use a recording of the
voice being converted.

To add a chunk size, export and quantise for it specifically:

```powershell
python -m tools.export_onnx --voice voice\my_voice.pth --skip-encoder --chunk 300
python -m tools.quantize --audio voice\sample.wav --chunk 300
```

The encoder is voice-independent and length-agnostic, so `--skip-encoder` is right for
everything after the first export.

## Use

### Check the install first

```powershell
uv run rtvc doctor
```

Checks the runtime, the model files, the exported voices and chunk sizes, the audio
devices and the virtual cable, and prints the command that fixes anything missing.
Exits non-zero only when something actually blocks conversion. Run this before anything
else; most first-run trouble is a missing model file or a missing cable, and both
otherwise surface much later as something that reads like a bug.

```powershell
uv run rtvc devices
```

Lists audio devices and flags any virtual cable it finds.

### Judge quality by ear, without a device

```powershell
uv run rtvc convert --in voice\sample.wav --out voice\converted.wav --chunk 200
```

Runs the same window/tail/crossfade geometry as the live engine, as fast as the machine
allows, and reports the real-time factor.

### Measure timing, without a device

```powershell
uv run rtvc simulate --seconds 45 --chunk 200
```

Drives the engine at real-time pace with no audio hardware, and reports inference
percentiles, underruns and the latency breakdown. This is the check that matters before
touching a meeting.

### Live, into a meeting app

Requires [VB-CABLE](https://vb-audio.com/Cable/): install as administrator, reboot, then
set both `CABLE Input` and `CABLE Output` to 48000 Hz 16-bit in Windows sound settings.

```powershell
uv run rtvc devices                                  # find the indices
uv run rtvc run --converter passthrough --in 17 --out 24    # verify plumbing first
uv run rtvc run --in 17 --out 24 --chunk 200 --vad-db -45   # then convert
```

Then set the meeting app's microphone to `CABLE Output`.

### Desktop panel

```powershell
uv run rtvc gui
```

Device pickers, live pitch/gain/gate/bypass controls, latency and glitch readouts, and
preset save/load. It remembers the devices, voice and chunk size that last started
successfully, matching devices by name so a reconnected headset does not silently
select something else.

Input and output level meters sit at the top of the measurements. They answer the
question that actually comes up — "can they hear me?" — and separate the two failures
that otherwise look identical: a silent input meter is the microphone, a live input
with a silent output is the conversion.

## Meeting app settings

Conferencing apps treat converted speech as noise and will destroy it. Turn that off:

- **Zoom** — microphone `CABLE Output`, automatic volume adjustment **off**, background
  noise suppression **low**.
- **Google Meet** — microphone `CABLE Output`, noise cancellation **off**.

## Options worth knowing

| Option | Effect |
|---|---|
| `--chunk 150` | Lower latency, tighter budget. Needs a generator exported for it. |
| `--gen-fp32` | fp32 generator: better quality. Measured at chunk 200 ms — median 191 ms against a 200 ms budget, p95 211 ms, zero underruns, latency 491 ms. It sustains, but with no slack for a competing load. |
| `--enc-fp32` | fp32 encoder. Measured median 220 ms against the same budget, 92 underruns in 15 s. It does not sustain; kept for comparison only. |
| `--key -2` | Pitch shift in semitones. |
| `--vad-db -45` | Skip inference during silence. Thermal protection on a 35 W part, not a latency win. |
| `--converter passthrough` | Verify the audio path with the model out of the picture. Combine with `--gain` to make it audible. |
| `--threads 6` | Lower if inference contends with the audio callback. |

## Development

```powershell
uv run pytest -q          # preprocessing pinned against librosa and torch
uv run ruff check .
```

## Design notes

- Inference is ONNX Runtime and OpenVINO. PyTorch appears only in the offline export
  tools, never at runtime.
- The audio callback does memcpy and nothing else: no allocation, no logging, no Python
  loops, no inference.
- Rings are single-producer/single-consumer with cumulative indices, so an overrun is
  detected rather than silently corrupting audio.
- GC is frozen while the engine runs. A collection pause lands directly on the
  inference tail, and the pipeline creates no reference cycles to collect.
- Mel and pitch preprocessing are reimplemented on NumPy to keep librosa and torch out
  of the runtime; `tests/test_preprocess.py` pins them against the originals.
- Vendored upstream RVC code under `third_party/rvc` is kept verbatim (MIT).
