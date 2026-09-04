"""Command line entry point.

    rtvc doctor                        check everything needed before converting
    rtvc devices                       list audio devices, flag virtual cables
    rtvc simulate --seconds 45         run at real-time pace with no device attached
    rtvc convert --in a.wav --out b.wav   offline conversion, for judging quality by ear
    rtvc run --in 17 --out 24          live conversion into a virtual cable
    rtvc gui                           desktop control panel
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

from . import __version__
from .config import DEFAULT_BACKEND, AudioConfig, Config, EngineConfig, ModelConfig, RuntimeParams
from .constants import FRAME_HZ

ROOT = Path(__file__).resolve().parents[2]


def parse_backend(spec: str) -> dict[str, str]:
    """Accept either 'ort' for every stage or 'enc=ov,pit=ort,gen=ort' per stage."""
    if "=" in spec:
        return dict(kv.split("=", 1) for kv in spec.split(","))
    return {stage: spec for stage in ("enc", "pit", "gen")}


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        audio=AudioConfig(
            sample_rate=args.rate,
            block=args.block,
            input_device=getattr(args, "in_dev", None),
            output_device=getattr(args, "out_dev", None),
        ),
        engine=EngineConfig(
            chunk_ms=args.chunk,
            fade_ms=args.fade,
            context_ms=args.context,
            prefill_ms=args.prefill,
        ),
        model=ModelConfig(
            root=Path(args.models),
            voice=args.voice,
            int8_encoder=not args.enc_fp32,
            int8_generator=not args.gen_fp32,
            variant=args.variant,
            threads=args.threads,
            backend=parse_backend(args.backend),
        ),
        params=RuntimeParams(
            key_shift=args.key,
            vad_db=args.vad_db,
            output_gain=args.gain,
        ),
    )


def describe(cfg: Config, kind: str) -> str:
    generator = f"int8{cfg.model.variant}" if cfg.model.int8_generator else "fp32"
    encoder = "int8" if cfg.model.int8_encoder else "fp32"
    backend = ",".join(f"{k}={v}" for k, v in cfg.model.backend.items())
    return "\n".join(
        [
            f"converter {kind}   voice {cfg.model.voice}   "
            f"generator {generator}   encoder {encoder}",
            f"backend {backend}   threads {cfg.model.threads}   key {cfg.params.key_shift:+.0f} st",
            f"chunk {cfg.engine.chunk_ms:.0f}ms   context {cfg.engine.context_ms:.0f}ms   "
            f"fade {cfg.engine.fade_ms:.0f}ms   generator frames {cfg.generator_frames}",
            "prefill " + ("auto" if cfg.engine.prefill_ms is None else f"{cfg.engine.prefill_ms:.0f}ms"),
        ]
    )


# ---------------------------------------------------------------------- subcommands
def cmd_devices(args: argparse.Namespace) -> int:
    from .devices import find_cable_output, format_table, list_devices

    print(format_table(list_devices()))
    cable = find_cable_output()
    if cable:
        print(f"\nFound a virtual cable: use --out {cable.index}  ({cable.name})")
    else:
        print("\nNo virtual cable found. Install VB-CABLE to feed a meeting app.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import format_report, run_checks, worst_status

    checks = run_checks(config_from_args(args))
    print(format_report(checks))
    return worst_status(checks)


def cmd_simulate(args: argparse.Namespace) -> int:
    from .session import Session

    cfg = config_from_args(args)
    print(describe(cfg, args.converter))
    print(f"\nSimulating {args.seconds:.0f}s at real-time pace, no audio device.\n")

    with Session(cfg, args.converter) as session:
        t0 = time.perf_counter()
        session.simulate(args.seconds)
        print("\n" + "=" * 74)
        print(session.engine.report(time.perf_counter() - t0))
        print("=" * 74)
        print("\nlatency breakdown (device latency excluded):")
        for name, value in session.engine.latency_budget_ms().items():
            print(f"  {name:<36} {value:>8.1f}ms")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from . import wavio
    from .offline import convert_file_audio, summarise
    from .session import build_converter

    cfg = config_from_args(args)

    source = Path(args.infile)
    if not source.exists():
        print(f"{source} does not exist.", file=sys.stderr)
        return 2
    try:
        audio, rate = wavio.read(source)
    except (OSError, ValueError, wave.Error) as exc:
        print(f"cannot read {source}: {exc}", file=sys.stderr)
        return 2

    print(describe(cfg, args.converter))
    print(f"\ninput {args.infile}   {audio.shape[0] / rate:.1f}s at {rate} Hz\n")

    converter = build_converter(cfg, args.converter)
    try:
        converted, timings = convert_file_audio(converter, audio, rate, cfg, progress=True)
    finally:
        converter.close()

    wavio.write(Path(args.outfile), converted, rate)
    print(f"\nwrote {args.outfile}")
    print(summarise(timings, cfg.engine.chunk_ms))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .session import Session

    cfg = config_from_args(args)
    if cfg.audio.input_device is None or cfg.audio.output_device is None:
        print("--in and --out are required; run 'rtvc devices' to find the indices.", file=sys.stderr)
        return 2

    print(describe(cfg, args.converter))
    with Session(cfg, args.converter) as session:
        session.start()
        print(f"\ndevice latency {session.engine.device_latency_ms:.1f}ms.  Ctrl+C to stop.\n")
        t0 = time.perf_counter()
        try:
            while True:
                time.sleep(5.0)
                print(f"\n[{time.perf_counter() - t0:5.1f}s]")
                print(session.engine.report(time.perf_counter() - t0))
        except KeyboardInterrupt:
            print("\nstopping.")
        print("\nlatency breakdown:")
        for name, value in session.engine.latency_budget_ms().items():
            print(f"  {name:<36} {value:>8.1f}ms")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    # Probe the dependency directly. Guarding `from .gui.app import run` catches nothing,
    # because app.py imports PySide6 inside run() and so the import that can fail happens
    # after the guard; guarding the run() call instead would swallow real bugs.
    try:
        import PySide6.QtWidgets  # noqa: F401
    except ImportError as exc:
        print(
            f"The GUI needs the 'gui' extra ({exc.name} is missing). "
            "Install it with: uv sync --extra gui",
            file=sys.stderr,
        )
        return 2

    from .gui.app import run

    return run(config_from_args(args))


# ---------------------------------------------------------------------- parser
def add_common(p: argparse.ArgumentParser) -> None:
    model = p.add_argument_group("model")
    model.add_argument("--models", default=str(ROOT / "models"), help="model root directory")
    model.add_argument("--voice", default="my_voice", help="voice model name")
    model.add_argument(
        "--gen-fp32",
        dest="gen_fp32",
        action="store_true",
        help="fp32 generator: better quality, slower, still within budget",
    )
    model.add_argument(
        "--enc-fp32",
        dest="enc_fp32",
        action="store_true",
        help="fp32 encoder: measured over budget on this hardware, for comparison only",
    )
    model.add_argument(
        "--variant",
        default="_qdqc",
        choices=["_qdq", "_qdqc"],
        help="int8 generator suffix; _qdqc is calibrated on real speech",
    )
    model.add_argument("--threads", type=int, default=12)
    model.add_argument(
        "--backend",
        default=",".join(f"{k}={v}" for k, v in DEFAULT_BACKEND.items()),
        help="inference backend, e.g. 'ort' or 'enc=ov,pit=ort,gen=ort'",
    )

    engine = p.add_argument_group("engine")
    engine.add_argument(
        "--chunk",
        type=float,
        default=200.0,
        help="ms of audio per inference pass; a generator export must exist for it",
    )
    engine.add_argument("--fade", type=float, default=20.0, help="crossfade ms")
    engine.add_argument("--context", type=float, default=500.0, help="encoder context ms")
    engine.add_argument("--prefill", type=float, default=None, help="output prefill ms (default: auto)")
    engine.add_argument("--rate", type=int, default=48000)
    engine.add_argument("--block", type=int, default=480)

    voice = p.add_argument_group("voice")
    voice.add_argument("--key", type=float, default=0.0, help="pitch shift in semitones")
    voice.add_argument("--gain", type=float, default=1.0, help="output gain")
    voice.add_argument(
        "--vad-db",
        dest="vad_db",
        type=float,
        default=None,
        help="skip inference below this dBFS, e.g. -45 (default: off)",
    )
    voice.add_argument(
        "--converter",
        default="rvc",
        choices=["rvc", "passthrough"],
        help="passthrough verifies the audio path without the model",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtvc", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"rtvc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    doctor = sub.add_parser("doctor", help="check everything needed before converting")
    add_common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    sim = sub.add_parser("simulate", help="run at real-time pace with no audio device")
    sim.add_argument("--seconds", type=float, default=45.0)
    add_common(sim)
    sim.set_defaults(func=cmd_simulate)

    conv = sub.add_parser("convert", help="convert a wav file offline")
    conv.add_argument("--in", dest="infile", required=True)
    conv.add_argument("--out", dest="outfile", required=True)
    add_common(conv)
    conv.set_defaults(func=cmd_convert)

    run = sub.add_parser("run", help="live conversion through an audio device")
    run.add_argument("--in", dest="in_dev", type=int, help="input device index")
    run.add_argument("--out", dest="out_dev", type=int, help="output device index")
    add_common(run)
    run.set_defaults(func=cmd_run)

    gui = sub.add_parser("gui", help="desktop control panel")
    add_common(gui)
    gui.set_defaults(func=cmd_gui)

    return p


def validate(args: argparse.Namespace) -> str | None:
    """Reject settings that cannot work, naming the reason.

    Left unchecked these surface far from their cause: a chunk of zero looks for a
    generator whose frame count is nonsense, and a fade longer than the chunk breaks the
    crossfade arithmetic inside the worker thread rather than at the command line.
    """
    chunk = getattr(args, "chunk", None)
    if chunk is None:
        return None

    step = 1000.0 / FRAME_HZ
    if chunk <= 0:
        return "--chunk must be greater than 0"
    if abs(chunk / step - round(chunk / step)) > 1e-6:
        # A generator ONNX is bound to one frame count, so an off-grid chunk quietly
        # resolves to a file that was built for a different length.
        return f"--chunk must be a multiple of {step:.0f}ms"

    fade = getattr(args, "fade", 0.0)
    if fade < 0:
        return "--fade cannot be negative"
    if fade >= chunk:
        return f"--fade ({fade:.0f}ms) must be shorter than --chunk ({chunk:.0f}ms)"
    if getattr(args, "context", 0.0) < 0:
        return "--context cannot be negative"
    if getattr(args, "threads", 1) < 1:
        return "--threads must be at least 1"
    if getattr(args, "block", 1) < 1:
        return "--block must be at least 1"
    if getattr(args, "rate", 1) < 8000:
        return "--rate must be at least 8000"
    if getattr(args, "gain", 0.0) < 0:
        return "--gain cannot be negative"
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = validate(args)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
