"""Convert PyTorch weights to the ONNX files the runtime loads.

    python -m tools.export_onnx --voice voice/my_voice.pth
    python -m tools.export_onnx --voice voice/my_voice.pth --skip-encoder --chunk 300

Produces:
    models/rvc/onnx/encoder_contentvec.onnx          ContentVec, last hidden layer
    models/rvc/onnx/generator_<voice>_f<frames>.onnx one file per chunk size

Only the inference path of the synthesiser is exported:

    phone(768) -> enc_p -> flow(reverse) -> dec (NSF HiFi-GAN)

The posterior encoder and the discriminator exist for training and never run during
conversion, so exporting them would only add weight to the graph.

The generator is exported once per chunk size on purpose. Passing dynamic_axes is
accepted without error, but NSF writes the sequence length into the graph as a
constant, so the result only works at the length it was traced with -- feeding 46
frames to a 41-frame export fails at runtime rather than at export. Since the engine
fixes its chunk size for the life of a run, and OpenVINO compiles statically anyway,
static exports are the honest representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rtvc.constants import generator_frames

from .paths import CONTENT_VEC, RVC_MODELS, add_model_root, enable_vendored_rvc, onnx_dir

OPSET = 17

# configs/v2/32k.json flattened into the positional order SynthesizerTrnMs768NSFsid
# expects. Used only when a checkpoint carries no config of its own, which is the case
# for the pretrained f0G32k weights.
#   spec_channels = filter_length // 2 + 1 = 513,  segment_size = 12800 // 320 = 40
CONFIG_V2_32K = [
    513, 40, 192, 192, 768, 2, 6, 3, 0,
    "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000,
]

DEFAULT_CHUNKS = [100.0, 150.0, 200.0, 250.0]


def require_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"export needs the 'tools' extra ({exc.name} is missing). "
            "Install it with: uv sync --extra tools"
        ) from exc


def export_encoder(out: Path) -> None:
    import torch
    import torch.nn as nn
    from transformers import HubertModel

    print(f"\n== encoder  {CONTENT_VEC.name}")
    model = HubertModel.from_pretrained(str(CONTENT_VEC)).eval()

    class Encoder(nn.Module):
        """RVC v2 reads the last hidden layer. Masking and dropout are off in eval."""

        def __init__(self, inner: HubertModel) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (1, T) at 16 kHz
            return self.inner(wav).last_hidden_state  # (1, F50, 768)

    wrapped = Encoder(model).eval()
    dummy = torch.randn(1, int(16000 * 0.72))
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(out / "encoder_contentvec.onnx"),
            input_names=["wav"],
            output_names=["feats"],
            # The encoder genuinely is length-agnostic, unlike the generator.
            dynamic_axes={"wav": {1: "T"}, "feats": {1: "F"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )
    params = sum(p.numel() for p in model.parameters())
    print(f"   {params / 1e6:.1f}M params -> encoder_contentvec.onnx")


def export_generator(checkpoint_path: Path, tag: str, out: Path, frame_counts: list[int]) -> None:
    import torch
    import torch.nn as nn

    enable_vendored_rvc()
    from rvc.models import SynthesizerTrnMs768NSFsid

    print(f"\n== generator  {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("sr", "f0", "version"):
        if key in checkpoint:
            print(f"   {key}: {checkpoint[key]}")

    version = checkpoint.get("version")
    if version not in (None, "v2"):
        raise SystemExit(
            f"only v2 models are supported (this one is {version}). v1 uses 256-dimensional "
            "encoder features and needs a different content encoder."
        )
    if checkpoint.get("f0") == 0:
        raise SystemExit("f0=0 (pitch-unconditioned) models are outside this pipeline.")

    config = checkpoint.get("config") or CONFIG_V2_32K
    net = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    state = checkpoint.get("weight") or checkpoint.get("model") or checkpoint
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"   load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    net.remove_weight_norm()

    class Generator(nn.Module):
        """The inference path only: phone -> enc_p -> flow(reverse) -> dec."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, phone, phone_lengths, pitch, nsff0, sid):
            g = self.inner.emb_g(sid).unsqueeze(-1)
            m_p, logs_p, x_mask = self.inner.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.inner.flow(z_p, x_mask, g=g, reverse=True)
            return self.inner.dec(z * x_mask, nsff0, g=g)

    wrapped = Generator(net).eval()
    names = ["phone", "phone_lengths", "pitch", "nsff0", "sid"]

    def sample_inputs(frames: int):
        return (
            torch.randn(1, frames, 768),
            torch.tensor([frames], dtype=torch.int64),
            torch.randint(0, 255, (1, frames), dtype=torch.int64),
            torch.rand(1, frames) * 200 + 100,
            torch.tensor([0], dtype=torch.int64),
        )

    out.mkdir(parents=True, exist_ok=True)
    for frames in frame_counts:
        dst = out / f"generator_{tag}_f{frames}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                sample_inputs(frames),
                str(dst),
                input_names=names,
                output_names=["audio"],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
        print(f"   frames={frames:3d} -> {dst.name}")

    vocoder = sum(p.numel() for p in net.dec.parameters())
    total = sum(p.numel() for p in wrapped.parameters())
    print(f"   vocoder {vocoder / 1e6:.1f}M   whole inference path {total / 1e6:.1f}M params")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--voice", type=Path, default=None, help="trained .pth; defaults to the pretrained f0G32k"
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="the encoder is voice-independent, so it only needs exporting once",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        nargs="+",
        default=DEFAULT_CHUNKS,
        help="chunk sizes in ms to export generators for",
    )
    parser.add_argument("--fade", type=float, default=20.0, help="crossfade ms the engine will use")
    add_model_root(parser)
    args = parser.parse_args(argv)
    require_torch()

    out = onnx_dir(args.models)
    frame_counts = sorted({generator_frames(ms, args.fade) for ms in args.chunk})
    print("chunk -> frames: " + ", ".join(
        f"{ms:.0f}ms->{generator_frames(ms, args.fade)}" for ms in args.chunk
    ))

    if not args.skip_encoder:
        if not CONTENT_VEC.exists():
            print(f"{CONTENT_VEC} is missing. Run: python -m tools.fetch_models")
            return 2
        export_encoder(out)

    checkpoint = args.voice or (RVC_MODELS / "f0G32k.pth")
    if not checkpoint.exists():
        print(f"{checkpoint} is missing. Run: python -m tools.fetch_models")
        return 2
    export_generator(checkpoint, checkpoint.stem, out, frame_counts)

    print(f"\nexported -> {out}")
    print("Next, quantise with real speech: python -m tools.quantize --audio <recording>.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
