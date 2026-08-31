"""Report what is inside a trained RVC checkpoint.

    python -m tools.inspect_voice voice/my_voice.pth

An RVC training run stores its sample rate, pitch mode, version and model config
alongside the weights, so the pipeline settings can be read out of the file rather than
guessed. Run this before exporting: a v1 model or an f0=0 model needs a different
pipeline than the one in this repo, and this is where that shows up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import VOICE_DIR

CONFIG_FIELDS = [
    "spec_channels",
    "segment_size",
    "inter_channels",
    "hidden_channels",
    "filter_channels",
    "n_heads",
    "n_layers",
    "kernel_size",
    "p_dropout",
    "resblock",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "upsample_rates",
    "upsample_initial_channel",
    "upsample_kernel_sizes",
    "spk_embed_dim",
    "gin_channels",
    "sr",
]


def describe(path: Path) -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"reading a checkpoint needs the 'tools' extra ({exc.name} is missing). "
            "Install it with: uv sync --extra tools"
        ) from exc

    print(f"file: {path}  ({path.stat().st_size / 1e6:.1f} MB)\n")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        print(f"unexpected format: {type(checkpoint)}")
        return 1

    print(f"top-level keys: {sorted(checkpoint.keys())}\n")
    for key in ("sr", "f0", "version", "info"):
        if key in checkpoint:
            print(f"  {key:<10} {str(checkpoint[key])[:80]}")

    config = checkpoint.get("config")
    if config is not None:
        print(f"\nconfig ({len(config)} positional arguments):")
        for name, value in zip(CONFIG_FIELDS, config, strict=False):
            print(f"  {name:<26} {value}")

    weights = checkpoint.get("weight") or checkpoint.get("model")
    if weights is None:
        print("\nno weights found under 'weight' or 'model'.")
        return 1
    print(f"\n{len(weights)} weight tensors")

    # The phone embedding's input width separates v1 (256) from v2 (768); they need
    # different content encoders, so this decides whether the model fits this pipeline.
    embedding = weights.get("enc_p.emb_phone.weight")
    if embedding is not None:
        dim = embedding.shape[1]
        kind = "v2 ContentVec 768" if dim == 768 else f"v1 {dim}"
        print(f"  enc_p.emb_phone.weight: {tuple(embedding.shape)}  -> feature dim {dim} ({kind})")
    if "emb_g.weight" in weights:
        print(f"  emb_g (speakers): {tuple(weights['emb_g.weight'].shape)}")

    total = sum(v.numel() for v in weights.values() if hasattr(v, "numel"))
    print(f"  {total / 1e6:.1f}M parameters")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", type=Path, nargs="?", default=VOICE_DIR / "my_voice.pth", help="checkpoint to read"
    )
    args = parser.parse_args(argv)
    if not args.path.exists():
        print(f"{args.path} does not exist.")
        return 2
    return describe(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
