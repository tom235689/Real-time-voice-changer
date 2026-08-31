"""Download the base assets and the vendored RVC source.

    python -m tools.fetch_models

About 780 MB in total:

  content-vec-best  361 MB  RVC v2 content encoder, ported so transformers can load it
                            without fairseq
  rmvpe.onnx        345 MB  pitch tracker, already ONNX so no conversion is needed
  f0G32k.pth         71 MB  RVC v2 32k pretrained generator
  RVC model code     83 KB  class definitions required to unpickle a checkpoint (MIT)

Files that already exist at the right size are skipped, so re-running is cheap and an
interrupted download resumes by re-fetching only what is incomplete.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from .paths import RVC_MODELS, VENDOR

HF = "https://huggingface.co"
GH = "https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main"

ASSETS: list[tuple[str, Path]] = [
    (
        f"{HF}/lengyue233/content-vec-best/resolve/main/config.json",
        RVC_MODELS / "content-vec-best" / "config.json",
    ),
    (
        f"{HF}/lengyue233/content-vec-best/resolve/main/pytorch_model.bin",
        RVC_MODELS / "content-vec-best" / "pytorch_model.bin",
    ),
    (f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx", RVC_MODELS / "rmvpe.onnx"),
    (
        f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth",
        RVC_MODELS / "f0G32k.pth",
    ),
]

# Class definitions needed to unpickle a generator checkpoint. MIT; provenance is
# recorded in third_party/rvc/README.md.
VENDORED = ["models.py", "modules.py", "attentions.py", "commons.py", "transforms.py"]

# Sources for the preprocessing constants (RMVPE mel parameters, cent decoding, coarse
# pitch). The runtime reimplements these on NumPy, but the originals are what
# tests/test_features.py and the reimplementation are checked against.
VENDORED_REFERENCE = [("infer/rmvpe.py", "rmvpe.py"), ("infer/vc/pipeline.py", "pipeline.py")]

VENDOR_README = """\
Copy of `infer/module/` from RVC-Project/Retrieval-based-Voice-Conversion-WebUI (MIT).

Vendored because loading a generator checkpoint (`f0G32k.pth`, and any trained voice
`.pth`) requires the original class definitions. Used by the offline export tools only;
nothing here runs at conversion time.

Kept verbatim, and excluded from lint in `pyproject.toml`.

Source: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
"""


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001 -- any failure just means "size unknown", not fatal
        return None


def fetch(url: str, dst: Path) -> None:
    want = remote_size(url)
    if dst.exists() and want and dst.stat().st_size == want:
        print(f"  skip     {dst.name}  ({want / 1e6:.1f} MB, already present)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Download to a .part file so an interrupted run never leaves a truncated model that
    # looks complete to the next one.
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  download {dst.name}  ({(want or 0) / 1e6:.1f} MB)", end="", flush=True)
    got = 0
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:  # noqa: S310
        while chunk := response.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if want:
                print(
                    f"\r  download {dst.name}  {got / 1e6:7.1f} / {want / 1e6:.1f} MB"
                    f"  {got * 100 // want:3d}%",
                    end="",
                    flush=True,
                )
    tmp.replace(dst)
    print(f"\r  done     {dst.name}  {got / 1e6:7.1f} MB{' ' * 20}")


def main(argv: list[str] | None = None) -> int:
    # argparse rather than a bare main(): without it --help starts a 780 MB download.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--code-only", action="store_true", help="fetch only the vendored RVC source, not the weights"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be fetched and what is already here"
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print("Base assets")
        for url, dst in ASSETS:
            want = remote_size(url)
            have = dst.exists() and want and dst.stat().st_size == want
            state = "present" if have else "would download"
            print(f"  {state:<15} {dst.name}  ({(want or 0) / 1e6:.1f} MB)")
        return 0

    if not args.code_only:
        print("Base assets")
        for url, dst in ASSETS:
            fetch(url, dst)

    print("\nVendored RVC model code (MIT)")
    vendor = VENDOR / "rvc"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "__init__.py").write_text("", encoding="utf-8")
    for name in VENDORED:
        fetch(f"{GH}/infer/module/{name}", vendor / name)
    for src, name in VENDORED_REFERENCE:
        fetch(f"{GH}/{src}", vendor / name)
    (vendor / "README.md").write_text(VENDOR_README, encoding="utf-8")

    print(f"\nready -> {RVC_MODELS}")
    print("Next: python -m tools.export_onnx")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
