"""Paths and argument defaults shared by the offline tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RVC_MODELS = MODELS / "rvc"
ONNX_DIR = RVC_MODELS / "onnx"
VOICE_DIR = ROOT / "voice"
VENDOR = ROOT / "third_party"

# The generator checkpoint can only be unpickled with the upstream class definitions.
CONTENT_VEC = RVC_MODELS / "content-vec-best"


def add_model_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", type=Path, default=MODELS, help="model root directory (default: ./models)"
    )


def onnx_dir(models_root: Path) -> Path:
    return models_root / "rvc" / "onnx"


# The vendored files import each other by their upstream path, `infer.module.*`, because
# they are kept byte-for-byte as published. Import order matters: each module is aliased
# before the next one that imports it is loaded.
_VENDORED_SUBMODULES = ("commons", "transforms", "modules", "attentions")


def enable_vendored_rvc() -> None:
    """Make `import rvc.models` work by presenting the vendored code under `infer.module`.

    Patching the imports in third_party would be simpler and wrong: fetch_models
    re-downloads those files verbatim, so the patch disappears on the next fetch and the
    export breaks again with no obvious cause. Aliasing here survives re-fetching.
    """
    path = str(VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    if "infer.module" in sys.modules:
        return

    import importlib
    import types

    import rvc

    infer = types.ModuleType("infer")
    infer.__path__ = []  # declaring it a package lets `infer.module` resolve
    sys.modules["infer"] = infer
    sys.modules["infer.module"] = rvc
    infer.module = rvc

    for name in _VENDORED_SUBMODULES:
        submodule = importlib.import_module(f"rvc.{name}")
        sys.modules[f"infer.module.{name}"] = submodule
        setattr(rvc, name, submodule)
