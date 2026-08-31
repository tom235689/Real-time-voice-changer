"""Offline model preparation. None of this runs during conversion.

    python -m tools.fetch_models              base assets and the vendored RVC code
    python -m tools.inspect_voice             read sr / f0 / version out of a .pth
    python -m tools.export_onnx               PyTorch weights -> ONNX
    python -m tools.quantize --audio a.wav    ONNX -> int8

Requires the 'tools' extra:  uv sync --extra tools
"""
