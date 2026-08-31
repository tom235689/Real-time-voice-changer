"""Inference backends.

Two runtimes are used side by side because neither wins at every stage on this
hardware: OpenVINO is far faster on the int8 encoder, while the int8 generator only
produces correct output under ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _session_options(threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # With several sessions resident, idle worker threads spin-wait and burn cores that
    # the next stage needs. Disabling the spin is worth a 2.4x difference across the
    # three-session pipeline; it is not a micro-optimisation.
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


class OrtRunner:
    """ONNX Runtime CPU session with a single input tensor."""

    def __init__(self, path: Path, threads: int, provider: str = "CPUExecutionProvider") -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), _session_options(threads), providers=[provider])
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.in_name: x})[0]

    def close(self) -> None:
        self.sess = None


class OvRunner:
    """OpenVINO CPU session with a single input tensor.

    The engine's window and tail are fixed for the life of a run, so the model is
    compiled statically for the first shape seen. A shape change triggers one
    recompilation, which in practice happens during warmup and never again.
    """

    _core = None

    def __init__(self, path: Path, threads: int) -> None:
        import openvino as ov

        if OvRunner._core is None:
            OvRunner._core = ov.Core()
        self._ov = ov
        self.path = path
        self.threads = threads
        self._req = None
        self._out = None
        self._shape: tuple[int, ...] | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._req is None or x.shape != self._shape:
            model = OvRunner._core.read_model(self.path)
            model.reshape({model.inputs[0]: self._ov.PartialShape(list(x.shape))})
            compiled = OvRunner._core.compile_model(
                model,
                "CPU",
                {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": self.threads},
            )
            self._req = compiled.create_infer_request()
            self._out = compiled.outputs[0]
            self._shape = x.shape
        return self._req.infer({0: np.ascontiguousarray(x)})[self._out]

    def close(self) -> None:
        self._req = None


class GeneratorRunner:
    """The RVC generator takes five inputs, so the single-input runners do not fit."""

    def __init__(self, path: Path, threads: int, backend: str = "ort") -> None:
        self.backend = backend
        self.path = path
        self.threads = threads
        self._sid = np.array([0], dtype=np.int64)
        if backend == "ov":
            import openvino as ov

            self.core = ov.Core()
            self.req = None
            self.out = None
        else:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(
                str(path), _session_options(threads), providers=["CPUExecutionProvider"]
            )

    def __call__(
        self, phone: np.ndarray, lengths: np.ndarray, pitch: np.ndarray, nsff0: np.ndarray
    ) -> np.ndarray:
        if self.backend != "ov":
            return self.sess.run(
                None,
                {
                    "phone": phone,
                    "phone_lengths": lengths,
                    "pitch": pitch,
                    "nsff0": nsff0,
                    "sid": self._sid,
                },
            )[0]
        if self.req is None:
            compiled = self.core.compile_model(
                self.core.read_model(self.path),
                "CPU",
                {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": self.threads},
            )
            self.req = compiled.create_infer_request()
            self.out = compiled.outputs[0]
        return self.req.infer([phone, lengths, pitch, nsff0, self._sid])[self.out]

    def close(self) -> None:
        self.sess = None
        self.req = None


def make_runner(path: Path, threads: int, backend: str) -> OrtRunner | OvRunner:
    if backend == "ov":
        return OvRunner(path, threads)
    return OrtRunner(path, threads)
