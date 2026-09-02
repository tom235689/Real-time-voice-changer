"""The real-time engine.

Three threads:
  1) audio callback -- writes the input ring, reads the output ring. memcpy only.
  2) inference worker -- peeks a window per chunk, converts, crossfades, publishes.
  3) whatever drives the UI -- polls snapshot(); never touches engine internals.

Crossfade geometry:
    The window [context | chunk C | fade F] ends at pos + C.
    Its trailing C + F samples correspond to input [pos - F, pos + C).
    The leading F are crossfaded against the previous pass, so each pass emits exactly
    C frames and the seam between passes is inaudible.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig, EngineConfig, RuntimeParams
from .convert.base import Converter
from .ring import InputRing, OutputRing, Overrun

# Warmup sees few samples, so its worst observed inference time understates the real
# tail. Scale it before turning it into a prefill margin.
PREFILL_MARGIN = 1.5

# Inference times are kept in a fixed ring: a long meeting must not grow a list without
# bound, and a UI polling percentiles wants recent behaviour, not the whole session.
_HISTORY = 4096

# Pausing the collector is process-wide, so engines have to share it. Stopping one while
# another still runs must not hand the second a collector it was promised would be off.
_gc_lock = threading.Lock()
_gc_users = 0
_gc_was_enabled = True


def _pause_gc() -> None:
    """Stop the collector for as long as any engine is running.

    A collection pause lands directly on the inference tail, and the pipeline creates no
    reference cycles of its own, so refcounting alone reclaims what it allocates.
    """
    global _gc_users, _gc_was_enabled
    with _gc_lock:
        if _gc_users == 0:
            _gc_was_enabled = gc.isenabled()
            gc.collect()
            gc.freeze()
            gc.disable()
        _gc_users += 1


def _resume_gc() -> None:
    global _gc_users
    with _gc_lock:
        _gc_users = max(_gc_users - 1, 0)
        if _gc_users == 0:
            if _gc_was_enabled:
                gc.enable()
            gc.unfreeze()


@dataclass
class Telemetry:
    """A consistent read of engine state. Safe to hand to a UI thread."""

    blocks: int = 0
    chunks: int = 0
    underruns: int = 0
    startup_underruns: int = 0
    overruns: int = 0
    drops: int = 0
    grows: int = 0
    vad_skips: int = 0
    infer_p50_ms: float = 0.0
    infer_p95_ms: float = 0.0
    infer_max_ms: float = 0.0
    budget_ms: float = 0.0
    prefill_ms: float = 0.0
    device_latency_ms: float = 0.0
    worker_error: str | None = None
    """Set when the inference worker stopped on an exception. Audio has gone silent."""

    @property
    def failed(self) -> bool:
        return self.worker_error is not None

    @property
    def headroom(self) -> float:
        """p95 inference time over the per-chunk budget. Above 1.0 leaves no slack."""
        return self.infer_p95_ms / self.budget_ms if self.budget_ms else 0.0

    @property
    def verdict(self) -> str:
        """Whether this configuration can sustain real time.

        The median decides it, not p95. A chunk that overruns is repaid by the next one
        that comes in early, so the queue only diverges when the *typical* pass exceeds
        the budget. p95 over budget means no slack left for a competing load, which is a
        warning worth printing but not the same as failing.
        """
        if not self.budget_ms:
            return "unknown"
        if self.infer_p50_ms >= self.budget_ms:
            return "CANNOT KEEP UP"
        if self.infer_p95_ms >= self.budget_ms:
            return "no slack"
        if self.headroom >= 0.7:
            return "workable"
        return "comfortable"

    @property
    def total_latency_ms(self) -> float:
        return self.prefill_ms + self.device_latency_ms


class _Stats:
    """Counters owned by the engine. Read through Engine.snapshot(), not directly."""

    def __init__(self) -> None:
        self.blocks = 0
        self.chunks = 0
        self.underruns = 0
        self.startup_underruns = 0
        self.overruns = 0
        self.drops = 0
        self.grows = 0
        self.vad_skips = 0
        self._infer = np.zeros(_HISTORY, dtype=np.float64)
        self._n = 0
        self.infer_max = 0.0
        self.worker_error: str | None = None

    def record_infer(self, ms: float) -> None:
        self._infer[self._n % _HISTORY] = ms
        self._n += 1
        if ms > self.infer_max:
            self.infer_max = ms

    def percentile(self, q: float) -> float:
        filled = min(self._n, _HISTORY)
        if not filled:
            return 0.0
        return float(np.percentile(self._infer[:filled], q))


class Engine:
    def __init__(
        self,
        converter: Converter,
        audio: AudioConfig | None = None,
        settings: EngineConfig | None = None,
        params: RuntimeParams | None = None,
    ) -> None:
        self.conv = converter
        self.audio = audio or AudioConfig()
        self.cfg = settings or EngineConfig()
        self.params = params or RuntimeParams()

        sr = self.audio.sample_rate
        self.sr = sr
        self.block = self.audio.block
        self.C = int(sr * self.cfg.chunk_ms / 1000)
        self.F = int(sr * self.cfg.fade_ms / 1000)
        self.ctx = int(sr * converter.context_ms / 1000)
        self.window = self.ctx + self.C + self.F

        self._auto_prefill = self.cfg.prefill_ms is None
        self.prefill = int(sr * (self.cfg.prefill_ms or 0.0) / 1000)

        # Adaptive prefill: every steady-state underrun buys a little more margin, until
        # underruns stop. Latency grows by exactly what was needed, and no further.
        self._grow_pending = 0
        self._grow_step = int(sr * 0.020)
        self._grow_cap = int(sr * 0.800)
        self._probe_ms = 0.0
        self._vad_left = self.params.vad_hang_chunks

        # Four windows of input, so the worker can slip a chunk or two without its
        # window being overwritten underneath it.
        self.rin = InputRing(max(self.window * 4, sr))
        self.rout = OutputRing(max(self.prefill + self.C * 8, sr * 2))
        self.rout.prefill(self.prefill)
        # Seed the input ring with silence worth one context. Without it the worker
        # would have to wait a full window before its first chunk, and the output would
        # starve for exactly that long.
        self.rin.write(np.zeros(self.ctx + self.F, dtype=np.float32))

        # Pre-allocated. The worker loop must not allocate.
        self._win = np.zeros(self.window, dtype=np.float32)
        self._emit = np.zeros(self.C, dtype=np.float32)
        self._prev_tail = np.zeros(self.F, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.F, dtype=np.float32)
        self._fade_in = np.sin(t * np.pi / 2) ** 2  # equal-power pair
        self._fade_out = np.cos(t * np.pi / 2) ** 2

        self.stats = _Stats()
        self.device_latency_ms = 0.0
        self._pos = self.ctx + self.F  # start of the next chunk to emit
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._gc_paused = False

    # ------------------------------------------------------------------ properties
    @property
    def chunk_budget_ms(self) -> float:
        """Wall-clock time one inference pass may take before the output starves."""
        return self.C * 1000.0 / self.sr

    @property
    def prefill_ms(self) -> float:
        return self.prefill * 1000.0 / self.sr

    @property
    def window_ms(self) -> float:
        return self.window * 1000.0 / self.sr

    # ------------------------------------------------------------------ audio callback
    def _count_underrun(self) -> None:
        """Underruns before the first converted chunk are startup artefacts, counted apart.

        A steady-state underrun means the prefill margin is too small. The callback is a
        consumer, so it cannot push silence itself; it records a request and returns.
        """
        if self.stats.chunks == 0:
            self.stats.startup_underruns += 1
        else:
            self.stats.underruns += 1
            if self.cfg.adaptive_prefill:
                self._grow_pending += 1

    def callback(self, indata, outdata, frames, _time, status) -> None:
        if status and status.output_underflow:
            self._count_underrun()
        self.rin.write(indata[:, 0])
        if not self.rout.read_into(outdata[:, 0]):
            self._count_underrun()
        self.stats.blocks += 1

    # ------------------------------------------------------------------ inference worker
    def _grow_prefill(self) -> None:
        if not self._grow_pending or self.prefill >= self._grow_cap:
            return
        # One underrun means one block of output was missing. Growing by the actual
        # shortfall converges in a pass or two; a fixed step crawls.
        want = max(self._grow_pending * self.block, self._grow_step)
        self._grow_pending = 0
        step = min(want, self._grow_cap - self.prefill)
        self.rout.prefill(step)
        self.prefill += step
        self.stats.grows += 1

    def _emit_silence(self) -> None:
        """Fade the previous tail out into silence, so gating does not click."""
        np.multiply(self._prev_tail, self._fade_out, out=self._emit[: self.F])
        self._emit[self.F :] = 0.0
        self._prev_tail[:] = 0.0

    def _gated(self) -> bool:
        """True when the chunk sits below the VAD gate and inference should be skipped."""
        vad_db = self.params.vad_db
        if vad_db is None:
            return False
        segment = self._win[self.ctx :]
        rms = float(np.sqrt(np.mean(segment * segment)))
        if rms >= 10.0 ** (vad_db / 20.0):
            self._vad_left = self.params.vad_hang_chunks
        else:
            self._vad_left = max(self._vad_left - 1, -1)
        return self._vad_left < 0

    def _run(self) -> None:
        """Wrapper that makes a dying worker visible.

        Without this the thread just disappears: the audio callback keeps draining the
        output ring, so the sound stops while every counter still looks healthy and the
        UI goes on saying "running". Recording the reason is what lets a caller say
        something useful instead.
        """
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001 -- reported through telemetry, not swallowed
            self.stats.worker_error = f"{type(exc).__name__}: {exc}"

    def _loop(self) -> None:
        C, F = self.C, self.F
        while not self._stop.is_set():
            self._grow_prefill()
            if self.rin.written < self._pos + C:
                time.sleep(0.001)
                continue
            try:
                self.rin.read_window(self._pos + C, self.window, self._win)
            except Overrun:
                self.stats.overruns += 1
                self._pos = self.rin.written - C  # resynchronise to the live edge
                continue

            if self._gated():
                self._emit_silence()
                self.stats.vad_skips += 1
            else:
                if self.params.bypass:
                    tail = self._win[-(C + F) :]
                else:
                    t0 = time.perf_counter()
                    tail = self.conv.process(self._win, C + F)
                    self.stats.record_infer((time.perf_counter() - t0) * 1000.0)

                # Crossfade the leading F against the previous pass, keep the rest, and
                # stash this pass trailing F for the next one.
                np.multiply(self._prev_tail, self._fade_out, out=self._emit[:F])
                self._emit[:F] += tail[:F] * self._fade_in
                self._emit[F:] = tail[F:C]
                self._prev_tail[:] = tail[C:]

            # Applied to both paths. Gating still emits a fade-out of the previous tail,
            # so scaling only the converted branch would leak audio past a muted output.
            gain = self.params.output_gain
            if gain != 1.0:
                self._emit *= gain

            if not self.rout.write(self._emit):
                self.stats.drops += 1
            self._pos += C
            self.stats.chunks += 1

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Measure inference time on silence, size the prefill, then start the worker."""
        probe: list[float] = []
        for _ in range(6):
            t0 = time.perf_counter()
            self.conv.process(np.zeros(self.window, dtype=np.float32), self.C + self.F)
            probe.append((time.perf_counter() - t0) * 1000.0)
        self._probe_ms = max(probe[2:])  # the first two include lazy graph compilation

        if self._auto_prefill:
            per_ms = self.sr / 1000.0
            need = self.chunk_budget_ms + self._probe_ms * PREFILL_MARGIN
            self.prefill = int(need * per_ms)
            self.rout = OutputRing(max(self.prefill + self.C * 8, self.sr * 2))
            self.rout.prefill(self.prefill)

        _pause_gc()
        self._gc_paused = True

        self._stop.clear()
        self.stats.worker_error = None
        self._worker = threading.Thread(target=self._run, name="rtvc-infer", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._gc_paused:
            _resume_gc()
            self._gc_paused = False

    # ------------------------------------------------------------------ observation
    def snapshot(self) -> Telemetry:
        s = self.stats
        return Telemetry(
            blocks=s.blocks,
            chunks=s.chunks,
            underruns=s.underruns,
            startup_underruns=s.startup_underruns,
            overruns=s.overruns,
            drops=s.drops,
            grows=s.grows,
            vad_skips=s.vad_skips,
            infer_p50_ms=s.percentile(50),
            infer_p95_ms=s.percentile(95),
            infer_max_ms=s.infer_max or self._probe_ms,
            budget_ms=self.chunk_budget_ms,
            prefill_ms=self.prefill_ms,
            device_latency_ms=self.device_latency_ms,
            worker_error=s.worker_error,
        )

    def latency_budget_ms(self) -> dict[str, float]:
        """A sample entering at t leaves at t + prefill: prefill is the processing latency.

        Chunk size and inference time do not add to latency; they set the floor below
        which prefill cannot go:  prefill >= chunk + worst-case inference.
        """
        t = self.snapshot()
        floor = t.budget_ms + t.infer_max_ms
        return {
            "device stream": t.device_latency_ms,
            "prefill (processing)": t.prefill_ms,
            "total": t.total_latency_ms,
            "required floor (chunk + max infer)": floor,
            "margin": t.prefill_ms - floor,
        }

    def report(self, elapsed: float) -> str:
        t = self.snapshot()
        floor = t.budget_ms + t.infer_max_ms
        lines = []
        if t.failed:
            lines.append(f"  ENGINE STOPPED: {t.worker_error}  (output is silent)")
        return "\n".join(
            lines
            + [
                f"  chunk {self.chunk_budget_ms:.0f}ms  context {self.ctx * 1000 / self.sr:.0f}ms  "
                f"fade {self.F * 1000 / self.sr:.0f}ms  window {self.window_ms:.0f}ms",
                f"  elapsed {elapsed:.1f}s   blocks {t.blocks}   chunks {t.chunks}",
                f"  inference  p50 {t.infer_p50_ms:.1f}ms   p95 {t.infer_p95_ms:.1f}ms   "
                f"max {t.infer_max_ms:.1f}ms   (budget {t.budget_ms:.0f}ms)",
                f"  underruns {t.underruns} (steady)   startup {t.startup_underruns}   "
                f"overruns {t.overruns}   drops {t.drops}   prefill grew {t.grows}x   "
                f"vad skipped {t.vad_skips}/{t.chunks}",
                f"  p95 uses {t.headroom * 100:.0f}% of budget, median "
                f"{t.infer_p50_ms / t.budget_ms * 100:.0f}%  ->  {t.verdict}",
                f"  prefill {t.prefill_ms:.0f}ms  (floor {floor:.0f}ms = chunk + max infer)"
                f"  ->  processing latency {t.prefill_ms:.0f}ms",
            ]
        )


class Simulator:
    """Drives the engine at real-time pace with no audio device attached.

    The timing and glitch behaviour is identical to a device run, so the engine can be
    validated before a virtual cable is installed.
    """

    def __init__(self, engine: Engine) -> None:
        self.e = engine

    def run(self, seconds: float, report_every: float = 5.0) -> None:
        e = self.e
        blk = e.block
        # Two seconds of speech-level noise alternating with two of near-silence, so the
        # VAD gate is exercised rather than merely configured.
        rng = np.random.default_rng(1)
        talk = (rng.standard_normal(e.sr * 2) * 0.05).astype(np.float32)
        hush = (rng.standard_normal(e.sr * 2) * 1e-5).astype(np.float32)
        src = np.concatenate([talk, hush, talk, hush])
        sink = np.zeros(blk, dtype=np.float32)

        e.start()
        t0 = time.perf_counter()
        next_block = t0
        next_report = t0 + report_every
        i = 0
        try:
            while True:
                now = time.perf_counter()
                if now >= t0 + seconds:
                    break
                if now < next_block:
                    time.sleep(min(0.0005, next_block - now))
                    continue
                s = (i * blk) % (src.shape[0] - blk)
                e.rin.write(src[s : s + blk])
                if not e.rout.read_into(sink):
                    e._count_underrun()
                e.stats.blocks += 1
                i += 1
                next_block = t0 + i * blk / e.sr
                if time.perf_counter() >= next_report:
                    print(f"\n[{time.perf_counter() - t0:5.1f}s]")
                    print(e.report(time.perf_counter() - t0))
                    next_report += report_every
        finally:
            e.stop()
