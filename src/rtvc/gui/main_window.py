"""The control panel window.

Two groups of controls, and the split is the same one Config makes:

    Setup    -- devices, chunk size, voice, precision. Locked while running, because
                each is baked into an allocated buffer or a compiled graph.
    Live     -- pitch, gain, VAD, bypass. Written straight into RuntimeParams, which
                the worker re-reads every chunk.

Starting a session loads several hundred megabytes of ONNX and compiles it, which takes
seconds. That runs on a worker thread; doing it inline would freeze the window and make
a working start look like a hang.
"""

from __future__ import annotations

import json
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..catalog import exported_voices
from ..config import Config, user_settings_path
from ..devices import list_devices
from ..session import Session

POLL_MS = 250
CHUNK_CHOICES = (100.0, 150.0, 200.0, 250.0)
METER_FLOOR_DB = -60


class MainWindow(QMainWindow):
    _session_ready = Signal(object)
    _session_failed = Signal(str)

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.session: Session | None = None
        self._starting = False
        self._closing = False

        self.setWindowTitle("Real-time Voice Changer")
        self.setMinimumWidth(520)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_setup())
        layout.addWidget(self._build_live())
        layout.addWidget(self._build_telemetry())
        layout.addLayout(self._build_actions())
        layout.addStretch(1)
        self.setCentralWidget(root)

        self._session_ready.connect(self._on_started)
        self._session_failed.connect(self._on_failed)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_telemetry)
        self._timer.start(POLL_MS)

        self._reload_devices()
        self._reload_voices()
        self._restore_ui_state()
        self._sync_from_config()

    # ------------------------------------------------------------------ construction
    def _build_setup(self) -> QGroupBox:
        box = QGroupBox("Setup (locked while running)")
        form = QFormLayout(box)

        self.cmb_input = QComboBox()
        self.cmb_output = QComboBox()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._reload_devices)

        out_row = QHBoxLayout()
        out_row.addWidget(self.cmb_output, 1)
        out_row.addWidget(btn_refresh)

        self.cmb_voice = QComboBox()
        self.cmb_voice.currentTextChanged.connect(lambda _: self._reload_chunks())

        self.cmb_chunk = QComboBox()
        self.cmb_precision = QComboBox()
        self.cmb_precision.addItem("int8, calibrated (fastest)", "_qdqc")
        self.cmb_precision.addItem("int8 (fast)", "_qdq")
        self.cmb_precision.addItem("fp32 (best quality, slowest)", "")
        self.cmb_precision.currentIndexChanged.connect(lambda _: self._reload_chunks())

        form.addRow("Microphone", self.cmb_input)
        form.addRow("Output (virtual cable)", out_row)
        form.addRow("Voice", self.cmb_voice)
        form.addRow("Chunk size", self.cmb_chunk)
        form.addRow("Generator precision", self.cmb_precision)
        self._setup_widgets = (
            self.cmb_input,
            self.cmb_output,
            self.cmb_voice,
            self.cmb_chunk,
            self.cmb_precision,
        )
        return box

    def _build_live(self) -> QGroupBox:
        box = QGroupBox("Live controls")
        form = QFormLayout(box)

        self.sld_key = QSlider(Qt.Horizontal)
        self.sld_key.setRange(-12, 12)
        self.lbl_key = QLabel("0 st")
        self.sld_key.valueChanged.connect(self._on_key)
        form.addRow("Pitch", self._with_label(self.sld_key, self.lbl_key))

        self.sld_gain = QSlider(Qt.Horizontal)
        self.sld_gain.setRange(0, 200)
        self.lbl_gain = QLabel("100%")
        self.sld_gain.valueChanged.connect(self._on_gain)
        form.addRow("Output gain", self._with_label(self.sld_gain, self.lbl_gain))

        self.chk_vad = QCheckBox("Skip inference below")
        self.sld_vad = QSlider(Qt.Horizontal)
        self.sld_vad.setRange(-70, -20)
        self.lbl_vad = QLabel("-45 dBFS")
        self.chk_vad.toggled.connect(self._on_vad)
        self.sld_vad.valueChanged.connect(self._on_vad)
        vad_row = QHBoxLayout()
        vad_row.addWidget(self.chk_vad)
        vad_row.addWidget(self.sld_vad, 1)
        vad_row.addWidget(self.lbl_vad)
        vad_wrap = QWidget()
        vad_wrap.setLayout(vad_row)
        form.addRow("Silence gate", vad_wrap)

        self.chk_bypass = QCheckBox("Send the microphone through unconverted")
        self.chk_bypass.toggled.connect(self._on_bypass)
        form.addRow("Bypass", self.chk_bypass)
        return box

    def _build_telemetry(self) -> QGroupBox:
        box = QGroupBox("Live measurements")
        grid = QGridLayout(box)

        self.lbl_state = QLabel("stopped")
        self.lbl_latency = QLabel("-")
        self.lbl_infer = QLabel("-")
        self.lbl_glitches = QLabel("-")
        self.bar_budget = QProgressBar()
        self.bar_budget.setRange(0, 100)
        self.bar_budget.setFormat("%p% of budget")

        # Meters answer the question people actually have -- "can they hear me?" -- and
        # separate a dead microphone from a dead model, which no other readout does.
        self.bar_input = self._meter()
        self.bar_output = self._meter()

        for row, (name, widget) in enumerate(
            [
                ("State", self.lbl_state),
                ("Input level", self.bar_input),
                ("Output level", self.bar_output),
                ("Latency", self.lbl_latency),
                ("Inference", self.lbl_infer),
                ("Glitches", self.lbl_glitches),
                ("Budget used", self.bar_budget),
            ]
        ):
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)
        return box

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._toggle)
        btn_load = QPushButton("Load preset")
        btn_save = QPushButton("Save preset")
        btn_load.clicked.connect(self._load_preset)
        btn_save.clicked.connect(self._save_preset)
        row.addWidget(self.btn_start, 1)
        row.addWidget(btn_load)
        row.addWidget(btn_save)
        return row

    @staticmethod
    def _meter() -> QProgressBar:
        """A dBFS meter. Scaled -60..0, which is where speech actually lives."""
        bar = QProgressBar()
        bar.setRange(METER_FLOOR_DB, 0)
        bar.setValue(METER_FLOOR_DB)
        bar.setFormat("%v dBFS")
        bar.setTextVisible(True)
        return bar

    @staticmethod
    def _with_label(slider: QSlider, label: QLabel) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(label)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    # ------------------------------------------------------------------ population
    def _reload_devices(self) -> None:
        try:
            devices = list_devices()
        except Exception as exc:  # a missing or busy audio backend must not kill the window
            self.lbl_state.setText(f"audio backend unavailable: {exc}")
            return

        self.cmb_input.clear()
        self.cmb_output.clear()
        for d in devices:
            if d.is_input:
                self.cmb_input.addItem(f"[{d.index}] {d.name}  ({d.hostapi})", d.index)
            if d.is_output:
                mark = " *cable*" if d.is_virtual_cable else ""
                self.cmb_output.addItem(f"[{d.index}] {d.name}{mark}  ({d.hostapi})", d.index)

        self._select_data(self.cmb_input, self.cfg.audio.input_device)
        if self.cfg.audio.output_device is not None:
            self._select_data(self.cmb_output, self.cfg.audio.output_device)
        else:
            for i in range(self.cmb_output.count()):
                if "*cable*" in self.cmb_output.itemText(i):
                    self.cmb_output.setCurrentIndex(i)
                    break

    def _reload_voices(self) -> None:
        self._voices = exported_voices(self.cfg.model.onnx_dir)
        self.cmb_voice.clear()
        for name in self._voices:
            self.cmb_voice.addItem(name)
        if self.cfg.model.voice in self._voices:
            self.cmb_voice.setCurrentText(self.cfg.model.voice)
        if not self._voices:
            self.lbl_state.setText(f"no exported voices under {self.cfg.model.onnx_dir}")
        self._reload_chunks()

    def _reload_chunks(self) -> None:
        """Offer only chunk sizes that have a generator exported for the chosen variant."""
        self.cmb_chunk.clear()
        entry = self._voices.get(self.cmb_voice.currentText())
        if entry is None:
            return
        variant = self.cmb_precision.currentData()
        sizes = entry.chunk_sizes(self.cfg.engine.fade_ms, variant)
        for ms in sizes:
            if ms in CHUNK_CHOICES:
                self.cmb_chunk.addItem(f"{ms:.0f} ms", ms)
        self._select_data(self.cmb_chunk, self.cfg.engine.chunk_ms)
        if self.cmb_chunk.count() == 0:
            self.lbl_state.setText("no generator exported for this voice and precision")

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        if value is None:
            return
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _sync_from_config(self) -> None:
        p = self.cfg.params
        self.sld_key.setValue(int(p.key_shift))
        self.sld_gain.setValue(int(p.output_gain * 100))
        self.chk_vad.setChecked(p.vad_db is not None)
        self.sld_vad.setValue(int(p.vad_db) if p.vad_db is not None else -45)
        self.chk_bypass.setChecked(p.bypass)
        self._on_key()
        self._on_gain()
        self._on_vad()

    # ------------------------------------------------------------------ live controls
    def _on_key(self) -> None:
        value = float(self.sld_key.value())
        self.lbl_key.setText(f"{value:+.0f} st")
        self.cfg.params.key_shift = value

    def _on_gain(self) -> None:
        percent = self.sld_gain.value()
        self.lbl_gain.setText(f"{percent}%")
        self.cfg.params.output_gain = percent / 100.0

    def _on_vad(self) -> None:
        enabled = self.chk_vad.isChecked()
        self.sld_vad.setEnabled(enabled)
        self.lbl_vad.setText(f"{self.sld_vad.value()} dBFS")
        self.cfg.params.vad_db = float(self.sld_vad.value()) if enabled else None

    def _on_bypass(self, checked: bool) -> None:
        self.cfg.params.bypass = checked

    # ------------------------------------------------------------------ lifecycle
    def _collect_setup(self) -> bool:
        if self.cmb_input.currentData() is None or self.cmb_output.currentData() is None:
            QMessageBox.warning(self, "Devices", "Select an input and an output device.")
            return False
        if self.cmb_chunk.currentData() is None:
            QMessageBox.warning(self, "Model", "No generator is exported for this configuration.")
            return False
        self.cfg.audio.input_device = self.cmb_input.currentData()
        self.cfg.audio.output_device = self.cmb_output.currentData()
        self.cfg.model.voice = self.cmb_voice.currentText()
        self.cfg.engine.chunk_ms = self.cmb_chunk.currentData()
        variant = self.cmb_precision.currentData()
        # The combo governs the generator only. The encoder stays int8: fp32 there is
        # measured over the real-time budget, so it is not offered as a choice.
        self.cfg.model.int8_generator = variant != ""
        self.cfg.model.variant = variant or "_qdqc"
        return True

    def _toggle(self) -> None:
        if self.session is not None:
            self._stop()
        elif not self._starting:
            self._start()

    def _start(self) -> None:
        if not self._collect_setup():
            return
        self._starting = True
        self._set_setup_enabled(False)
        self.btn_start.setEnabled(False)
        self.lbl_state.setText("loading model...")

        def work() -> None:
            try:
                session = Session(self.cfg, "rvc")
                session.start()
            except Exception as exc:
                self._session_failed.emit(str(exc))
            else:
                self._session_ready.emit(session)

        threading.Thread(target=work, name="rtvc-start", daemon=True).start()

    def _on_started(self, session: Session) -> None:
        if self._closing:
            # The window was closed while the model was loading, which takes long enough
            # to be easy to do. Adopting the session now would leave an audio device and
            # a worker thread held by a window nobody can reach.
            session.close()
            return
        self.session = session
        self._starting = False
        self.btn_start.setText("Stop")
        self.btn_start.setEnabled(True)
        self.lbl_state.setText("running")
        # Remember a combination that actually started, not one that merely got picked.
        self._save_ui_state()

    def _on_failed(self, message: str) -> None:
        self._starting = False
        self._set_setup_enabled(True)
        self.btn_start.setEnabled(True)
        self.lbl_state.setText("failed to start")
        QMessageBox.critical(self, "Could not start", message)

    def _stop(self) -> None:
        session, self.session = self.session, None
        self.btn_start.setText("Start")
        self.lbl_state.setText("stopped")
        self._set_setup_enabled(True)
        if session is not None:
            session.close()

    def _set_setup_enabled(self, enabled: bool) -> None:
        for widget in self._setup_widgets:
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------ telemetry
    def _refresh_telemetry(self) -> None:
        if self.session is None:
            self.bar_budget.setValue(0)
            self.bar_input.setValue(METER_FLOOR_DB)
            self.bar_output.setValue(METER_FLOOR_DB)
            return
        t = self.session.snapshot()
        if t.failed:
            # The worker is gone and the output has gone silent. Saying "running" here
            # would be worse than useless, so tear the session down and say what broke.
            message = t.worker_error
            self._stop()
            self.lbl_state.setText(f"stopped after an error: {message}")
            QMessageBox.critical(self, "Conversion stopped", message)
            return
        self.lbl_latency.setText(
            f"{t.total_latency_ms:.0f} ms total   "
            f"({t.prefill_ms:.0f} processing + {t.device_latency_ms:.0f} device)"
        )
        self.lbl_infer.setText(
            f"p50 {t.infer_p50_ms:.0f} ms   p95 {t.infer_p95_ms:.0f} ms   "
            f"max {t.infer_max_ms:.0f} ms   budget {t.budget_ms:.0f} ms"
        )
        self.lbl_glitches.setText(
            f"underruns {t.underruns}   overruns {t.overruns}   drops {t.drops}   "
            f"gate skipped {t.vad_skips}/{t.chunks}"
        )
        self.bar_budget.setValue(min(int(t.headroom * 100), 100))
        self.bar_input.setValue(max(int(t.input_dbfs), METER_FLOOR_DB))
        self.bar_output.setValue(max(int(t.output_dbfs), METER_FLOOR_DB))

    # ------------------------------------------------------------------ remembered state
    def _ui_state(self) -> dict:
        """What the panel should remember between runs.

        Devices are recorded by name as well as index: indices shift as soon as a headset
        is plugged in or removed, so an index alone silently selects the wrong device.
        """
        params = self.cfg.params
        return {
            "input_device": self.cmb_input.currentData(),
            "input_name": self.cmb_input.currentText(),
            "output_device": self.cmb_output.currentData(),
            "output_name": self.cmb_output.currentText(),
            "voice": self.cmb_voice.currentText(),
            "chunk_ms": self.cmb_chunk.currentData(),
            "variant": self.cmb_precision.currentData(),
            "key_shift": params.key_shift,
            "vad_db": params.vad_db,
            "output_gain": params.output_gain,
        }

    def _save_ui_state(self) -> None:
        try:
            path = user_settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._ui_state(), indent=2), encoding="utf-8")
        except OSError:
            pass  # remembering preferences is a convenience, never a reason to fail

    def _restore_ui_state(self) -> None:
        try:
            path = user_settings_path()
            if not path.is_file():
                return
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # a corrupt settings file must not stop the panel from opening

        self._select_device(self.cmb_input, state.get("input_device"), state.get("input_name"))
        self._select_device(self.cmb_output, state.get("output_device"), state.get("output_name"))
        if state.get("voice") in self._voices:
            self.cmb_voice.setCurrentText(state["voice"])
        if state.get("variant") is not None:
            self._select_data(self.cmb_precision, state["variant"])
        self._reload_chunks()
        self._select_data(self.cmb_chunk, state.get("chunk_ms"))

        params = self.cfg.params
        params.key_shift = float(state.get("key_shift", params.key_shift))
        params.output_gain = float(state.get("output_gain", params.output_gain))
        params.vad_db = state.get("vad_db", params.vad_db)

    @staticmethod
    def _select_device(combo: QComboBox, index, name) -> None:
        """Match the remembered device by name first, then fall back to its old index."""
        if name:
            at = combo.findText(name)
            if at >= 0:
                combo.setCurrentIndex(at)
                return
        MainWindow._select_data(combo, index)

    # ------------------------------------------------------------------ presets
    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.json)")
        if not path:
            return
        try:
            loaded = Config.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if self.session is not None:
            # Setup fields are baked into the running session; apply only what is live.
            self.cfg.params = loaded.params
            self.session.engine.params = loaded.params
            self.session.converter.params = loaded.params
        else:
            self.cfg = loaded
            self._reload_devices()
            self._reload_voices()
        self._sync_from_config()

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "preset.json", "Preset (*.json)")
        if path:
            try:
                self.cfg.save(path)
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event) -> None:
        self._closing = True
        self._timer.stop()
        self._save_ui_state()
        self._stop()
        super().closeEvent(event)
