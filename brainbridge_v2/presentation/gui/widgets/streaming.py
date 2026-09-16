import os
import math
from datetime import datetime
from typing import List
from collections import deque
import numpy as np
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QGroupBox, QComboBox, QGridLayout,
                           QMessageBox, QCheckBox,
                           QLineEdit, QSpinBox, QDialog, QInputDialog, QFrame,
                           QScrollArea, QSplitter, QSizePolicy, QLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer, QThreadPool, Qt
from brainbridge_v2.presentation.gui.inference_worker import (
    InferenceOutcome, InferenceWorker, RLUpdateOutcome, RLUpdateWorker)
from brainbridge_v2.application.game_inference_coordinator import (
    GameInferenceCoordinator,
)
from brainbridge_v2.application.free_run_coordinator import (
    FreeRunInferenceCoordinator,
)
from brainbridge_v2.application.eeg_quality import EEGWindowQualityValidator
from brainbridge_v2.application.pipeline_telemetry import PipelineTelemetry
from brainbridge_v2.application.runtime_config import DEFAULT_RUNTIME_CONFIG, get_runtime
from brainbridge_v2.application.unity_command_mapper import UnityCommandMapper
from brainbridge_v2.infrastructure.config.settings import get_recording_path
from brainbridge_v2.interface_adapters.controllers.eeg_stream_controller import (
    EEGStreamController,
)
from brainbridge_v2.interface_adapters.controllers.esp32_controller import (
    ESP32Controller,
)
from brainbridge_v2.interface_adapters.controllers.inference_controller import (
    InferenceController,
)
from brainbridge_v2.interface_adapters.controllers.marker_controller import (
    MarkerController,
)
from brainbridge_v2.interface_adapters.controllers.patient_controller import (
    PatientController,
)
from brainbridge_v2.interface_adapters.controllers.recording_controller import (
    RecordingController,
)
from brainbridge_v2.interface_adapters.controllers.session_controller import (
    SessionController,
)
from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)
from brainbridge_v2.interface_adapters.controllers.unity_controller import (
    UnityController,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    AccuracyPresenter,
    AccuracyTrialViewModel,
    ConnectionStatusPresenter,
    GameRuntimePresenter,
    MarkerStateViewModel,
    ModelViewModel,
    PredictionViewModel,
    ProgressionPresenter,
    SessionViewModel,
    StartRecordingRequest,
    StartSessionRequest,
    StreamingSessionStatePresenter,
    StreamingSessionStateViewModel,
    TaskViewStatePresenter,
)
from brainbridge_v2.presentation.gui.widgets.eeg_plot import EEGPlotWidget
from brainbridge_v2.infrastructure.signal_processing.butter_filter import ButterworthFilter
from brainbridge_v2.presentation.gui.styles import Theme

from brainbridge_v2.presentation.gui.dialogs.training_dialog import TrainingDialog

# Importar logger do novo módulo (compatível com OpenBCI)
try:
    from brainbridge_v2.infrastructure.acquisition.data_logger import OpenBCICSVLogger
    USE_OPENBCI_LOGGER = True
except Exception:
    USE_OPENBCI_LOGGER = False


class DeveloperSettingsDialog(QDialog):
    def __init__(self, *, telemetry_enabled: bool, parent=None):
        super().__init__(parent)
        from brainbridge_v2.application.runtime_config import get_runtime
        self.setWindowTitle("Modo Desenvolvedor")
        self.setModal(True)
        self.resize(380, 320)
        self.setStyleSheet(Theme.get_stylesheet())
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Configurações de Desenvolvimento")
        title.setStyleSheet(Theme.section_title("15px"))
        layout.addWidget(title)

        self.telemetry_checkbox = QCheckBox("Ativar telemetria da IA")
        self.telemetry_checkbox.setChecked(bool(telemetry_enabled))
        layout.addWidget(self.telemetry_checkbox)

        self.rl_checkbox = QCheckBox("Ativar RL online (aprende com ✓/✗)")
        self.rl_checkbox.setChecked(bool(get_runtime("rl_enabled", False)))
        layout.addWidget(self.rl_checkbox)

        rl_k_row = QHBoxLayout()
        rl_k_row.addWidget(QLabel("RL: aplicar a cada K feedbacks"))
        self.rl_k_spin = QSpinBox()
        self.rl_k_spin.setRange(1, 50)
        self.rl_k_spin.setValue(int(get_runtime("rl_batch_k", 5)))
        rl_k_row.addWidget(self.rl_k_spin)
        layout.addLayout(rl_k_row)

        calib_row = QHBoxLayout()
        calib_row.addWidget(QLabel("Calibração: trials mínimos"))
        self.calib_trials_spin = QSpinBox()
        self.calib_trials_spin.setRange(2, 60)
        self.calib_trials_spin.setValue(int(get_runtime("calib_trials_required", 10)))
        calib_row.addWidget(self.calib_trials_spin)
        layout.addLayout(calib_row)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Aplicar")
        apply_btn.setStyleSheet(Theme.btn_blue("6px 16px", "13px", "700"))
        apply_btn.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def telemetry_enabled(self) -> bool:
        return self.telemetry_checkbox.isChecked()

    def rl_enabled(self) -> bool:
        return self.rl_checkbox.isChecked()

    def rl_batch_k(self) -> int:
        return int(self.rl_k_spin.value())

    def calib_trials_required(self) -> int:
        return int(self.calib_trials_spin.value())


class StreamingWidget(QWidget):
    """Widget para streaming e gravação de dados"""
    
    # Signal para processar mensagens de acurácia de forma thread-safe
    accuracy_message_signal = pyqtSignal(str)
    unity_message_signal = pyqtSignal(str, int)
    
    def __init__(
        self,
        eeg_stream_controller: EEGStreamController,
        inference_controller: InferenceController,
        training_controller: TrainingController,
        patient_controller: PatientController,
        recording_controller: RecordingController,
        session_controller: SessionController,
        marker_controller: MarkerController,
        unity_controller: UnityController,
        esp32_controller: ESP32Controller,
        parent=None,
    ):
        super().__init__(parent)
        self.eeg_stream_controller = eeg_stream_controller
        self.inference_controller = inference_controller
        self.training_controller = training_controller
        self.patient_controller = patient_controller
        self.recording_controller = recording_controller
        self.session_controller = session_controller
        self.marker_controller = marker_controller
        self.unity_controller = unity_controller
        self.esp32_controller = esp32_controller
        self.connect_button_enabled = True
        self.record_button_enabled = False
        self.eeg_connection_phase = "standby"
        self.unity_connection_phase = "standby"
        self.orthosis_connection_phase = "standby"
        self.disconnection_in_progress = False
        self._inference_job = None
        self._inference_closed = False

    # Streaming / logging state
        self.csv_logger = None
        self.is_recording = False
        self.session_affected_hand = None
        self._movement_authorization = None
        self._movement_sent = set()
        self.current_recording_id = None
        self.pending_marker = None  # Para marcadores pendentes no logger OpenBCI
        self.baseline_timer = QTimer()  # Timer para baseline

    # Timer de sessão
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_elapsed_seconds = 0

    # Configuracao base do modelo
        # Force canonical window_size to 250 (HardThinking canonical)
        self.window_size = DEFAULT_RUNTIME_CONFIG.window_size  # 2s @ 125Hz
        self.channels = DEFAULT_RUNTIME_CONFIG.channels
        # Acquisition remains the full 16-channel cap, 250 samples at 125 Hz.

        self.predictions = deque(maxlen=50)  # Últimas predições

    # Estados do servidor UDP
        self.udp_server_active = False

    # Inicializar callbacks de comunicação
        self.eeg_stream_controller.set_data_callback(self.on_data_received)
        self.eeg_stream_controller.set_connection_callback(self.on_connection_status)
        self.unity_controller.set_message_callback(self._on_unity_message)
        self.unity_controller.set_connection_callback(self._on_unity_connection)
        self.esp32_controller.set_connection_callback(self._on_esp32_connection)
        self.esp32_connected = False

    # Timer para ações automáticas no jogo
        self.game_action_timer = QTimer()
        self.game_action_timer.setSingleShot(True)

    # Controle para aguardar resposta antes do próximo sinal
        self.waiting_for_response = False

    # Controle de janela de tempo para IA (configurável; default reduzido)
        self.ai_prediction_enabled = False
        self.task_start_time = None
        # Reduce default AI window to 2 seconds to send triggers sooner (milliseconds)
        self.ai_window_duration = DEFAULT_RUNTIME_CONFIG.ai_window_duration_ms
        self.game_inference = GameInferenceCoordinator(
            window_size=self.window_size,
            channels=self.channels,
            window_duration_ms=self.ai_window_duration,
        )
        # Modo Livre: loop continuo sem VR obrigatorio (janela deslizante).
        self.free_inference = FreeRunInferenceCoordinator(
            window_size=self.window_size,
            channels=self.channels,
            stride=DEFAULT_RUNTIME_CONFIG.free_stride,
        )
        self.free_running = False
        self.free_predictions = deque(maxlen=50)
        self._free_inference_busy = False
        self._free_last_send_ms: float = 0.0
        # RL online (feedback humano): buffer de (janela, pred, conf, rotulo?).
        self._rl_feedback: list = []
        self._rl_job = None
        self._rl_closed = False
        self._rl_labeled_since_update = 0
        self.developer_mode_enabled = False
        self.pipeline_telemetry = PipelineTelemetry(enabled=False)
        self.eeg_quality_validator = EEGWindowQualityValidator(
            max_abs_amplitude=DEFAULT_RUNTIME_CONFIG.eeg_max_abs_amplitude,
            min_channel_std=DEFAULT_RUNTIME_CONFIG.eeg_min_channel_std,
        )
        self.eeg_buffer = self.game_inference.eeg_buffer
        self.samples_since_last_prediction = self.game_inference.samples_since_window_start
        self.prediction_locked = self.game_inference.prediction_locked
        # Fallback interval for automatic game actions (was 30s); reduce to 10s
        self.game_action_interval = DEFAULT_RUNTIME_CONFIG.game_action_interval_ms

    # Variáveis para cálculo de acurácia
        self.accuracy_trials: List[AccuracyTrialViewModel] = []

    # UDP receiver para acurácia (recebe mensagens do sistema externo)
        self.accuracy_udp_receiver = None
        self.accuracy_thread = None

    # Conectar signal para processar mensagens de acurácia
        self.accuracy_message_signal.connect(self.process_accuracy_message)
        self.unity_message_signal.connect(self._process_unity_message)
        self.streaming_state = StreamingSessionStateViewModel(
            patient_id=None,
            task_type=None,
            recording_id=None,
            started_at_epoch=None,
            game_mode=False,
            recording_active=False,
            baseline_active=False,
            baseline_remaining_seconds=0,
            t1_count=0,
            t2_count=0,
            eeg_connected=False,
            eeg_mock_mode=False,
            unity_server_active=False,
            esp32_connected=False,
            model_loaded=False,
            model_name=None,
        )
        self.setup_ui()
        self._refresh_streaming_state()

    def _get_current_session(self):
        return self.session_controller.get_current_session()

    def _refresh_streaming_state(
        self,
        *,
        session: SessionViewModel | None = None,
        marker_state: MarkerStateViewModel | None = None,
        loaded_model: ModelViewModel | None = None,
    ) -> StreamingSessionStateViewModel:
        current_session = session if session is not None else self.session_controller.get_current_session()
        current_marker_state = (
            marker_state if marker_state is not None else self.marker_controller.get_state()
        )
        current_loaded_model = (
            loaded_model
            if loaded_model is not None
            else self.inference_controller.get_loaded_model()
        )
        self.streaming_state = StreamingSessionStatePresenter.present(
            session=current_session,
            marker_state=current_marker_state,
            loaded_model=current_loaded_model,
            recording_active=self.is_recording,
            eeg_connected=self.eeg_stream_controller.is_running(),
            eeg_mock_mode=self.eeg_stream_controller.is_mock_mode(),
            unity_server_active=self.udp_server_active,
            esp32_connected=self.esp32_connected,
        )
        return self.streaming_state

    def _get_session_started_at(self):
        return self.streaming_state.started_at_epoch

    def _is_game_mode(self) -> bool:
        return self.streaming_state.game_mode

    def _is_free_mode(self) -> bool:
        task = (self.task_combo.currentText() if hasattr(self, "task_combo") else "")
        if task.strip().lower() == "livre":
            return True
        return bool(self.streaming_state.free_mode)

    def _is_free_task_selected(self) -> bool:
        try:
            return self.task_combo.currentText().strip().lower() == "livre"
        except Exception:
            return False

    def _start_free_run(self) -> None:
        """Liga o loop continuo do modo Livre (sem exigir VR/ortese/modelo)."""
        self.free_inference.start()
        self.free_running = True
        self._free_inference_busy = False
        if self.inference_controller.has_loaded_model():
            self._set_ai_status("free_running")
        else:
            self._set_ai_status("free_no_model")

    def _stop_free_run(self) -> None:
        self.free_running = False
        self._free_inference_busy = False
        try:
            self.free_inference.stop()
        except Exception:
            pass

    def _free_can_send(self, confidence: float) -> bool:
        try:
            threshold = float(DEFAULT_RUNTIME_CONFIG.free_send_min_confidence)
        except Exception:
            threshold = 0.6
        if not (math.isfinite(confidence) and confidence >= threshold):
            return False
        now_ms = time.monotonic() * 1000
        try:
            cooldown = float(DEFAULT_RUNTIME_CONFIG.free_send_cooldown_ms)
        except Exception:
            cooldown = 3000.0
        if now_ms - float(self._free_last_send_ms) < cooldown:
            return False
        return True

    def _free_mark_sent(self) -> None:
        self._free_last_send_ms = float(time.monotonic() * 1000)

    # ---- RL online (feedback humano) ------------------------------------
    def _rl_enabled(self) -> bool:
        try:
            return bool(get_runtime("rl_enabled", False))
        except Exception:
            return False

    def _rl_push_prediction(self, window, predicted_index: int, confidence: float) -> None:
        """Guarda predicao para futuro rotulo (✓/✗ ou CORRECT/WRONG do VR)."""
        if not self._rl_enabled():
            return
        try:
            buf_max = int(get_runtime("rl_buffer_max", 200))
        except Exception:
            buf_max = 200
        try:
            self._rl_feedback.append({
                "window": np.array(window, copy=True),
                "pred": int(predicted_index),
                "conf": float(confidence),
                "label": None,
                "from_vr": False,
            })
            while len(self._rl_feedback) > max(10, buf_max):
                self._rl_feedback.pop(0)
            self._update_rl_status()
        except Exception as exc:
            print(f"[RL] Falha ao enfileirar: {exc}")

    def _rl_latest_unlabeled(self):
        for item in reversed(self._rl_feedback):
            if item.get("label") is None:
                return item
        return None

    def rl_feedback(self, correct: bool, *, from_vr: bool = False) -> bool:
        """Rotula a predicao mais recente (✓=certa, ✗=errada->outra classe)."""
        if not self._rl_enabled():
            return False
        item = self._rl_latest_unlabeled()
        if item is None:
            return False
        pred = int(item["pred"])
        item["label"] = pred if correct else (1 - pred)
        item["from_vr"] = bool(from_vr)
        self._rl_labeled_since_update += 1
        try:
            mistake_w = float(get_runtime("rl_mistake_weight", 3.0))
        except Exception:
            mistake_w = 3.0
        item["weight"] = 1.0 if correct else max(1.0, mistake_w)
        self._record_pipeline_event(
            "RL_FEEDBACK", correct=bool(correct), from_vr=bool(from_vr),
            pred=pred, label=int(item["label"]))
        self._update_rl_status()
        self._rl_maybe_apply()
        return True

    def _rl_maybe_apply(self) -> None:
        """Aplica sozinho a cada K feedbacks (em background)."""
        if not self._rl_enabled() or self._rl_job is not None:
            return
        try:
            k = int(get_runtime("rl_batch_k", 5))
            max_updates = int(get_runtime("rl_max_updates", 20))
        except Exception:
            k, max_updates = 5, 20
        if self._rl_labeled_since_update < max(1, k):
            return
        if self.inference_controller.rl_updates_count() >= max(1, max_updates):
            print("[RL] Limite de updates da sessao atingido.")
            return
        newly = [it for it in self._rl_feedback if it.get("label") is not None
                 and not it.get("applied")]
        if not newly:
            self._rl_labeled_since_update = 0
            return
        try:
            epochs = int(get_runtime("rl_epochs", 3))
            lr = float(get_runtime("rl_lr", 5e-5))
        except Exception:
            epochs, lr = 3, 5e-5
        windows = [it["window"] for it in newly]
        labels = [int(it["label"]) for it in newly]
        weights = [float(it.get("weight", 1.0)) for it in newly]
        for it in newly:
            it["applied"] = True
        self._rl_labeled_since_update = 0
        try:
            worker = RLUpdateWorker(self.inference_controller, windows, labels,
                                    weights, epochs=epochs, lr=lr,
                                    freeze_backbone=False)
            worker.signals.finished.connect(self._on_rl_finished, Qt.QueuedConnection)
            self._rl_job = worker
            self._record_pipeline_event("RL_UPDATE_START", n=len(newly))
            QThreadPool.globalInstance().start(worker)
        except Exception as exc:
            self._rl_job = None
            print(f"[RL] Falha ao enfileirar update: {exc}")

    @pyqtSlot(object)
    def _on_rl_finished(self, outcome) -> None:
        self._rl_job = None
        if getattr(self, "_rl_closed", False):
            return
        if not isinstance(outcome, RLUpdateOutcome):
            return
        if outcome.error is not None:
            print(f"[RL] Update falhou: {outcome.error}")
            self._record_pipeline_event("RL_UPDATE_FAILED", error=outcome.error)
        else:
            print(f"[RL] Update aplicado: n={outcome.n} loss={outcome.loss}")
            self._record_pipeline_event("RL_UPDATE_DONE", n=outcome.n,
                                        loss=outcome.loss,
                                        updates=outcome.updates_applied)
        self._update_rl_status()

    def _rl_restore_base(self) -> None:
        if self.inference_controller.rl_restore():
            self._rl_feedback.clear()
            self._rl_labeled_since_update = 0
            QMessageBox.information(self, "RL", "Modelo restaurado para o checkpoint pre-RL.")
        else:
            QMessageBox.warning(self, "RL", "Sem checkpoint pre-RL para restaurar.")
        self._update_rl_status()

    def _update_rl_status(self) -> None:
        if not hasattr(self, "rl_status_label"):
            return
        try:
            if not self._rl_enabled():
                self.rl_status_label.setText("RL: desligado")
                return
            labeled = sum(1 for it in self._rl_feedback if it.get("label") is not None)
            updates = self.inference_controller.rl_updates_count()
            self.rl_status_label.setText(
                f"RL: {labeled} feedbacks · {updates} updates aplicados")
        except Exception:
            pass

    def _ea_feed_sample(self, data) -> bool:
        """Alimenta a calibracao EA; retorna True se a IA deve aguardar.

        So atua quando ha modelo com EA carregado e ainda nao calibrado.
        Mostra progresso no painel Livre e no status da IA.
        """
        try:
            ctrl = self.inference_controller
            if ctrl is None or not ctrl.ea_required():
                return False
            if ctrl.ea_calibrated():
                return False
            try:
                ctrl.ea_observe_sample(data)
            except Exception as exc:
                print(f"[EA] Falha na calibracao: {exc}")
                return False
            if ctrl.ea_calibrated():
                print("[EA] Calibracao concluida; IA liberada.")
                self._record_pipeline_event("EA_CALIBRATED")
                if self._is_free_task_selected():
                    self._set_ai_status("free_running")
                return False
            done, total = ctrl.ea_progress()
            msg = f"Livre: calibrando IA ({done}/{total})"
            if hasattr(self, "free_result_label"):
                self.free_result_label.setText(msg)
            return True
        except Exception:
            return False

    def _sync_ai_prediction_state(self):
        self.eeg_buffer = self.game_inference.eeg_buffer
        self.samples_since_last_prediction = (
            self.game_inference.samples_since_window_start
        )
        self.ai_prediction_enabled = self.game_inference.is_window_open
        self.prediction_locked = self.game_inference.prediction_locked
        self.task_start_time = self.game_inference.window_started_at_ms

    def _reset_ai_prediction_window(self):
        self._movement_authorization = None
        self.game_inference.reset()
        self._sync_ai_prediction_state()

    def _schedule_game_callback(self, delay_ms, callback):
        generation = self.game_inference.generation
        QTimer.singleShot(delay_ms, lambda: (
            callback() if self.is_recording and self._is_game_mode()
            and generation == self.game_inference.generation else None
        ))

    def _arm_game_fallback(self):
        self.game_action_timer.stop()
        try:
            self.game_action_timer.timeout.disconnect()
        except TypeError:
            pass
        generation = self.game_inference.generation
        self.game_action_timer.timeout.connect(lambda: (
            self.game_random_action() if generation == self.game_inference.generation
            and self.is_recording and self._is_game_mode() else None
        ))
        self.game_action_timer.start(self.game_action_interval)

    def _authorize_transport(self, direction, transport):
        authorization = self._movement_authorization
        if not self.is_recording or not self._is_game_mode() or authorization is None:
            return False
        generation, index, confidence = authorization
        if not self.game_inference.allows_movement(
            generation, self.session_affected_hand, index, confidence
        ) or direction != UnityCommandMapper.from_prediction(index).direction:
            return False
        if transport in self._movement_sent:
            return False
        self._movement_sent.add(transport)  # No retries: firmware has no ACK contract.
        return True

    def _record_pipeline_event(self, name: str, **details):
        try:
            self.pipeline_telemetry.record(name, **details)
        except Exception as exc:
            print(f"[PIPELINE] Falha ao registrar evento {name}: {exc}")

    def open_developer_settings(self):
        dialog = DeveloperSettingsDialog(
            telemetry_enabled=self.developer_mode_enabled,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.set_developer_mode(dialog.telemetry_enabled())
            try:
                from brainbridge_v2.application.runtime_config import set_runtime
                set_runtime("rl_enabled", bool(dialog.rl_enabled()))
                set_runtime("rl_batch_k", int(dialog.rl_batch_k()))
                set_runtime("calib_trials_required",
                            int(dialog.calib_trials_required()))
            except Exception as exc:
                print(f"[CONFIG] Falha ao aplicar: {exc}")
            self._update_rl_status()

    def set_developer_mode(self, enabled: bool):
        self.developer_mode_enabled = bool(enabled)
        self.pipeline_telemetry.set_enabled(self.developer_mode_enabled)
        if hasattr(self, "developer_settings_btn"):
            label = "Dev: On" if self.developer_mode_enabled else "Dev: Off"
            self.developer_settings_btn.setText(label)
            self.developer_settings_btn.setStyleSheet(
                Theme.btn_dev(self.developer_mode_enabled)
                + " padding: 3px 8px; font-size: 11px; border-radius: 4px;"
            )
        
    @staticmethod
    def _v_separator():
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(Theme.vertical_separator())
        return line

    def setup_ui(self):
        """Layout sem scroll: cartoes no topo + plot expansivel + sidebar fixa."""
        T = Theme
        self.setStyleSheet(T.get_stylesheet() + T.compact_overrides())
        task_btn = T.btn_default("5px 6px", "12px", "700")
        task_btn_jogo = T.btn_default("5px 6px", "12px", "700")
        calib_btn = T.btn_default("4px 8px", "12px", "600")
        calib_btn_sm = T.btn_default("4px 6px", "11px", "600")
        connect_sm = T.btn_green("3px 8px", "11px", "600") + " border-radius: 4px;"
        combo_style = (
            f"padding: 3px 6px; font-size: 12px; background: {T.BTN_BG}; color: {T.TEXT_DARK}; "
            f"border: 1px solid {T.BTN_BORDER}; border-radius: 4px; font-weight: 600;"
        )
        card_title_style = T.card_title()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ================= FILEIRA DE CARTOES =================
        cards_row = QHBoxLayout()
        cards_row.setSpacing(6)

        # ---- CARD 1: Paciente e tarefa ----
        card1 = QGroupBox("1 · Paciente e tarefa")
        c1 = QVBoxLayout(card1)
        c1.setContentsMargins(8, 6, 8, 6)
        c1.setSpacing(4)

        self.patient_display_label = QLabel("Paciente: ####")
        self.patient_display_label.setWordWrap(True)
        self.patient_display_label.setStyleSheet(T.section_title("13px"))
        c1.addWidget(self.patient_display_label)

        pac_row = QHBoxLayout()
        pac_row.setSpacing(4)
        self.patient_combo = QComboBox()
        self.patient_combo.setStyleSheet(combo_style)
        self.patient_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.patient_combo.setMinimumContentsLength(10)
        self.patient_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.patient_combo.currentTextChanged.connect(self._on_patient_changed)
        self.refresh_patients_btn = QPushButton("↻")
        self.refresh_patients_btn.setToolTip("Atualizar lista de pacientes")
        self.refresh_patients_btn.setStyleSheet(T.btn_default("4px 8px", "12px", "700"))
        self.refresh_patients_btn.setFixedWidth(34)
        self.refresh_patients_btn.clicked.connect(self.refresh_patients)
        pac_row.addWidget(self.patient_combo, 1)
        pac_row.addWidget(self.refresh_patients_btn, 0)
        c1.addLayout(pac_row)

        task_row = QHBoxLayout()
        task_row.setSpacing(4)
        self.btn_baseline = QPushButton("Baseline")
        self.btn_baseline.setStyleSheet(task_btn)
        self.btn_baseline.clicked.connect(lambda: self._set_task("Baseline"))
        self.btn_jogo = QPushButton("Jogo")
        self.btn_jogo.setStyleSheet(task_btn_jogo)
        self.btn_jogo.clicked.connect(lambda: self._set_task("Jogo"))
        self.btn_livre = QPushButton("Livre")
        self.btn_livre.setStyleSheet(task_btn_jogo)
        self.btn_livre.setToolTip("Modo Livre: IA em tela sem VR obrigatorio. VR/ortese opcionais.")
        self.btn_livre.clicked.connect(lambda: self._set_task("Livre"))
        task_row.addWidget(self.btn_baseline, 1)
        task_row.addWidget(self.btn_jogo, 1)
        task_row.addWidget(self.btn_livre, 1)
        c1.addLayout(task_row)

        train_row = QHBoxLayout()
        train_row.setSpacing(4)
        self.btn_iniciar_treino = QPushButton("▶ Treino")
        self.btn_iniciar_treino.setStyleSheet(calib_btn)
        self.btn_iniciar_treino.clicked.connect(lambda: self._set_task("Treino"))
        self.btn_calib_esq = QPushButton("◀ Esq")
        self.btn_calib_esq.setStyleSheet(calib_btn_sm)
        self.btn_calib_esq.clicked.connect(lambda: self.add_marker("T1"))
        self.btn_calib_dir = QPushButton("Dir ▶")
        self.btn_calib_dir.setStyleSheet(calib_btn_sm)
        self.btn_calib_dir.clicked.connect(lambda: self.add_marker("T2"))
        train_row.addWidget(self.btn_iniciar_treino, 1)
        train_row.addWidget(self.btn_calib_esq, 1)
        train_row.addWidget(self.btn_calib_dir, 1)
        c1.addLayout(train_row)
        cards_row.addWidget(card1, 6)

        # ---- CARD 2: Conexoes (compacto, alinhado ao topo) ----
        card2 = QGroupBox("2 · Conexões")
        c2 = QVBoxLayout(card2)
        c2.setContentsMargins(8, 4, 8, 4)
        c2.setSpacing(3)

        conn_top = QHBoxLayout()
        conn_top.setSpacing(4)
        self.connect_btn = QPushButton("Conectar tudo")
        self.connect_btn.setStyleSheet(T.btn_green("3px 8px", "11px", "600") + " border-radius: 4px;")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.developer_settings_btn = QPushButton("Dev: Off")
        self.developer_settings_btn.setStyleSheet(
            T.btn_dev(False) + " padding: 3px 8px; font-size: 11px; border-radius: 4px;")
        self.developer_settings_btn.clicked.connect(self.open_developer_settings)
        conn_top.addWidget(self.connect_btn, 1)
        conn_top.addWidget(self.developer_settings_btn, 0)
        c2.addLayout(conn_top)

        status_grid = QGridLayout()
        status_grid.setSpacing(2)
        status_grid.setContentsMargins(0, 0, 0, 0)
        status_grid.setColumnStretch(0, 1)
        status_grid.setColumnStretch(1, 0)

        self.status_eeg = QLabel("EEG - Standby")
        self.status_eeg.setStyleSheet(T.status_text("off"))
        self.connect_eeg_btn = QPushButton("Conectar")
        self.connect_eeg_btn.setStyleSheet(connect_sm)
        self.connect_eeg_btn.clicked.connect(self.toggle_eeg_connection)

        self.status_vr = QLabel("VR - Standby")
        self.status_vr.setStyleSheet(T.status_text("off"))
        self.connect_vr_btn = QPushButton("Conectar")
        self.connect_vr_btn.setStyleSheet(connect_sm)
        self.connect_vr_btn.clicked.connect(self.toggle_udp_server)

        self.status_ortese = QLabel("ORTESE - Standby")
        self.status_ortese.setStyleSheet(T.status_text("off"))
        self.connect_ortese_btn = QPushButton("Conectar")
        self.connect_ortese_btn.setStyleSheet(connect_sm)
        self.connect_ortese_btn.clicked.connect(self.toggle_esp32_connection)

        status_grid.addWidget(self.status_eeg, 0, 0)
        status_grid.addWidget(self.connect_eeg_btn, 0, 1)
        status_grid.addWidget(self.status_vr, 1, 0)
        status_grid.addWidget(self.connect_vr_btn, 1, 1)
        status_grid.addWidget(self.status_ortese, 2, 0)
        status_grid.addWidget(self.connect_ortese_btn, 2, 1)
        c2.addLayout(status_grid)
        c2.addStretch(1)
        cards_row.addWidget(card2, 5)

        # ---- CARD 3: Gravacao ----
        card3 = QGroupBox("3 · Gravação")
        c3 = QVBoxLayout(card3)
        c3.setContentsMargins(8, 6, 8, 6)
        c3.setSpacing(4)
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.setStyleSheet(T.btn_green("5px 10px", "12px", "600"))
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.gravacao_status = QLabel("Não gravando")
        self.gravacao_status.setStyleSheet(Theme.recording_status_label())
        self.gravacao_status.setWordWrap(True)
        self.session_timer_label = QLabel("Sessão: 00:00:00")
        self.session_timer_label.setStyleSheet(T.section_title("13px"))
        c3.addWidget(self.record_btn)
        c3.addWidget(self.gravacao_status)
        c3.addWidget(self.session_timer_label)
        cards_row.addWidget(card3, 5)

        layout.addLayout(cards_row)

        # ================= SIDEBAR (largura fixa, sem scroll) =================
        side_panel = QWidget()
        side_panel.setFixedWidth(T.SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(6)

        # ---- CARD 4: IA / Resultado ----
        model_group = QGroupBox("4 · IA / Resultado")
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(8, 6, 8, 6)
        model_layout.setSpacing(3)

        self.free_result_label = QLabel("Livre: aguardando EEG")
        self.free_result_label.setStyleSheet(
            "font-size: 19px; font-weight: 800; padding: 2px;")
        self.free_result_label.setWordWrap(True)
        model_layout.addWidget(self.free_result_label)

        self.prediction_label = QLabel("")
        self.prediction_label.setWordWrap(True)
        model_layout.addWidget(self.prediction_label)

        probs_row = QHBoxLayout()
        probs_row.setSpacing(4)
        self.prob_left_label = QLabel("")
        self.prob_left_label.setWordWrap(True)
        self.prob_left_label.setStyleSheet("font-size: 11px;")
        self.prob_right_label = QLabel("")
        self.prob_right_label.setWordWrap(True)
        self.prob_right_label.setStyleSheet("font-size: 11px;")
        probs_row.addWidget(self.prob_left_label, 1)
        probs_row.addWidget(self.prob_right_label, 1)
        model_layout.addLayout(probs_row)

        self.accuracy_label = QLabel("Acurácia: 0% (0/0)")
        self.accuracy_label.setWordWrap(True)
        model_layout.addWidget(self.accuracy_label)

        self.ai_status_label = QLabel("")
        self.ai_status_label.setWordWrap(True)
        model_layout.addWidget(self.ai_status_label)

        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        self.model_status_label = QLabel("Sem modelo")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("font-size: 11px;")
        self.free_model_btn = QPushButton("Modelo")
        self.free_model_btn.setToolTip(
            "Carrega o modelo generalizado mais recente (treino opcional).")
        self.free_model_btn.clicked.connect(self.load_model)
        model_row.addWidget(self.model_status_label, 1)
        model_row.addWidget(self.free_model_btn, 0)
        model_layout.addLayout(model_row)

        opts_row = QHBoxLayout()
        opts_row.setSpacing(4)
        self.free_vr_checkbox = QCheckBox("VR")
        self.free_vr_checkbox.setToolTip("Espelhar resultado no VR (opcional)")
        self.free_vr_checkbox.setChecked(True)
        self.free_ortese_checkbox = QCheckBox("Órtese")
        self.free_ortese_checkbox.setToolTip("Acionar órtese (opcional)")
        self.free_ortese_checkbox.setChecked(True)
        self.ia_live_label = QLabel("Esq 0 · Dir 0")
        self.ia_live_label.setStyleSheet(
            f"font-size: 11px; font-weight: 800; color: {T.ORANGE}; background: transparent;")
        self.ia_live_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        opts_row.addWidget(self.free_vr_checkbox, 0)
        opts_row.addWidget(self.free_ortese_checkbox, 0)
        opts_row.addWidget(self.ia_live_label, 1)
        model_layout.addLayout(opts_row)

        self.rl_status_label = QLabel("RL: desligado")
        self.rl_status_label.setWordWrap(True)
        self.rl_status_label.setStyleSheet("font-size: 11px;")
        model_layout.addWidget(self.rl_status_label)

        rl_row = QHBoxLayout()
        rl_row.setSpacing(4)
        self.rl_correct_btn = QPushButton("✓ Acertou")
        self.rl_correct_btn.setToolTip("Marca a última predição como correta (recompensa).")
        self.rl_correct_btn.clicked.connect(lambda: self.rl_feedback(True))
        self.rl_wrong_btn = QPushButton("✗ Errou")
        self.rl_wrong_btn.setToolTip("Marca a última predição como errada (a outra classe vira o rótulo).")
        self.rl_wrong_btn.clicked.connect(lambda: self.rl_feedback(False))
        self.rl_restore_btn = QPushButton("↺ Base")
        self.rl_restore_btn.setToolTip("Restaurar base: desfaz updates de RL da sessão (trava anti-drift).")
        self.rl_restore_btn.clicked.connect(self._rl_restore_base)
        rl_row.addWidget(self.rl_correct_btn, 1)
        rl_row.addWidget(self.rl_wrong_btn, 1)
        rl_row.addWidget(self.rl_restore_btn, 1)
        model_layout.addLayout(rl_row)
        side_layout.addWidget(model_group)

        # ---- CARD 5: Marcadores ----
        marc_group = QGroupBox("Marcadores")
        marc_layout = QHBoxLayout(marc_group)
        marc_layout.setContentsMargins(8, 6, 8, 6)
        marc_layout.setSpacing(4)
        self.marcador_text = QLabel("T1: 0 | T2: 0")
        self.marcador_text.setStyleSheet(T.section_title("13px"))
        self.t1_btn = QPushButton("T1")
        self.t1_btn.setStyleSheet(T.btn_dark("4px 8px", "12px", "700"))
        self.t1_btn.clicked.connect(lambda: self.add_marker("T1"))
        self.t2_btn = QPushButton("T2")
        self.t2_btn.setStyleSheet(T.btn_blue("4px 8px", "12px", "700"))
        self.t2_btn.clicked.connect(lambda: self.add_marker("T2"))
        marc_layout.addWidget(self.marcador_text, 1)
        marc_layout.addWidget(self.t1_btn, 0)
        marc_layout.addWidget(self.t2_btn, 0)
        side_layout.addWidget(marc_group)
        side_layout.addStretch(1)

        # Placar vivo no lugar da tabela estatica (atualizado em _apply_game_stats).

        # ============ EEG AO VIVO (expansivel) ============
        self.bci_status_label = QLabel("Sistema BCI inicializado")
        self.bci_status_label.setStyleSheet(T.status_text("connected"))
        self.bci_status_label.setWordWrap(True)

        self.plot_widget = EEGPlotWidget()
        self.plot_widget.setMinimumSize(200, 120)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        plot_heading = QLabel("EEG AO VIVO  |  16 canais  |  125 Hz")
        plot_heading.setStyleSheet(T.section_title("13px"))
        plot_heading.setWordWrap(True)
        plot_layout.addWidget(plot_heading)
        plot_layout.addWidget(self.plot_widget, 1)
        plot_layout.addWidget(self.bci_status_label)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(6)
        self.workspace_splitter.addWidget(plot_panel)
        self.workspace_splitter.addWidget(side_panel)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 0)
        self.workspace_splitter.setSizes([1000, T.SIDEBAR_WIDTH])
        layout.addWidget(self.workspace_splitter, 1)

        self.setLayout(layout)

        # ============ Widgets internos (não visíveis, para compatibilidade) ============
        self.host_edit = QLineEdit("localhost")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(12345)
        self.status_label = self.status_eeg  # alias
        self.recording_label = self.gravacao_status  # alias
        self.t1_counter_label = QLabel("T1: 0")
        self.t2_counter_label = QLabel("T2: 0")
        self.baseline_label = QLabel("")
        self.baseline_timer = QTimer()
        self.baseline_timer.timeout.connect(self.update_baseline_timer)

        # task_combo interno (não visível) para compatibilidade com on_task_changed
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Baseline", "Treino", "Teste", "Jogo", "Livre"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)

        # Game mode labels (ocultos, para compatibilidade com presenters)
        self.accuracy_details_label = QLabel("")
        self.accuracy_group = QWidget()
        self.status_table_group = QWidget()
        self.total_predictions_label = QLabel("0")
        self.left_predictions_label = QLabel("0")
        self.right_predictions_label = QLabel("0")
        self.transitions_label = QLabel("0")
        self.confidence_label = QLabel("0%")
        self.stats_group = QWidget()
        self.game_group = QWidget()
        # Os labels visiveis (prediction/probs/model/accuracy/ai/free/rl) ja
        # foram criados na sidebar acima; aqui so os contadores ocultos.

        # Internal presenter targets must not become independent top-level windows.
        for group in (self.accuracy_group, self.status_table_group,
                      self.stats_group, self.game_group):
            group.setParent(self)
            group.setMaximumSize(0, 0)

        # Inicializar UDP auto-send checkboxes (para compatibilidade)
        self.udp_auto_send_checkbox = QCheckBox()
        self.udp_auto_send_checkbox.setChecked(True)
        self.esp32_auto_send_checkbox = QCheckBox()
        self.esp32_auto_send_checkbox.setChecked(True)
        self.udp_toggle_btn = QPushButton("")
        self.udp_status_label = QLabel("")
        self.udp_test_left_btn = QPushButton("")
        self.udp_test_right_btn = QPushButton("")
        self.esp32_toggle_btn = QPushButton("")
        self.esp32_status_label = QLabel("")
        self.esp32_test_left_btn = QPushButton("")
        self.esp32_test_right_btn = QPushButton("")

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_game_stats)
        self.stats_timer.start(1000)

        self._apply_connection_panel()
        self._apply_task_view_state()
        self._apply_accuracy_display()
        self._apply_prediction_display()
        self._set_ai_status("stopped")
        self.refresh_patients()

    def _set_task(self, task):
        """Muda a tarefa ativa via botão"""
        idx = self.task_combo.findText(task)
        if idx >= 0:
            self.task_combo.setCurrentIndex(idx)

    def _on_patient_changed(self, text):
        """Atualiza o label de paciente quando muda no combo"""
        if text and text != "Selecionar paciente...":
            try:
                patient_id = self.patient_combo.currentData()
                count = self._patient_session_count(int(patient_id)) if patient_id else 0
                self.patient_display_label.setText(
                    f"Paciente: {text} ({ProgressionPresenter.summary_text(count)})"
                )
            except Exception:
                self.patient_display_label.setText(f"Paciente: {text}")
        else:
            self.patient_display_label.setText("Paciente: ####")

    def _patient_calibrated(self, patient_id) -> bool:
        """True se o paciente ja tem modelo proprio (calibracao feita)."""
        try:
            return bool(self.training_controller.patient_model_available(int(patient_id)))
        except Exception:
            return False

    def _count_trials_or_zero(self, csv_path) -> int:
        try:
            from brainbridge_v2.infrastructure.ml.trainer import (
                count_labeled_trials_total)
            return int(count_labeled_trials_total(str(csv_path)))
        except Exception as exc:
            print(f"[CALIB] Falha ao contar trials: {exc}")
            return 0

    def _patient_session_count(self, patient_id: int) -> int:
        """Quantas gravações/sessões o paciente já possui (progressão)."""
        try:
            return max(0, len(self.recording_controller.list_patient_recordings(int(patient_id))))
        except Exception:
            return 0

    def _update_marcador_text(self):
        """Atualiza o texto dos marcadores na barra"""
        self.marcador_text.setText(
            f"T1: {self.streaming_state.t1_count} | T2: {self.streaming_state.t2_count}"
        )

    def _update_marker_labels(self, state: MarkerStateViewModel):
        self._refresh_streaming_state(marker_state=state)
        self.t1_counter_label.setText(f"T1: {state.t1_count}")
        self.t2_counter_label.setText(f"T2: {state.t2_count}")
        self._update_marcador_text()

    def _apply_status_label(self, label: QLabel, text: str, style_sheet: str):
        label.setText(text)
        label.setStyleSheet(style_sheet)

    def _apply_connection_panel(self):
        panel = ConnectionStatusPresenter.present(
            eeg_phase=self.eeg_connection_phase,
            vr_phase=self.unity_connection_phase,
            orthosis_phase=self.orthosis_connection_phase,
            connect_button_enabled=self.connect_button_enabled,
            record_button_enabled=self.record_button_enabled,
        )
        self._apply_status_label(self.status_eeg, panel.eeg.text, panel.eeg.style_sheet)
        self._apply_status_label(self.status_vr, panel.vr.text, panel.vr.style_sheet)
        self._apply_status_label(
            self.status_ortese,
            panel.orthosis.text,
            panel.orthosis.style_sheet,
        )
        self.connect_btn.setText(panel.connect_button_text)
        self.connect_btn.setStyleSheet(
            panel.connect_button_style
            + " padding: 3px 8px; font-size: 11px; border-radius: 4px;")
        self.connect_btn.setEnabled(panel.connect_button_enabled)
        
        # Atualizar botões individuais (compactos, iguais ao "Conectar tudo")
        if hasattr(self, 'connect_eeg_btn'):
            self.connect_eeg_btn.setText(panel.eeg_button_text)
            self.connect_eeg_btn.setStyleSheet(
                panel.eeg_button_style
                + " padding: 3px 8px; font-size: 11px; border-radius: 4px;")
            self.connect_eeg_btn.setEnabled(panel.connect_button_enabled)
        if hasattr(self, 'connect_vr_btn'):
            self.connect_vr_btn.setText(panel.vr_button_text)
            self.connect_vr_btn.setStyleSheet(
                panel.vr_button_style
                + " padding: 3px 8px; font-size: 11px; border-radius: 4px;")
            self.connect_vr_btn.setEnabled(panel.connect_button_enabled)
        if hasattr(self, 'connect_ortese_btn'):
            self.connect_ortese_btn.setText(panel.orthosis_button_text)
            self.connect_ortese_btn.setStyleSheet(
                panel.orthosis_button_style
                + " padding: 3px 8px; font-size: 11px; border-radius: 4px;")
            self.connect_ortese_btn.setEnabled(panel.connect_button_enabled)
            
        self.record_btn.setEnabled(panel.record_button_enabled)

    def _recording_status_text(self, hint: str = "") -> str:
        task = self.task_combo.currentText()
        if not self.is_recording:
            if task == "Livre" and getattr(self, "free_running", False):
                base = "Livre ao vivo"
                return f"{base} · {hint}" if hint else base
            return "Não gravando"
        if task == "Jogo":
            base = "Jogando"
        elif task == "Livre":
            base = "Livre gravando"
        else:
            base = "Gravando"
        if hint:
            return f"{base} · {hint}"
        return base

    def _apply_recording_ui(self, hint: str = ""):
        """Atualiza botão e texto curto de gravação (sem caminhos de arquivo)."""
        if not hasattr(self, "record_btn"):
            return
        self._apply_task_view_state()
        if self.is_recording:
            self.record_btn.setStyleSheet(Theme.btn_recording_active("5px 14px", "12px", "600"))
        elif self.record_btn.isEnabled():
            self.record_btn.setStyleSheet(Theme.btn_green("5px 14px", "12px", "600"))
        if hasattr(self, "gravacao_status"):
            self.gravacao_status.setText(self._recording_status_text(hint))
            self.gravacao_status.setStyleSheet(Theme.recording_status_label())

    def _apply_task_view_state(self):
        for button, task in ((self.btn_baseline, "Baseline"),
                             (self.btn_iniciar_treino, "Treino"),
                             (self.btn_jogo, "Jogo"),
                             (self.btn_livre, "Livre")):
            button.setCheckable(True)
            button.setChecked(self.task_combo.currentText() == task)
        task_view = TaskViewStatePresenter.present(
            self.task_combo.currentText(),
            self.is_recording,
        )
        self.record_btn.setText(task_view.record_button_text)
        self.status_table_group.setVisible(task_view.status_table_visible)
        self.game_group.setVisible(task_view.game_visible)
        self.stats_group.setVisible(task_view.stats_visible)
        self.accuracy_group.setVisible(task_view.accuracy_visible)

    def _apply_accuracy_display(self):
        accuracy_view = AccuracyPresenter.present(self.accuracy_trials)
        self.accuracy_label.setText(accuracy_view.summary_text)
        self.accuracy_label.setStyleSheet(accuracy_view.summary_style_sheet)
        self.accuracy_details_label.setText(accuracy_view.details_text)

    def _set_ai_status(self, state: str):
        ai_status = GameRuntimePresenter.present_ai_status(state)
        self.ai_status_label.setText(ai_status.text)
        self.ai_status_label.setStyleSheet(ai_status.style_sheet)

    def _apply_prediction_display(
        self,
        prediction: PredictionViewModel | None = None,
    ):
        prediction_view = GameRuntimePresenter.present_prediction(prediction)
        self.prediction_label.setText(prediction_view.prediction_text)
        self.prediction_label.setStyleSheet(prediction_view.prediction_style_sheet)
        self.prob_left_label.setText(prediction_view.left_probability_text)
        self.prob_right_label.setText(prediction_view.right_probability_text)
        if hasattr(self, "free_result_label"):
            if prediction is None:
                self.free_result_label.setText("Livre: aguardando EEG")
            else:
                side = "ESQUERDA" if int(prediction.predicted_index) == 0 else "DIREITA"
                conf = float(prediction.confidence)
                self.free_result_label.setText(f"Livre: {side} ({conf:.0%})")

    def _apply_game_stats(self):
        stats_view = GameRuntimePresenter.present_stats(self.predictions)
        self.total_predictions_label.setText(stats_view.total_predictions_text)
        self.left_predictions_label.setText(stats_view.left_predictions_text)
        self.right_predictions_label.setText(stats_view.right_predictions_text)
        self.transitions_label.setText(stats_view.transitions_text)
        self.confidence_label.setText(stats_view.confidence_text)
        if hasattr(self, "ia_live_label"):
            try:
                left = sum(1 for _, p, _ in self.predictions if p == 0)
                right = sum(1 for _, p, _ in self.predictions if p == 1)
                self.ia_live_label.setText(f"Esq {left} · Dir {right}")
            except Exception:
                pass
        
    def refresh_patients(self):
        """Atualiza a lista de pacientes"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecionar paciente...")
        
        try:
            patients = self.patient_controller.list_patients()
            for patient in patients:
                count = self._patient_session_count(patient['id'])
                self.patient_combo.addItem(
                    f"{patient['name']} (ID: {patient['id']}) - {count} sessões",
                    patient['id']
                )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")
    
    def toggle_connection(self):
        """Conecta/desconecta tudo de uma vez (EEG + UDP + ESP32)"""
        if not self.eeg_stream_controller.is_running():
            # === CONECTAR TUDO ===
            host = self.host_edit.text()
            port = self.port_spin.value()

            self.eeg_connection_phase = "connecting"
            self.unity_connection_phase = "connecting"
            self.orthosis_connection_phase = "connecting"
            self.connect_button_enabled = False
            self.record_button_enabled = False
            self._apply_connection_panel()
            
            # 1. EEG
            self.eeg_stream_controller.connect(host, port)

            # 2. UDP Unity
            try:
                if self.unity_controller.start_server():
                    self.udp_server_active = True
                    self.unity_connection_phase = "connected"
                else:
                    self.unity_connection_phase = "failed"
            except Exception:
                self.unity_connection_phase = "failed"

            # 3. ESP32
            try:
                if self.esp32_controller.connect():
                    self.esp32_connected = True
                    self.orthosis_connection_phase = "connected"
                else:
                    self.esp32_connected = False
                    self.orthosis_connection_phase = "standby"
            except Exception:
                self.esp32_connected = False
                self.orthosis_connection_phase = "failed"
            self._apply_connection_panel()

        else:
            # === DESCONECTAR TUDO ===
            self.disconnection_in_progress = True
            try:
                self.unity_controller.stop_server()
                self.udp_server_active = False
            except Exception:
                pass
            try:
                self.esp32_controller.disconnect()
                self.esp32_connected = False
            except Exception:
                pass
            self.eeg_connection_phase = "standby"
            self.unity_connection_phase = "standby"
            self.orthosis_connection_phase = "standby"
            self.connect_button_enabled = True
            self.record_button_enabled = False
            self._apply_connection_panel()
            self.eeg_stream_controller.disconnect()
        self._refresh_streaming_state()

    def toggle_eeg_connection(self):
        """Conecta ou desconecta apenas do EEG"""
        if not self.eeg_stream_controller.is_running():
            host = self.host_edit.text()
            port = self.port_spin.value()
            self.eeg_connection_phase = "connecting"
            self._apply_connection_panel()
            self.eeg_stream_controller.connect(host, port)
        else:
            self.disconnection_in_progress = True
            self.eeg_connection_phase = "standby"
            self._apply_connection_panel()
            self.eeg_stream_controller.disconnect()
        self._refresh_streaming_state()
    
    def manual_esp32_test(self, direction):
        """Teste manual do envio serial para ESP32"""
        # Manual/debug entry points obey the same gate, never synthesize predictions.
        if not self._authorize_transport(direction, "serial"):
            return False
        if self.esp32_connected:
            success = self.esp32_controller.send_direction(direction)
            
            if success:
                side_text = "esquerda" if direction == 'esquerda' else "direita"
                QMessageBox.information(self, "Teste ESP32", f"Trigger enviado: Mão {side_text}")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao enviar comando para ESP32!")
        else:
            QMessageBox.warning(self, "Aviso", "ESP32 não está conectado!")
    
    def send_esp32_signal(self, direction):
        """Envia sinal serial para ESP32 se conectado e o envio automático estiver habilitado"""
        if self.esp32_connected and self.esp32_auto_send_checkbox.isChecked() and self._authorize_transport(direction, "serial"):
            success = self.esp32_controller.send_direction(direction)
            
            if not success:
                print(f"Falha ao enviar sinal serial para ESP32: {direction}")
            return success
        return False

    def toggle_esp32_connection(self):
        """Conecta ou desconecta do ESP32"""
        try:
            if not self.esp32_connected:
                self.orthosis_connection_phase = "connecting"
                self._apply_connection_panel()
                
                # Tentar conectar
                connected = self.esp32_controller.connect()
                
                if connected:
                    self.esp32_connected = True
                    self.orthosis_connection_phase = "connected"
                    QMessageBox.information(self, "Sucesso", "ESP32 conectado com sucesso na COM4!")
                else:
                    self.orthosis_connection_phase = "failed"
                    QMessageBox.critical(self, "Erro", "Falha ao conectar ESP32.\nVerifique se o ESP32 está conectado na COM4.")
            else:
                # Desconectar
                self.esp32_controller.disconnect()
                self.esp32_connected = False
                self.orthosis_connection_phase = "standby"
                QMessageBox.information(self, "Sucesso", "ESP32 desconectado com sucesso!")
                
        except Exception as e:
            self.orthosis_connection_phase = "failed"
            QMessageBox.critical(self, "Erro", f"Erro ao conectar/desconectar ESP32: {e}")
        self._apply_connection_panel()
        self._refresh_streaming_state()

    def _on_esp32_connection(self, connected: bool):
        """Callback para mudanças de conexão ESP32"""
        self.esp32_connected = connected
        self.orthosis_connection_phase = "connected" if connected else "standby"
        if not connected and hasattr(self, 'esp32_status_label'):
            self.esp32_status_label.setText("ESP32: Desconectado")
            self.esp32_status_label.setStyleSheet(Theme.status_text("error") + " font-size: 12px;")
            self.esp32_toggle_btn.setText("Conectar ESP32")
            self.esp32_toggle_btn.setStyleSheet(Theme.btn_dev(True))
            self.esp32_test_left_btn.setEnabled(False)
            self.esp32_test_right_btn.setEnabled(False)
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def manual_udp_test(self, direction):
        """Teste manual do envio UDP"""
        if not self._authorize_transport(direction, "unity"):
            return False
        if self.udp_server_active:
            success = self.unity_controller.send_action(direction)
            if success:
                side_text = "esquerda" if direction == 'esquerda' else "direita"
                QMessageBox.information(self, "Teste UDP", f"Sinal enviado: Mão {side_text}")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao enviar sinal UDP!")
        else:
            QMessageBox.warning(self, "Aviso", "Servidor UDP não está ativo!")
    
    def send_udp_signal(self, direction):
        """Envia sinal UDP se o servidor estiver ativo e o envio automático estiver habilitado"""
        if self.udp_server_active and self.udp_auto_send_checkbox.isChecked() and self._authorize_transport(direction, "unity"):
            success = self.unity_controller.send_action(direction)
            if not success:
                print(f"Falha ao enviar sinal UDP para {direction}")
            return success
        return False

    def toggle_udp_server(self):
        """Inicia ou para o servidor UDP manualmente (conectado ao botão)."""
        try:
            if not self.udp_server_active:
                self.unity_connection_phase = "connecting"
                self._apply_connection_panel()
                
                # Tentar iniciar servidor
                started = False
                try:
                    started = self.unity_controller.start_server()
                except Exception as e:
                    print(f"Erro ao iniciar servidor UDP: {e}")

                if started:
                    self.udp_server_active = True
                    self.unity_connection_phase = "connected"
                    QMessageBox.information(self, "Sucesso", "Servidor UDP iniciado com sucesso!\nBroadcast do IP enviado automaticamente.")
                else:
                    self.unity_connection_phase = "failed"
                    QMessageBox.critical(self, "Erro", "Falha ao iniciar servidor UDP")
            else:
                # Parar servidor
                try:
                    self.unity_controller.stop_server()
                except Exception:
                    pass
                self.udp_server_active = False
                self.unity_connection_phase = "standby"
                QMessageBox.information(self, "Sucesso", "Servidor UDP parado com sucesso!")
        except Exception as e:
            self.unity_connection_phase = "failed"
            QMessageBox.critical(self, "Erro", f"Erro ao alternar servidor UDP: {e}")
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Iniciar gravação
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Erro", "Selecione um paciente!")
                return
            
            selected_patient_id = int(self.patient_combo.currentData())
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            # Obter tarefa do dropdown
            task = self.task_combo.currentText().lower().replace(" ", "_")  # ex: "Baseline" -> "baseline"
            self._reset_ai_prediction_window()
            self.session_affected_hand = None
            try:
                patient = next((p for p in self.patient_controller.list_patients()
                                if p['id'] == selected_patient_id), {})
                self.session_affected_hand = patient.get("affected_hand")
            except Exception:
                pass  # Fail closed; recording without movement remains available.
            if task == "jogo" and self.session_affected_hand not in ("left", "right"):
                QMessageBox.warning(self, "Cadastro incompleto", "Cadastre a mao afetada do paciente (esquerda/direita) antes de iniciar o jogo.")
                return

            # Calibracao obrigatoria (1x por paciente) p/ Jogo e Livre:
            # sem modelo proprio do paciente, oferece o Treino de calibracao.
            if task in ("jogo", "livre") and not self._patient_calibrated(selected_patient_id):
                try:
                    required = int(get_runtime("calib_trials_required", 10))
                except Exception:
                    required = 10
                answer = QMessageBox.question(
                    self, "Calibração necessária",
                    f"Paciente sem modelo próprio.\n\n"
                    f"Grave um Treino de calibração com pelo menos {required} trials "
                    f"(T1/T2) para liberar o modo {self.task_combo.currentText()}.\n\n"
                    f"Iniciar Treino agora?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if answer == QMessageBox.Yes:
                    self._set_task("Treino")
                return
            
            # Verificar se é modo jogo (VR obrigatorio no fluxo classico)
            if task == "jogo":
                if not self.inference_controller.has_loaded_model():
                    if not self.load_model():
                        return
                # Limpar variáveis do jogo
                self.predictions.clear()
                self._reset_ai_prediction_window()
                self._apply_prediction_display()
                self._apply_game_stats()

                # Resetar dados de acurácia
                self.reset_accuracy_data()

                # Resetar controle de resposta
                self.waiting_for_response = False

                # Resetar status visual da IA
                self._set_ai_status("waiting_task")

                # Resetar contadores de ações no início da gravação
                self.reset_action_counters()

                # Iniciar UDP receiver para acurácia - agora sempre disponível
                try:
                    self.start_accuracy_udp_receiver()
                except Exception as e:
                    print(f"Erro ao iniciar UDP receiver de acurácia: {e}")

                # Iniciar primeiro sinal aleatório imediatamente (não usar timer automático)
                # O próximo sinal será enviado apenas após receber CORRECT/WRONG
                self._schedule_game_callback(1000, self.send_next_random_signal)

                # Manter timer como fallback caso não receba resposta (usar game_action_interval)
                self._arm_game_fallback()

            # Modo Livre: treinamento/modelo opcional, VR/ortese opcionais.
            if task == "livre":
                self.free_predictions.clear()
                self.predictions.clear()
                self._apply_prediction_display()
                self._start_free_run()
                self._apply_recording_ui(hint="Livre")
            
            try:
                # Usar logger OpenBCI se disponível
                if USE_OPENBCI_LOGGER:
                    self.csv_logger = OpenBCICSVLogger(
                        patient_id=f"P{selected_patient_id:03d}",
                        task=task,
                        patient_name=patient_name,  # Adicionar nome do paciente
                        base_path=os.path.dirname(get_recording_path(""))
                    )
                    filename = self.csv_logger.filename
                    # Mostrar caminho relativo para feedback visual
                    display_path = f"{self.csv_logger.patient_folder}/{filename}"

                
                self.is_recording = True
                self._apply_recording_ui()
                
                # Habilitar botões de marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                # self.baseline_btn.setEnabled(True)  # Botão removido
                
                # Resetar contadores
                self.reset_action_counters()
                
                # Registrar gravação via application/interface adapters
                recording_path = display_path if USE_OPENBCI_LOGGER else filename
                self.current_recording_id = self.recording_controller.start_recording(
                    StartRecordingRequest(
                        patient_id=selected_patient_id,
                        filename=recording_path,
                        task_type=task,
                    )
                )
                started_session = self.session_controller.start_session(
                    StartSessionRequest(
                        patient_id=selected_patient_id,
                        task_type=task,
                        recording_id=self.current_recording_id,
                        started_at_epoch=time.time(),
                    )
                )
                self._refresh_streaming_state(session=started_session)
                
                # =====================================================================
                # PUBLICAR SESSÃO REAL NO VR + TRIGGER
                # Publica nome/lado/tarefa reais (antes ia sempre debug João/Esquerdo).
                # O VR precisa enviar "Confirm" para liberar o trigger (READY).
                # =====================================================================
                try:
                    if self.unity_controller.is_server_active() and self.unity_controller.is_client_connected():
                        lado_vr = "Esquerdo" if (self.session_affected_hand or "left") == "left" else "Direito"
                        # Progressão: nível deriva das sessões já realizadas (0-11).
                        # Conta antes desta gravação para não contar a sessão atual.
                        try:
                            previous_count = max(0, len(
                                self.recording_controller.list_patient_recordings(selected_patient_id)
                            ) - 1)
                        except Exception:
                            previous_count = 0
                        nivel_vr = ProgressionPresenter.level_for_session_count(previous_count)
                        try:
                            self.unity_controller.set_pending_session(
                                patient_name, nivel_vr, lado_vr, task, previous_count
                            )
                            print(f"[GRAVAÇÃO] sessão VR publicada: {patient_name} / {lado_vr} / {task} / nível {nivel_vr} ({previous_count} sessões)", flush=True)
                        except Exception as e:
                            print(f"[GRAVAÇÃO] Erro ao publicar sessão VR: {e}", flush=True)
                        time.sleep(0.5)  # Pequeno delay para garantir que tudo está pronto
                        if not self.unity_controller.send_trigger():
                            print("[GRAVAÇÃO] trigger VR pendente (aguardando Confirm do VR)", flush=True)
                        else:
                            print("[GRAVAÇÃO] send_trigger() enviado para VR", flush=True)
                except Exception as e:
                    print(f"[GRAVAÇÃO] Erro ao enviar send_trigger(): {e}", flush=True)
                # =====================================================================
                
                # Iniciar timer de sessão
                self.session_timer.start(1000)  # Atualizar a cada segundo
                
            except Exception as e:
                self.is_recording = False
                self._reset_ai_prediction_window()
                self.game_action_timer.stop()
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
        else:
            self._reset_ai_prediction_window()
            # Parar gravação
            # Parar logging, mas manter referência para obter o caminho do arquivo
            logger = None
            if self.csv_logger:
                logger = self.csv_logger
                try:
                    logger.stop_logging()
                except Exception:
                    pass
            # Limpar a referência de longo prazo (UI não mais grava)
            self.csv_logger = None
            
            self.is_recording = False
            
            # =====================================================================
            # ENVIAR END_TASK PARA O VR
            # =====================================================================
            current_task = self.task_combo.currentText()
            try:
                if self.unity_controller.is_server_active() and self.unity_controller.is_client_connected():
                    # Enviar end_task
                    self.unity_controller.end_task()
                    print(f"[GRAVAÇÃO] end_task() enviado para VR", flush=True)
                    
                    # Se for jogo, também enviar end_session com mensagem motivacional
                    if current_task == "Jogo":
                        time.sleep(0.3)  # Pequeno delay
                        self.unity_controller.end_session("Parabéns! Sessão finalizada com sucesso!")
                        print(f"[GRAVAÇÃO] end_session() com mensagem enviada para VR", flush=True)
            except Exception as e:
                print(f"[GRAVAÇÃO] Erro ao enviar end_task/end_session: {e}", flush=True)
            
            # =====================================================================
            
            # Parar UDP receiver de acurácia
            self.stop_accuracy_udp_receiver()
            
            # Parar timer de ações automáticas no jogo
            if self.game_action_timer.isActive():
                self.game_action_timer.stop()
            
            # Resetar controle de resposta
            self.waiting_for_response = False

            # Resetar controle de IA (modo Jogo). Modo Livre continua ao vivo.
            self._reset_ai_prediction_window()

            # Resetar status visual da IA
            if current_task == "Livre":
                # Mantem o loop livre rodando para ver resposta em tela sem gravar.
                self._start_free_run()
            else:
                self._stop_free_run()
                self._set_ai_status("stopped")
            
            # Resetar contadores de ações
            self.reset_action_counters()
                
            self._apply_recording_ui()
            
            # Desabilitar botões de marcadores
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            # self.baseline_btn.setEnabled(False)  # Botão removido
            
            # Parar timer de baseline se estiver rodando
            if self.baseline_timer.isActive():
                self.baseline_timer.stop()
                self.baseline_label.setText("")
            self.marker_controller.reset_state()
            self._update_marker_labels(self.marker_controller.get_state())
            
            # Parar timer de sessão
            self.session_timer.stop()
            self.session_elapsed_seconds = 0
            # Atualizar progressão exibida (a sessão recém-gravada conta agora)
            try:
                self._on_patient_changed(self.patient_combo.currentText())
            except Exception:
                pass
            self.update_session_timer()

            current_session = self._get_current_session()
            if self.current_recording_id is not None:
                duration_seconds = 0
                if current_session is not None:
                    duration_seconds = max(
                        0,
                        int(time.time() - float(current_session.started_at_epoch)),
                    )
                self.recording_controller.stop_recording(self.current_recording_id, duration_seconds)
                self.current_recording_id = None
            ended_session = self.session_controller.end_session()
            self._refresh_streaming_state()
            
            # Verificar se é tarefa de treino para mostrar popup de treinamento
            print(f"[DEBUG] stop_recording: current_task={current_task}, logger_present={logger is not None}")
            if current_task == "Treino":
                # Obter informações para o treino
                patient_name = self.patient_combo.currentText().split(" (ID:")[0]
                csv_file_path = None
                
                # Obter caminho do arquivo CSV gravado a partir da referência local 'logger'
                if USE_OPENBCI_LOGGER and hasattr(logger, 'get_full_path'):
                    try:
                        csv_file_path = logger.get_full_path()
                    except Exception:
                        csv_file_path = None
                elif logger is not None and hasattr(logger, 'filename'):
                    # Construir caminho completo para logger simples
                    csv_file_path = str(get_recording_path(logger.filename))
                
                print(f"[DEBUG] stop_recording: csv_file_path={csv_file_path}")
                if csv_file_path and os.path.exists(csv_file_path):
                    # Calibracao exige N trials rotulados (configuravel).
                    try:
                        required = int(get_runtime("calib_trials_required", 10))
                    except Exception:
                        required = 10
                    n_trials = self._count_trials_or_zero(csv_file_path)
                    print(f"[CALIB] trials rotulados: {n_trials} (minimo {required})")
                    if n_trials < required:
                        answer = QMessageBox.question(
                            self, "Calibração insuficiente",
                            f"Sessão com {n_trials} trials rotulados (mínimo {required}).\n\n"
                            f"Treinar assim mesmo?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if answer != QMessageBox.Yes:
                            QMessageBox.information(
                                self, "Calibração",
                                "Treino descartado para calibração. Grave novamente com mais trials.")
                            return
                    # Iniciar fluxo automático de treino sem pedir confirmação
                    # show_training_dialog agora suporta auto_start=True
                    try:
                        print("[DEBUG] stop_recording: launching auto training dialog")
                        patient_id = self.patient_combo.currentData()
                        if ended_session is not None:
                            patient_id = ended_session.patient_id
                        self.show_training_dialog(csv_file_path, patient_id, patient_name, auto_start=True)
                    except Exception as e:
                        print(f"[DEBUG] stop_recording: failed to start training dialog: {e}")
                        QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
                else:
                    QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
            else:
                QMessageBox.information(self, "Sucesso", "Gravação finalizada!")
    

    def game_random_action(self):
        """Executa uma ação aleatória no jogo (fallback caso não receba resposta)"""
        if self.is_recording and self.csv_logger and self._is_game_mode():
            # Verificar se não está aguardando resposta
            if self.waiting_for_response:
                print("⚠️  Timeout: Não recebeu resposta CORRECT/WRONG, enviando sinal de fallback")
                # Resetar estado e enviar novo sinal
                self.waiting_for_response = False
                
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True

            self.add_marker(action)
            self._record_pipeline_event(
                "TASK_SENT",
                marker=action,
                source="fallback",
                window_size=self.window_size,
            )

    def send_next_random_signal(self):
        """Envia o próximo sinal aleatório após receber resposta"""
        if self.is_recording and self.csv_logger and self._is_game_mode():
            print("🎲 Enviando próximo sinal aleatório")
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True

            self.add_marker(action)
            self._record_pipeline_event(
                "TASK_SENT",
                marker=action,
                source="random",
                window_size=self.window_size,
            )

    def _start_ai_prediction_window(self, status: str, source: str = ""):
        """
        Abre a janela de inferencia zerando qualquer amostra anterior ao sinal.

        A predicao so pode acontecer depois que window_size amostras novas
        chegarem a partir do marcador enviado para Unity em add_marker().
        """
        generation = self.game_inference.start_window(
            started_at_ms=time.monotonic() * 1000, task_hand=self._current_task_hand
        )
        self._movement_authorization = None
        self._movement_sent.clear()
        self._sync_ai_prediction_state()
        self._record_pipeline_event(
            "AI_WINDOW_OPENED",
            duration_ms=self.ai_window_duration,
            window_size=self.window_size,
            source=source or "random",
        )

        suffix = f" ({source})" if source else ""
        print(f"🤖 Janela de IA aberta por {self.ai_window_duration/1000}s{suffix}")

        self._set_ai_status(status)
        QTimer.singleShot(self.game_inference.collection_deadline_ms,
                          lambda: self.close_ai_window(generation))

    def close_ai_window(self, generation=None):
        """Fecha a janela de IA após o tempo configurado."""
        if not self.game_inference.close_window(generation):
            return
        self._movement_authorization = None
        self._sync_ai_prediction_state()
        self._record_pipeline_event(
            "AI_WINDOW_CLOSED",
            samples_collected=self.samples_since_last_prediction,
        )
        print("🚫 Janela de IA fechada automaticamente")
        
        # Atualizar status visual
        self._set_ai_status("inactive")

    def add_marker(self, marker_type):
        """Adiciona um marcador durante a gravação"""
        if self.is_recording and self.csv_logger:
            current_task = self.task_combo.currentText()
            result = self.marker_controller.register_marker(marker_type, current_task)
            if not result.accepted:
                if result.reason == "baseline_active":
                    QMessageBox.warning(
                        self,
                        "Baseline Ativo",
                        "Não é possível adicionar marcadores durante o baseline",
                    )
                return

            state = result.state
            self._update_marker_labels(state)

            if result.external_signal and self.udp_server_active:
                self.unity_controller.send_action(result.external_signal)
            # A manual cue also replaces the attempt; it never moves the orthosis.
            if self._is_game_mode():
                self._current_task_hand = {"T1": "left", "T2": "right"}.get(result.marker_type)
                self._start_ai_prediction_window("active_window")
                self._arm_game_fallback()

            if USE_OPENBCI_LOGGER:
                # Marcar para adicionar na próxima amostra
                self.pending_marker = marker_type
            else:
                # Logger simples
                marker = self.csv_logger.add_marker(marker_type)
            
            self._apply_recording_ui(hint=marker_type)
            QTimer.singleShot(2000, lambda: self._apply_recording_ui())
    
    def start_baseline(self):
        """Inicia o período de baseline"""
        if self.is_recording and self.csv_logger:
            self._reset_ai_prediction_window()
            self.game_action_timer.stop()
            baseline_state = self.marker_controller.start_baseline(300)
            if USE_OPENBCI_LOGGER:
                # Logger OpenBCI
                if hasattr(self.csv_logger, 'start_baseline'):
                    self.csv_logger.start_baseline()
                else:
                    # Fallback
                    self.csv_logger.add_marker("BASELINE")
            else:
                # Logger simples
                self.csv_logger.add_marker("BASELINE")

            # Iniciar timer visual
            if not self.baseline_timer.isActive():
                self.baseline_timer.start(1000)
            self._update_marker_labels(baseline_state)
             
            # Desabilitar outros botões por 5 minutos
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False) 
            # self.baseline_btn.setEnabled(False)  # Botão removido
            
            self._apply_recording_ui(hint="Baseline")
    
    def update_baseline_timer(self):
        """Atualiza o timer de baseline"""
        result = self.marker_controller.tick_baseline()
        state = result.state
        remaining = state.baseline_remaining_seconds
        self._update_marker_labels(state)

        if remaining > 0:
            minutes = remaining // 60
            seconds = remaining % 60
            self.baseline_label.setText(f"Baseline: {minutes:02d}:{seconds:02d}")
            self._apply_recording_ui(hint=f"{minutes:02d}:{seconds:02d}")
        else:
            # Baseline terminado
            self.baseline_timer.stop()
            self.baseline_label.setText("")
            
            # Reabilitar botões se ainda estiver gravando
            if self.is_recording:
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                # self.baseline_btn.setEnabled(True)  # Botão removido
                self._apply_recording_ui()
                QMessageBox.information(self, "Baseline", "Período de baseline finalizado!")

    def reset_recording_label(self):
        """Mantém compatibilidade com timers antigos."""
        self._apply_recording_ui()
    
    def load_model(self):
        """Carrega modelo CNN para inferência"""
        if self._inference_job is not None:
            return False
        try:
            model = self.inference_controller.load_latest_model()
            self._update_model_status(model)
            print(f"Carregando modelo TensorFlow encontrado: {model.path}")
            return True
        except ValueError as exc:
            self.model_status_label.setText(f"Erro: {exc}")
            QMessageBox.warning(
                self,
                "Erro",
                f"Modelo não encontrado! {exc}",
            )
            return False
        except Exception as e:
            self.model_status_label.setText(f"Erro ao carregar modelo: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao carregar modelo: {e}")
            return False

    def load_model_from_path(self, model_path: str) -> bool:
        """Tenta carregar um modelo explicitamente a partir de um caminho.

        Retorna True se carregado com sucesso, False caso contrário.
        """
        if self._inference_job is not None:
            return False
        try:
            model = self.inference_controller.load_model(model_path)
            self._update_model_status(model)
            self._refresh_streaming_state(loaded_model=model)
            print(f"Modelo TensorFlow carregado: {model_path}")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model_status_label.setText(f"Erro ao carregar modelo: {e}")
            return False

    def find_tf_models(self) -> List[str]:
        """Procura por arquivos .keras/.h5 em locais comuns e retorna caminhos absolutos ordenados por data (mais recente primeiro)."""
        try:
            return [model.path for model in self.inference_controller.list_models()]
        except Exception as e:
            print(f"Erro ao listar modelos TensorFlow: {e}")
            return []

    def _update_model_status(self, model: ModelViewModel):
        self._refresh_streaming_state(loaded_model=model)
        self.model_status_label.setText(f"Modelo carregado: {model.name}")
        expected_time_steps = model.expected_time_steps
        if expected_time_steps is not None and expected_time_steps != self.window_size:
            print(
                f"Aviso: modelo espera {expected_time_steps} timesteps, "
                f"runtime window_size={self.window_size}. Adaptacao sera aplicada na inferencia."
            )
        # Se o modo Livre estava sem modelo, promove para rodando.
        if self._is_free_task_selected() and getattr(self, "free_running", False):
            self._set_ai_status("free_running")
        self._update_rl_status()

    def update_game_stats(self):
        """Atualiza estatísticas do jogo/livre"""
        if self._is_game_mode() or self._is_free_task_selected():
            self._apply_game_stats()
        elif self._is_free_mode():
            self._apply_game_stats()
        
    def process_accuracy_message(self, message):
        """Processa mensagem UDP recebida para cálculo de acurácia"""
        print(f"🔍 DEBUG: Mensagem recebida para acurácia: '{message}'")
        
        if not self._is_game_mode():
            print("🔍 DEBUG: Ignorando mensagem - não está em modo jogo")
            return
            
        try:
            trial = AccuracyPresenter.parse_message(message)
            if trial is None:
                print(f"Mensagem de acurácia ignorada: {message}")
                return

            self.accuracy_trials.append(trial)
            self.update_accuracy_display()

            status = "✓" if trial.is_correct else "✗"
            print(
                f"Acurácia: {trial.expected_action} vs {trial.real_action} {status}"
            )
        except Exception as e:
            print(f"Erro ao processar mensagem de acurácia: {e}")
            
    def update_accuracy_display(self):
        """Atualiza a interface de acurácia"""
        self._apply_accuracy_display()
        
    def reset_accuracy_data(self):
        """Reseta os dados de acurácia"""
        self.accuracy_trials.clear()
        self._apply_accuracy_display()
    
    def reset_action_counters(self):
        """Reseta os contadores de ações T1 e T2"""
        state = self.marker_controller.reset_state()
        self._update_marker_labels(state)
        print("🔄 Contadores de ações resetados")
        
    def start_accuracy_udp_receiver(self):
        """
        Inicia o receptor de acurácia.
        Agora usa o sistema de callbacks do UnityCommunicator.
        """
        print("✅ Sistema de acurácia ativo - usando callbacks do UnityCommunicator")
        # O receptor de mensagens já está ativo através dos callbacks do unity_communicator
        # As mensagens serão processadas automaticamente via _on_unity_message()
        
    def stop_accuracy_udp_receiver(self):
        """Para o UDP receiver de acurácia"""
        print("Sistema de acurácia parado - callbacks mantidos ativos")
        
    def predict_movement(self, eeg_data):
        """Submit at most one prediction; all session decisions stay on Qt."""
        if self._inference_closed or self._inference_job is not None:
            return
        if not self.is_recording or not self._is_game_mode() or not self.inference_controller.has_loaded_model():
            return
        generation = self.game_inference.generation
        if not self.game_inference.claim_prediction(generation):
            return

        try:
            worker = InferenceWorker(self.inference_controller, np.array(eeg_data, copy=True), generation)
            worker.signals.finished.connect(self._on_inference_finished, Qt.QueuedConnection)
            self._inference_job = worker
            self._last_game_window = np.array(eeg_data, copy=True)
            for name in ("btn_jogo", "btn_iniciar_treino"):
                if hasattr(self, name):
                    getattr(self, name).setEnabled(False)
            self._sync_ai_prediction_state()
            QThreadPool.globalInstance().start(worker)
        except Exception as exc:
            self._on_inference_finished(InferenceOutcome(generation, error=str(exc)))

    def predict_free_movement(self, eeg_data):
        """Modo Livre: no max 1 job; nao exige gravacao nem resposta do VR."""
        if getattr(self, "_inference_closed", False):
            return
        if not getattr(self, "free_running", False):
            return
        if not self.eeg_stream_controller.is_running():
            return
        if not self.inference_controller.has_loaded_model():
            return
        if getattr(self, "_free_inference_busy", False):
            return
        try:
            window = np.array(eeg_data, copy=True)
        except Exception:
            return
        try:
            worker = InferenceWorker(self.inference_controller, window,
                                     -int(self.free_inference.windows_emitted))
            worker.signals.finished.connect(self._on_free_inference_finished,
                                            Qt.QueuedConnection)
            self._free_inference_busy = True
            self._last_free_window = np.array(window, copy=True)
            QThreadPool.globalInstance().start(worker)
        except Exception as exc:
            self._free_inference_busy = False
            print(f"[LIVRE] Falha ao enfileirar inferencia: {exc}")

    @pyqtSlot(object)
    def _on_free_inference_finished(self, outcome):
        self._free_inference_busy = False
        if getattr(self, "_inference_closed", False):
            return
        if not getattr(self, "free_running", False):
            return
        if not self._is_free_task_selected():
            return
        if outcome.error is not None:
            self._record_pipeline_event("FREE_PREDICTION_FAILED",
                                        error=outcome.error)
            return
        try:
            prediction = outcome.prediction
            pred = int(prediction.predicted_index)
            conf = float(prediction.confidence)
            runtime_action = UnityCommandMapper.from_prediction(pred)
            timestamp = datetime.now()
            self._apply_prediction_display(prediction)
            self.free_predictions.append((timestamp, pred, conf))
            self.predictions.append((timestamp, pred, conf))
            _fw = getattr(self, "_last_free_window", None)
            if _fw is not None:
                self._rl_push_prediction(_fw, pred, conf)
            self._apply_game_stats()
            self._record_pipeline_event("FREE_PREDICTION_DONE",
                                        predicted_index=pred, confidence=conf,
                                        latency_ms=round(float(outcome.latency_ms), 2))
            # VR/ortese puramente opcionais, com cooldown + limiar.
            if self._free_can_send(conf):
                sent_any = False
                if getattr(self, "free_vr_checkbox", None) is not None and self.free_vr_checkbox.isChecked():
                    if self.send_udp_signal_free(runtime_action.direction):
                        sent_any = True
                if getattr(self, "free_ortese_checkbox", None) is not None and self.free_ortese_checkbox.isChecked():
                    if self.send_esp32_signal_free(runtime_action.direction):
                        sent_any = True
                if sent_any:
                    self._free_mark_sent()
        except Exception as exc:
            self._record_pipeline_event("FREE_PREDICTION_FAILED", error=str(exc))
            print(f"[LIVRE] Erro na predicao: {exc}")

    def send_udp_signal_free(self, direction) -> bool:
        """Envio opcional no modo Livre: so exige VR conectado + checkbox."""
        try:
            if not self.udp_server_active:
                return False
            if getattr(self, "free_vr_checkbox", None) is not None and not self.free_vr_checkbox.isChecked():
                return False
            if self.udp_auto_send_checkbox is not None and not self.udp_auto_send_checkbox.isChecked():
                return False
            return bool(self.unity_controller.send_action(direction))
        except Exception as exc:
            print(f"[LIVRE] Falha ao enviar UDP: {exc}")
            return False

    def send_esp32_signal_free(self, direction) -> bool:
        """Envio opcional no modo Livre: so exige ortese conectada + checkbox."""
        try:
            if not getattr(self, "esp32_connected", False):
                return False
            if getattr(self, "free_ortese_checkbox", None) is not None and not self.free_ortese_checkbox.isChecked():
                return False
            if self.esp32_auto_send_checkbox is not None and not self.esp32_auto_send_checkbox.isChecked():
                return False
            return bool(self.esp32_controller.send_direction(direction))
        except Exception as exc:
            print(f"[LIVRE] Falha ao enviar ESP32: {exc}")
            return False

    @pyqtSlot(object)
    def _on_inference_finished(self, outcome):
        self._inference_job = None
        for name in ("btn_jogo", "btn_iniciar_treino"):
            if hasattr(self, name):
                getattr(self, name).setEnabled(True)
        generation = outcome.generation
        if (self._inference_closed or generation != self.game_inference.generation
                or not self.game_inference.is_window_open
                or not self.is_recording or not self._is_game_mode()):
            return
        if outcome.error is not None:
            self.close_ai_window(generation)
            self._record_pipeline_event("PREDICTION_FAILED", error=outcome.error)
            return
        started = self.game_inference.window_started_at_ms
        if started is not None and time.monotonic() * 1000 - started > self.game_inference.collection_deadline_ms:
            self.close_ai_window(generation)
            return
        try:
            prediction = outcome.prediction
            inference_latency_ms = outcome.latency_ms
            pred = prediction.predicted_index
            runtime_action = UnityCommandMapper.from_prediction(pred)

            # Atualizar interface
            timestamp = datetime.now()
            self._apply_prediction_display(prediction)
            self._movement_authorization = (generation, pred, float(prediction.confidence))
            try:
                unity_success = self.send_udp_signal(runtime_action.direction)
                self.send_esp32_signal(runtime_action.direction)
            finally:
                self._movement_authorization = None

            self.game_inference.mark_prediction_used()
            self._sync_ai_prediction_state()
            self._record_pipeline_event(
                "PREDICTION_DONE",
                predicted_index=pred,
                confidence=float(prediction.confidence),
                latency_ms=round(inference_latency_ms, 2),
                samples=self.window_size,
            )
            self._record_pipeline_event(
                "UNITY_COMMAND_SENT",
                direction=runtime_action.direction,
                success=bool(unity_success),
            )
            
            # Salvar predição
            self.predictions.append((timestamp, pred, float(prediction.confidence)))
            _gw = getattr(self, "_last_game_window", None)
            if _gw is not None:
                self._rl_push_prediction(_gw, pred, float(prediction.confidence))
            
        except Exception as e:
            self.close_ai_window(generation)
            self._record_pipeline_event("PREDICTION_FAILED", error=str(e))
            print(f"Erro na predição: {e}")
    
    def on_data_received(self, data):
        """Callback para dados recebidos"""
        # Confirmar conexão do EEG no primeiro dado recebido
        if self.eeg_connection_phase == "connecting":
            if self.eeg_stream_controller.is_mock_mode():
                self.eeg_connection_phase = "mock"
            else:
                self.eeg_connection_phase = "connected"
            
            self.record_button_enabled = True
            self._apply_connection_panel()
            self._refresh_streaming_state()
            print(f"[EEG] Conexão confirmada: Primeiro dado recebido ({len(data)} canais)")

        # Filter only a visual copy; IA and logger must receive raw samples.
        if not hasattr(self, 'plot_filter'):
            self.plot_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=6)
        self.plot_widget.add_data(self.plot_filter.apply_realtime_filter(np.array(data, copy=True)))
        current_time_seconds = time.monotonic()
        current_time_ms = current_time_seconds * 1000
        try:
            self.pipeline_telemetry.observe_eeg_sample(
                now_seconds=current_time_seconds,
            )
        except Exception as exc:
            print(f"[PIPELINE] Falha ao medir taxa EEG: {exc}")

        # Calibracao EA (nao supervisionada, ~30 s): enquanto um modelo com
        # Euclidean Alignment nao tiver referencia do sujeito, a amostra
        # alimenta o calibrador e as predicoes aguardam (EEG/gravacao seguem).
        ea_waiting = self._ea_feed_sample(data)
        # Adicionar ao buffer de dados e verificar predição
        if ea_waiting:
            pass
        elif self._is_game_mode():
            sample_result = self.game_inference.add_sample(
                data,
                now_ms=current_time_ms,
            )
            self._sync_ai_prediction_state()

            if sample_result.status == GameInferenceCoordinator.STATUS_EXPIRED:
                self._record_pipeline_event(
                    "AI_WINDOW_EXPIRED",
                    samples_collected=sample_result.samples_collected,
                    duration_ms=self.ai_window_duration,
                )
                print(f"🚫 Janela de IA fechada após {self.ai_window_duration/1000}s")
                self.close_ai_window()
            elif sample_result.status == GameInferenceCoordinator.STATUS_READY:
                elapsed_ms = None
                if self.task_start_time is not None:
                    elapsed_ms = round(current_time_ms - self.task_start_time, 2)
                self._record_pipeline_event(
                    "SAMPLE_250_READY",
                    samples=self.window_size,
                    elapsed_ms=elapsed_ms,
                    eeg_rate_hz=round(
                        self.pipeline_telemetry.sample_rate.latest_rate_hz,
                        2,
                    ),
                )
                quality_result = self.eeg_quality_validator.validate(
                    sample_result.window
                )
                if quality_result.accepted:
                    self.predict_movement(np.array(sample_result.window))
                else:
                    self._record_pipeline_event(
                        "WINDOW_REJECTED",
                        reason=quality_result.reason,
                        samples=self.window_size,
                    )
                    print(
                        f"Janela EEG rejeitada antes da IA: {quality_result.reason}"
                    )
                    self.game_inference.mark_prediction_used()
                    self._sync_ai_prediction_state()
        elif self._is_free_task_selected() and getattr(self, "free_running", False):
            # Modo Livre: loop continuo, sem depender de resposta do VR.
            try:
                free_result = self.free_inference.add_sample(data)
            except Exception as exc:
                print(f"[LIVRE] Falha no buffer: {exc}")
                free_result = None
            if free_result is not None and free_result.status == FreeRunInferenceCoordinator.STATUS_READY:
                quality = self.eeg_quality_validator.validate(free_result.window)
                if quality.accepted:
                    self.predict_free_movement(np.array(free_result.window))
                else:
                    self._record_pipeline_event("FREE_WINDOW_REJECTED",
                                                reason=quality.reason)
                
        # Enviar para logger se estiver gravando
        if self.is_recording and self.csv_logger:
            if USE_OPENBCI_LOGGER and hasattr(self.csv_logger, 'log_sample'):
                # Logger OpenBCI - verificar marcador pendente
                marker = self.pending_marker
                self.pending_marker = None  # Limpar marcador pendente
                
                # Garantir que temos 'channels' canais
                if len(data) == self.channels:
                    self.csv_logger.log_sample(data, marker)
                else:
                    # Ajustar dados se necessário
                    if len(data) >= self.channels:
                        eeg_data = data[:self.channels]
                    else:
                        eeg_data = data + [0.0] * (self.channels - len(data))
                    self.csv_logger.log_sample(eeg_data, marker)
            else:
                # Logger simples (fallback)
                self.csv_logger.log_data(data)
    
    def on_connection_status(self, connected):
        """Callback para status da conexão - agora aguarda o primeiro dado"""
        if connected:
            # Se for mock, podemos considerar conectado imediatamente se quiser,
            # mas vamos manter a lógica de esperar dado para ambos para consistência
            # ou apenas setar como 'connecting' para o real.
            if self.eeg_stream_controller.is_mock_mode():
                # No modo mock, como os dados são gerados localmente, 
                # podemos considerar 'mock' já ou esperar o primeiro dado.
                # Vamos esperar o dado para garantir que o loop está rodando.
                if self.eeg_connection_phase != "mock":
                    self.eeg_connection_phase = "connecting"
            else:
                self.eeg_connection_phase = "connecting"
            
            # Não habilitamos record_btn aqui, esperamos on_data_received
        else:
            self._reset_ai_prediction_window()
            try:
                self.inference_controller.ea_reset()
            except Exception:
                pass
            self.game_action_timer.stop()
            self.eeg_connection_phase = (
                "standby" if self.disconnection_in_progress else "failed"
            )
            self.record_button_enabled = False
            self.disconnection_in_progress = False

        self.connect_button_enabled = True
        self._apply_connection_panel()
        self._refresh_streaming_state()

    def stop_streaming(self):
        """Stops the EEG stream if it is running."""
        self._reset_ai_prediction_window()
        try:
            self._stop_free_run()
        except Exception:
            pass
        self._set_ai_status("stopped")
        self.game_action_timer.stop()
        self.eeg_stream_controller.disconnect()

    def closeEvent(self, event):
        # Logical cancellation only. The global pool outlives this widget;
        # Qt disconnects its receiver automatically if the parent destroys it.
        self._inference_closed = True
        self._rl_closed = True
        self._reset_ai_prediction_window()
        try:
            self._stop_free_run()
        except Exception:
            pass
        self.game_action_timer.stop()
        super().closeEvent(event)
    
    def update_session_timer(self):
        """Atualiza o display do timer de sessão"""
        started_at = self._get_session_started_at()
        if started_at is not None:
            # Calcular tempo decorrido
            elapsed = int(time.time() - float(started_at))
        else:
            elapsed = 0
        
        # Formatar tempo como HH:MM:SS
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        if self.is_recording:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet(Theme.status_text("connected") + " font-size: 20px;")
        else:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet(Theme.status_text("off") + " font-size: 20px;")
    
    def on_task_changed(self):
        """Callback chamado quando a tarefa é alterada"""
        task = self.task_combo.currentText()
        self._reset_ai_prediction_window()

        # Resetar contadores de ações sempre que mudar de tarefa
        self.reset_action_counters()
        self._apply_task_view_state()

        # Modo Livre: liga o loop ao selecionar, mesmo sem gravar/modelo.
        if task == "Livre":
            if not getattr(self, "free_running", False):
                self._start_free_run()
            else:
                # Atualiza status caso o modelo tenha sido carregado depois.
                if self.inference_controller.has_loaded_model():
                    self._set_ai_status("free_running")
            self._apply_recording_ui(hint="Livre ao vivo")
            return
        else:
            if getattr(self, "free_running", False):
                self._stop_free_run()

        if not self.is_recording:
            if task == "Jogo":
                if not self.inference_controller.has_loaded_model():
                    try:
                        candidates = self.find_tf_models()
                    except Exception:
                        candidates = []

                    if candidates:
                        items = [os.path.basename(p) for p in candidates]
                        item, ok = QInputDialog.getItem(self, "Selecionar Modelo", "Modelos TensorFlow encontrados:", items, 0, False)
                        if ok and item:
                            sel_index = items.index(item)
                            sel_path = candidates[sel_index]
                            loaded = self.load_model_from_path(sel_path)
                            if loaded:
                                QMessageBox.information(self, "Modelo carregado", f"Modelo carregado: {sel_path}")
                            else:
                                QMessageBox.warning(self, "Falha ao carregar", f"Falha ao carregar o modelo selecionado: {sel_path}")
                        else:
                            QMessageBox.information(self, "Nenhum modelo selecionado", "Nenhum modelo foi selecionado. Você pode treinar um modelo ou colocar um arquivo .keras em bci/models.")
                    else:
                        QMessageBox.warning(
                            self,
                            "Modelo não encontrado",
                            "Nenhum modelo TensorFlow (.keras/.h5) foi encontrado nos diretórios configurados.\n"
                            "Coloque um arquivo .keras em um diretório de modelos conhecido ou treine um modelo pela interface."
                        )
    
    def update_record_button_text(self):
        """Atualiza o texto do botão de gravação baseado no estado e tarefa"""
        self._apply_recording_ui()
    
    def _on_unity_message(self, message: str):
        # Marshal network callbacks to Qt and retain the generation at receipt.
        self.unity_message_signal.emit(message, self.game_inference.generation)

    def _process_unity_message(self, message: str, generation: int):
        """Callback para mensagens recebidas do Unity"""
        if generation != self.game_inference.generation:
            return
        # Wire responses have no trial ID. Generation rejects queued local work,
        # but cannot identify a delayed packet first received in a later trial.
        print(f"[Unity] Mensagem recebida: {message}")
        
        # Verificar se recebeu resposta CORRECT ou WRONG
        if "CORRECT" in message or "WRONG" in message:
            print(f"✅ Resposta recebida: {message}")
            self._record_pipeline_event(
                "UNITY_RESPONSE",
                message=message,
                waiting_for_response=bool(self.waiting_for_response),
            )
            # Alimenta a acurácia (1 tentativa por CORRECT/WRONG).
            try:
                self.accuracy_message_signal.emit(message)
            except Exception:
                pass
            # Feedback do jogo vale como recompensa p/ a rede (RL).
            # So aceita quando ha resposta esperada (evita rotular trial errado
            # com pacote atrasado de trial anterior).
            try:
                if self.waiting_for_response:
                    if "CORRECT" in message:
                        self.rl_feedback(True, from_vr=True)
                    elif "WRONG" in message:
                        self.rl_feedback(False, from_vr=True)
            except Exception as exc:
                print(f"[RL] Falha no feedback do VR: {exc}")
            if self.is_recording and self._is_game_mode() and self.waiting_for_response and self.game_inference.prediction_locked:
                self.waiting_for_response = False
                self.close_ai_window(self.game_inference.generation)
                print("🔓 Liberado para enviar próximo sinal aleatório")
                # Aguardar 7 segundos antes do próximo sinal
                self.game_action_timer.stop()
                self._schedule_game_callback(7000, self.send_next_random_signal)
        
        # Processar mensagens específicas do Unity
        if "FLOWER" in message:
            # Usar o signal existente para processar mensagens de acurácia
            self.accuracy_message_signal.emit(message)
        elif "CONNECTED" in message:
            print("[Unity] Confirmação de conexão recebida")
        elif "STATUS" in message:
            print(f"[Unity] Status: {message}")
    
    def _on_unity_connection(self, connected: bool):
        """Callback para mudanças no status de conexão com Unity"""
        if connected:
            self.unity_connection_phase = "connected"
            print("[Unity] TCP conectado")
        else:
            if not self.udp_server_active:
                self.unity_connection_phase = "standby"
            print("[Unity] TCP desconectado")
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def show_training_dialog(self, csv_file_path, patient_id, patient_name, auto_start: bool = False):
        """Mostra o diálogo de confirmação e execução do treino"""
        if self._inference_job is not None:
            self._apply_recording_ui(hint="Aguarde a inferencia antes de treinar")
            return
        try:
            dialog = TrainingDialog(
                self.training_controller,
                csv_file_path,
                int(patient_id),
                patient_name,
                auto_load_model=auto_start,
                parent=self,
            )
            dialog.training_progress_signal.connect(self._on_training_progress)
            dialog.training_finished_signal.connect(self._on_training_finished)
            dialog.model_ready_signal.connect(self._on_trained_model_ready)

            if auto_start:
                self._apply_recording_ui(hint="Treino")
                dialog.start_training()
                dialog.show()
                try:
                    dialog.raise_()
                    dialog.activateWindow()
                except Exception:
                    pass
                return

            result = dialog.exec_()
            if result == QDialog.Accepted:
                print(f"Iniciando treino para paciente {patient_name} com arquivo {csv_file_path}")
            else:
                QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
                
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao abrir diálogo de treino: {e}")
            QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")

    def _on_training_progress(self, message: str):
        if hasattr(self, "gravacao_status"):
            short = (message[:24] + "…") if len(message) > 24 else message
            self.gravacao_status.setText(f"Treino · {short}")

    def _on_training_finished(self, success: bool, _message: str):
        self._refresh_streaming_state()
        self._apply_recording_ui()

    def _on_trained_model_ready(self, model_path: str):
        loaded_model = self.inference_controller.get_loaded_model()
        if loaded_model is not None:
            self._update_model_status(loaded_model)
        else:
            self._refresh_streaming_state()
            self.model_status_label.setText(
                f"Modelo treinado pronto: {os.path.basename(model_path)}"
            )
