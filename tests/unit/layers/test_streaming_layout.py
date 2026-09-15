"""Real Qt layout/rendering with synthetic EEG and no external services."""
import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from brainbridge_v2.presentation.gui.styles import Theme
from brainbridge_v2.presentation.gui.widgets.streaming import StreamingWidget
from brainbridge_v2.interface_adapters.controllers.marker_controller import MarkerController
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import InMemoryMarkerStateStore


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    controllers = {name: Mock() for name in (
        "eeg_stream_controller", "inference_controller", "training_controller",
        "patient_controller", "recording_controller", "session_controller",
        "unity_controller", "esp32_controller")}
    controllers["patient_controller"].list_patients.return_value = [
        {"id": 1, "name": "Paciente de demonstracao com nome extenso", "affected_hand": "left"}]
    controllers["session_controller"].get_current_session.return_value = None
    controllers["inference_controller"].get_loaded_model.return_value = None
    controllers["eeg_stream_controller"].is_running.return_value = False
    controllers["eeg_stream_controller"].is_mock_mode.return_value = False
    w = StreamingWidget(**controllers, marker_controller=MarkerController.from_store(InMemoryMarkerStateStore()))
    w.setStyleSheet(Theme.get_stylesheet())
    yield app, w, controllers
    w.stats_timer.stop()
    w.plot_widget._backend.timer.stop()
    w.close()
    w.deleteLater()
    app.processEvents()


def _key_controls(w):
    return [
        w.btn_baseline, w.btn_jogo, w.btn_livre, w.btn_iniciar_treino,
        w.btn_calib_esq, w.btn_calib_dir, w.connect_btn,
        w.developer_settings_btn, w.connect_eeg_btn, w.connect_vr_btn,
        w.connect_ortese_btn, w.patient_combo, w.refresh_patients_btn,
        w.record_btn, w.free_model_btn, w.free_vr_checkbox,
        w.free_ortese_checkbox, w.rl_correct_btn, w.rl_wrong_btn,
        w.rl_restore_btn, w.t1_btn, w.t2_btn, w.free_result_label,
        w.prediction_label, w.accuracy_label, w.model_status_label,
        w.marcador_text, w.status_eeg, w.status_vr, w.status_ortese,
        w.gravacao_status, w.session_timer_label, w.patient_display_label,
        w.ia_live_label, w.rl_status_label, w.ai_status_label,
    ]


def _settle(app, w, size):
    # Offscreen may shrink top-level to sizeHint on extra event passes;
    # let geometry settle, then verify against the settled window.
    w.resize(*size)
    w.show()
    for _ in range(3):
        app.processEvents()
    return w.width(), w.height()


@pytest.mark.parametrize("size", [(1400, 850), (1024, 640), (800, 540)])
def test_no_scroll_all_controls_visible(panel, size):
    """Novo design: sem QScrollArea; tudo visivel e sem corte."""
    app, w, _controllers = panel
    width, height = _settle(app, w, size)
    assert not hasattr(w, "controls_scroll")
    # Garantia real de "sem corte": o minimo do layout cabe em 800x540.
    minimum = w.layout().totalMinimumSize()
    assert minimum.width() <= 800
    assert minimum.height() <= 540
    # Plot domina a janela assentada.
    assert w.plot_widget.width() > width * 0.55
    assert w.plot_widget.height() > height * 0.55
    for control in _key_controls(w):
        assert control.isVisibleTo(w), control
        assert control.visibleRegion().boundingRect() == control.rect(), control


def test_task_switch_keeps_layout_stable(panel):
    """Trocar de tarefa nao embaralha o layout (sem dialogs no caminho)."""
    app, w, controllers = panel
    controllers["inference_controller"].get_loaded_model.return_value = Mock(
        name="generalized_mi_multidataset_v3_eegnet.keras")
    _settle(app, w, (1400, 850))
    # Primeiro clique absorve o assentamento inicial do layout offscreen.
    w.btn_baseline.click()
    app.processEvents()
    app.processEvents()
    plot_size = w.plot_widget.size()
    for task, button in (("Jogo", w.btn_jogo), ("Treino", w.btn_iniciar_treino),
                         ("Baseline", w.btn_baseline), ("Livre", w.btn_livre)):
        button.click()
        app.processEvents()
        assert w.task_combo.currentText() == task
        assert button.isChecked()
        assert w.plot_widget.size() == plot_size
    w.workspace_splitter.setSizes([1000, 300])
    app.processEvents()
    assert w.plot_widget.isVisible()


def test_real_offscreen_capture(panel):
    app, w, _controllers = panel
    w.resize(1400, 850)
    w.show()
    for t in np.arange(1000) / 125:
        w.plot_widget.add_data(20 * np.sin(2 * np.pi * (8 + np.arange(16) / 4) * t))
    w.plot_widget._backend._flush_plot()
    app.processEvents()
    path = os.environ.get("BRAINBRIDGE_UI_SCREENSHOT")
    if path:
        assert w.grab().save(path)
