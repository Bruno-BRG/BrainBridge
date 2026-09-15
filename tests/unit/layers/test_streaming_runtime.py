"""Behavioral Qt tests: no sockets, serial ports, databases or model runtime."""
import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
from threading import Event
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import sip
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer, QThread, QThreadPool, QCoreApplication, QEvent
from PyQt5.QtTest import QTest
from brainbridge_v2.presentation.gui.widgets import streaming
from brainbridge_v2.application.game_inference_coordinator import GameInferenceCoordinator
from brainbridge_v2.application.eeg_quality import EEGWindowQualityValidator
from brainbridge_v2.interface_adapters.controllers.marker_controller import MarkerController
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import InMemoryMarkerStateStore


@pytest.fixture
def widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = streaming.StreamingWidget.__new__(streaming.StreamingWidget)
    QWidget.__init__(w)
    w._inference_job = None
    w._inference_closed = False
    w.game_inference = GameInferenceCoordinator(window_size=2)
    w.window_size, w.channels, w.ai_window_duration = 2, 16, 2000
    w.game_action_interval = 10000
    w.is_recording = True
    w.session_affected_hand = "left"
    w._movement_authorization = None
    w._movement_sent = set()
    w.streaming_state = SimpleNamespace(game_mode=True)
    w.csv_logger = Mock()
    w.unity_controller = Mock()
    w.esp32_controller = Mock()
    w.inference_controller = Mock()
    w.inference_controller.predict.return_value = SimpleNamespace(predicted_index=0, confidence=0.9)
    w.udp_server_active = w.esp32_connected = True
    w.udp_auto_send_checkbox = w.esp32_auto_send_checkbox = Mock()
    w.predictions = deque()
    w.task_combo = Mock()
    w.task_combo.currentText.return_value = "Jogo"
    w.marker_controller = MarkerController.from_store(InMemoryMarkerStateStore())
    w.game_action_timer = Mock()
    w.waiting_for_response = True
    w.pipeline_telemetry = Mock()
    w.eeg_quality_validator = EEGWindowQualityValidator()
    w.eeg_connection_phase = "connected"
    w.plot_widget = Mock()
    w.pipeline_telemetry.sample_rate.latest_rate_hz = 125
    for method in ("_apply_prediction_display", "_set_ai_status", "_record_pipeline_event",
                   "_update_marker_labels", "_apply_recording_ui"):
        setattr(w, method, Mock())
    callbacks = []
    monkeypatch.setattr(streaming.QTimer, "singleShot", lambda ms, fn: callbacks.append((ms, fn)))
    monkeypatch.setattr(streaming.QMessageBox, "warning", Mock())
    w.callbacks = callbacks
    yield w
    QThreadPool.globalInstance().waitForDone(3000)
    app.processEvents()
    if not sip.isdeleted(w):
        w.deleteLater()
    app.processEvents()


def ready(w, marker):
    w.add_marker(marker)
    w.game_inference.add_sample(range(16))
    w.game_inference.add_sample(range(16))


def wait_until(predicate):
    deadline = monotonic() + 3
    while not predicate() and monotonic() < deadline:
        QTest.qWait(5)
    assert predicate()


def finish(w):
    wait_until(lambda: w._inference_job is None)


@pytest.mark.parametrize("hand,marker,index,count", [
    ("left", "T1", 0, 1), ("right", "T2", 1, 1),
    (None, "T1", 0, 0), ("left", "T2", 0, 0),
    ("right", "T1", 1, 0), ("left", "T1", 1, 0),
    ("right", "T2", 0, 0), ("left", "T1", 0.5, 0),
])
def test_gate_and_duplicate_prediction(widget, hand, marker, index, count):
    w = widget
    w.session_affected_hand = hand
    w.inference_controller.predict.return_value.predicted_index = index
    ready(w, marker)
    w.unity_controller.send_action.assert_called_once_with("trigger_left" if marker == "T1" else "trigger_right")
    w.esp32_controller.send_direction.assert_not_called()
    w.unity_controller.reset_mock()
    w.predict_movement([[0] * 16] * 2)
    w.predict_movement([[0] * 16] * 2)
    finish(w)
    assert w.unity_controller.send_action.call_count == count
    assert w.esp32_controller.send_direction.call_count == count
    assert w.inference_controller.predict.call_count == 1
    if count:
        direction = "esquerda" if hand == "left" else "direita"
        w.unity_controller.send_action.assert_called_once_with(direction)
        w.esp32_controller.send_direction.assert_called_once_with(direction)


def test_old_timeout_and_result_cannot_touch_new_attempt(widget):
    w = widget
    ready(w, "T1")
    old_timeout = w.callbacks[0][1]
    release = Event()
    def predict(_data):
        assert release.wait(3)
        return SimpleNamespace(predicted_index=0, confidence=0.9)
    w.inference_controller.predict.side_effect = predict
    w.predict_movement([])
    w._reset_ai_prediction_window()
    ready(w, "T2")
    w.predict_movement([])
    old_timeout()
    release.set()
    finish(w)
    assert w.inference_controller.predict.call_count == 1
    assert w.game_inference.is_window_open
    assert not w.game_inference.prediction_locked
    w.esp32_controller.send_direction.assert_not_called()
    assert not w.predictions


def test_stop_invalidates_scheduled_next_attempt(widget):
    w = widget
    callback = Mock()
    w._schedule_game_callback(7000, callback)
    scheduled = w.callbacks[-1][1]
    w.eeg_stream_controller = Mock()
    w.stop_streaming()
    ready(w, "T2")
    scheduled()
    callback.assert_not_called()


def test_old_fallback_signal_cannot_replace_new_window(widget):
    w = widget
    ready(w, "T1")
    old_callback = w.game_action_timer.timeout.connect.call_args.args[0]
    ready(w, "T2")
    generation = w.game_inference.generation
    old_callback()
    assert w.game_inference.generation == generation
    assert w.game_inference.task_hand == "right"


def test_manual_and_debug_methods_cannot_bypass_gate(widget):
    w = widget
    for method in (w.manual_esp32_test, w.manual_udp_test, w.send_udp_signal, w.send_esp32_signal):
        assert method("esquerda") is False
    w.unity_controller.send_action.assert_not_called()
    w.esp32_controller.send_direction.assert_not_called()


def test_timeout_closes_without_serial_and_other_hand_is_recorded(widget):
    w = widget
    w.add_marker("T2")
    w.on_data_received(list(range(16)))
    w.csv_logger.log_sample.assert_called_once_with(list(range(16)), "T2")
    w.callbacks[0][1]()
    w.predict_movement([])
    w.inference_controller.predict.assert_not_called()
    w.esp32_controller.send_direction.assert_not_called()


def test_duplicate_response_schedules_once_and_stale_receipt_is_ignored(widget):
    w = widget
    ready(w, "T1")
    w.predict_movement([])
    finish(w)
    generation = w.game_inference.generation
    w._process_unity_message("CORRECT", generation)
    w._process_unity_message("CORRECT", generation)
    assert len([delay for delay, _ in w.callbacks if delay == 7000]) == 1
    ready(w, "T2")
    w._process_unity_message("CORRECT", generation)
    assert w.game_inference.is_window_open


def test_session_start_reads_updated_registration_and_blocks_legacy(widget):
    w = widget
    w.is_recording = False
    w.patient_combo = Mock()
    w.patient_combo.currentIndex.return_value = 1
    w.patient_combo.currentData.return_value = 7
    w.patient_combo.currentText.return_value = "Paciente (ID: 7)"
    w.patient_controller = Mock()
    w.patient_controller.list_patients.return_value = [{"id": 7, "affected_hand": None}]
    w.toggle_recording()
    assert not w.is_recording
    streaming.QMessageBox.warning.assert_called_once()
    # Stop before any logger/session side effect, after the refreshed hand gate.
    w.patient_controller.list_patients.return_value = [{"id": 7, "affected_hand": "right"}]
    w.inference_controller.has_loaded_model.return_value = False
    w.load_model = Mock(return_value=False)
    # Paciente ja calibrado: o gate de calibracao deixa passar ate o load_model.
    w.training_controller = Mock()
    w.training_controller.patient_model_available.return_value = True
    w.toggle_recording()
    assert w.session_affected_hand == "right"
    w.load_model.assert_called_once()


@pytest.mark.parametrize("invalidate", ["stop", "timeout", "deadline", "hand", "mode", "close"])
def test_delayed_prediction_keeps_gui_live_and_revalidates(widget, invalidate):
    w = widget
    entered, release = Event(), Event()
    threads, ticks = [], []
    def predict(_data):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(predicted_index=0, confidence=0.9)
    w.inference_controller.predict.side_effect = predict
    ready(w, "T1")
    w.unity_controller.reset_mock()
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    try:
        w.predict_movement([])
        wait_until(entered.is_set)
        wait_until(lambda: len(ticks) >= 4)
        assert threads[0] != QApplication.instance().thread()
        assert w._inference_job is not None
        assert not w.load_model()
        assert not w.load_model_from_path("unused.keras")
        w.show_training_dialog("unused.csv", 1, "test")
        w.inference_controller.load_model.assert_not_called()
        w.inference_controller.load_latest_model.assert_not_called()
        if invalidate == "stop":
            w.eeg_stream_controller = Mock()
            w.stop_streaming()
        elif invalidate == "timeout":
            w.close_ai_window()
        elif invalidate == "deadline":
            w.game_inference.window_started_at_ms -= 10000
        elif invalidate == "hand":
            w.session_affected_hand = "right"
        elif invalidate == "mode":
            w.streaming_state.game_mode = False
        else:
            w.close()
        assert not release.is_set()
    finally:
        timer.stop()
        release.set()
        finish(w)
    w.unity_controller.send_action.assert_not_called()
    w.esp32_controller.send_direction.assert_not_called()


def test_failure_finishes_job_on_gui_and_next_trial_can_predict(widget):
    w = widget
    threads = []
    w._apply_prediction_display.side_effect = lambda _: threads.append(QThread.currentThread())
    w.inference_controller.predict.side_effect = RuntimeError("failed")
    ready(w, "T1")
    w.predict_movement([])
    finish(w)
    assert not w.game_inference.is_window_open
    w._set_ai_status.assert_called_with("inactive")
    w.esp32_controller.send_direction.assert_not_called()
    w.inference_controller.predict.side_effect = None
    ready(w, "T1")
    w.predict_movement([])
    finish(w)
    assert threads == [QApplication.instance().thread()]
    w.esp32_controller.send_direction.assert_called_once_with("esquerda")


def test_submission_failure_releases_job_and_closes_window(widget, monkeypatch):
    ready(widget, "T1")
    pool = Mock()
    pool.start.side_effect = RuntimeError("pool unavailable")
    monkeypatch.setattr(streaming.QThreadPool, "globalInstance", lambda: pool)
    widget.predict_movement([])
    assert widget._inference_job is None
    assert not widget.game_inference.is_window_open
    widget._set_ai_status.assert_called_with("inactive")
    widget.inference_controller.predict.assert_not_called()


def test_worker_owns_submission_window(widget):
    entered, release = Event(), Event()
    observed = []
    def predict(data):
        entered.set()
        assert release.wait(3)
        observed.append(data.tolist())
        return SimpleNamespace(predicted_index=0, confidence=0.9)
    widget.inference_controller.predict.side_effect = predict
    ready(widget, "T1")
    data = [[1.0] * 16] * 2
    widget.predict_movement(data)
    try:
        wait_until(entered.is_set)
        data[0][0] = 999
    finally:
        release.set()
        finish(widget)
    assert observed[0] == [[1.0] * 16] * 2


def test_destroyed_parent_disconnects_running_worker(widget):
    w = widget
    parent = QWidget()
    w.setParent(parent)
    entered, release = Event(), Event()
    def predict(_data):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(predicted_index=0, confidence=0.9)
    w.inference_controller.predict.side_effect = predict
    ready(w, "T1")
    w.predict_movement([])
    wait_until(entered.is_set)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    release.set()
    assert QThreadPool.globalInstance().waitForDone(3000)
    QApplication.instance().processEvents()
    w._apply_prediction_display.assert_not_called()
    w.esp32_controller.send_direction.assert_not_called()
