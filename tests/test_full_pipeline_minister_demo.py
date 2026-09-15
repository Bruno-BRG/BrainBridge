"""
Pipeline completa BrainBridge: paciente -> gravacao/sessao -> VR -> jogo/IA -> ortese -> acuracia -> finalizacao.

Roteiro de demonstracao (sem GUI, sem hardware, sem TensorFlow real):
1. Cria paciente valido (+ rejeita invalidos) e atualiza mao afetada.
2. Abre gravacao + sessao em modo jogo, registra marcadores T1/T2.
3. Sessao VR com dados reais -> HEADER -> Confirm -> trigger -> comandos.
4. Rodada do jogo: cue -> janela 250x16 -> qualidade -> preprocess -> inferencia (fake) ->
   mapper -> autorizacao (allows_movement) -> Unity + ESP32 -> resposta VR -> acuracia.
5. Finaliza: END_TASK/END_SESSION + confirm_end, encerra sessao e gravacao.

Tudo com fakes apenas onde ha hardware/modelo real (serial, TF).
Falha em qualquer etapa = falha do teste.
"""
import shutil
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from brainbridge_v2.application.eeg_quality import EEGWindowQualityValidator
from brainbridge_v2.application.game_inference_coordinator import GameInferenceCoordinator
from brainbridge_v2.application.pipeline_telemetry import PipelineTelemetry
from brainbridge_v2.application.unity_command_mapper import UnityCommandMapper
from brainbridge_v2.application.use_cases.inference_use_cases import RunInferenceUseCase
from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult
from brainbridge_v2.domain.entities.session import Session
from brainbridge_v2.infrastructure.communication.unity import (
    PatientData,
    SessionPhase,
    ServerState,
    TaskType,
    UnityCommunicator,
)
from brainbridge_v2.infrastructure.communication.unity_gateway_adapter import UnityGatewayAdapter
from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.ml.eeg_pipeline import preprocess_window
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import SQLitePatientRepository
from brainbridge_v2.infrastructure.repositories.sqlite_recording_repository import SQLiteRecordingRepository
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import InMemoryMarkerStateStore
from brainbridge_v2.infrastructure.state.in_memory_session_store import InMemorySessionStore
from brainbridge_v2.interface_adapters.controllers.esp32_controller import ESP32Controller
from brainbridge_v2.interface_adapters.controllers.marker_controller import MarkerController
from brainbridge_v2.interface_adapters.controllers.patient_controller import PatientController
from brainbridge_v2.interface_adapters.controllers.recording_controller import RecordingController
from brainbridge_v2.interface_adapters.controllers.session_controller import SessionController
from brainbridge_v2.interface_adapters.controllers.unity_controller import UnityController
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    AccuracyPresenter,
    StartRecordingRequest,
    StartSessionRequest,
)


class FakeInferenceGateway:
    """Simula TF: sempre preve a mao da rodada com alta confianca."""

    def __init__(self, forced_index=0):
        self.forced_index = int(forced_index)
        self.calls = 0

    def predict(self, eeg_window):
        self.calls += 1
        probs = [0.0, 0.0]
        probs[self.forced_index] = 0.92
        probs[1 - self.forced_index] = 0.08
        return PredictionResult(
            predicted_index=self.forced_index,
            confidence=0.92,
            probabilities=probs,
        )


class FakeESP32Gateway:
    def __init__(self):
        self.connected = True
        self.sent = []

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def send_direction(self, direction):
        if direction not in ("esquerda", "direita"):
            raise ValueError(f"Direcao ESP32 invalida: {direction}")
        if not self.connected:
            return False
        self.sent.append(direction)
        return True

    def is_connected(self):
        return self.connected

    def set_connection_callback(self, callback):
        pass


class FakeUnityGateway:
    """Unity em memoria para a rodada do jogo (sem socket)."""

    def __init__(self):
        self.actions = []
        self.session = None

    def start_server(self):
        return True

    def stop_server(self):
        pass

    def start_session(self, nome, nivel, lado, tarefa, sessoes=0):
        self.session = (nome, nivel, lado, tarefa, sessoes)
        return True

    def set_pending_session(self, nome, nivel, lado, tarefa, sessoes=0):
        self.session = (nome, nivel, lado, tarefa, sessoes)

    def send_action(self, action):
        self.actions.append(action)
        return True

    def send_trigger(self):
        self.actions.append("Trigger")
        return True

    def end_task(self):
        return True

    def end_session(self, message):
        return True

    def is_server_active(self):
        return True

    def is_client_connected(self):
        return True

    def set_message_callback(self, callback):
        pass

    def set_connection_callback(self, callback):
        pass


def _cleanup_unity():
    try:
        comm = UnityCommunicator()
        if comm.server_state != ServerState.STOPPED:
            comm.stop_server()
        time.sleep(0.15)
    except Exception:
        pass
    UnityCommunicator._instance = None
    time.sleep(0.1)


def _make_controllers(tmp_dir):
    db = DatabaseManager(db_path=str(Path(tmp_dir) / "demo.db"))
    patient_controller = PatientController.from_repository(SQLitePatientRepository(db))
    recording_controller = RecordingController.from_repository(SQLiteRecordingRepository(db))
    session_controller = SessionController.from_store(InMemorySessionStore())
    marker_controller = MarkerController.from_store(InMemoryMarkerStateStore())
    return db, patient_controller, recording_controller, session_controller, marker_controller


def _synthetic_window(n=250, ch=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 20, size=(n, ch)).astype(np.float64)


def test_01_patient_lifecycle():
    td = tempfile.mkdtemp(prefix="bb_demo_")
    try:
        _, patient_controller, _, _, _ = _make_controllers(td)
        pid = patient_controller.register_patient({
            "name": "Maria Demo", "age": 62, "sex": "F",
            "affected_hand": "left", "time_since_event": 6, "notes": "demo",
        })
        assert pid > 0
        patients = patient_controller.list_patients()
        assert any(p["id"] == pid and p["name"] == "Maria Demo" for p in patients)
        patient_controller.update_patient_affected_hand(pid, "right")
        updated = next(p for p in patient_controller.list_patients() if p["id"] == pid)
        assert updated["affected_hand"] == "right"
        with pytest.raises((ValueError, TypeError)):
            patient_controller.register_patient({
                "name": "   ", "age": 60, "sex": "F",
                "affected_hand": "left", "time_since_event": 1,
            })
        with pytest.raises((ValueError, TypeError)):
            patient_controller.register_patient({
                "name": "X", "age": -1, "sex": "F",
                "affected_hand": "left", "time_since_event": 1,
            })
        with pytest.raises((ValueError, TypeError)):
            patient_controller.register_patient({
                "name": "X", "age": 60, "sex": "F",
                "affected_hand": "both", "time_since_event": 1,
            })
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_02_recording_session_markers():
    td = tempfile.mkdtemp(prefix="bb_demo_")
    try:
        _, patient_controller, recording_controller, session_controller, marker_controller = _make_controllers(td)
        pid = patient_controller.register_patient({
            "name": "Joao Demo", "age": 58, "sex": "M",
            "affected_hand": "left", "time_since_event": 12,
        })
        rid = recording_controller.start_recording(StartRecordingRequest(
            patient_id=pid, filename="demo.csv", task_type="jogo",
        ))
        assert rid > 0
        sess = session_controller.start_session(StartSessionRequest(
            patient_id=pid, task_type="jogo", recording_id=rid, started_at_epoch=time.time(),
        ))
        assert sess.game_mode is True
        with pytest.raises(ValueError):
            session_controller.start_session(StartSessionRequest(
                patient_id=pid, task_type="jogo", recording_id=rid, started_at_epoch=time.time(),
            ))
        assert Session(patient_id=pid, task_type="Jogo", recording_id=rid,
                       started_at_epoch=time.time()).game_mode is True
        marker_controller.reset_state()
        r1 = marker_controller.register_marker("T1", "jogo")
        assert r1.accepted and r1.external_signal == "trigger_left"
        r2 = marker_controller.register_marker("T2", "Jogo")
        assert r2.accepted and r2.external_signal == "trigger_right"
        with pytest.raises(ValueError):
            marker_controller.register_marker("TX", "jogo")
        state = marker_controller.get_state()
        assert (state.t1_count, state.t2_count) == (1, 1)
        marker_controller.start_baseline(2)
        blocked = marker_controller.register_marker("T1", "jogo")
        assert blocked.accepted is False and blocked.reason == "baseline_active"
        marker_controller.tick_baseline()
        marker_controller.tick_baseline()
        ok_again = marker_controller.register_marker("T1", "jogo")
        assert ok_again.accepted is True
        ended = session_controller.end_session()
        assert ended is not None
        recording_controller.stop_recording(rid, 10)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_03_eeg_window_quality_preprocess():
    window = _synthetic_window()
    assert window.shape == (250, 16)
    validator = EEGWindowQualityValidator()
    assert validator.validate(window).accepted is True
    assert validator.validate(np.zeros((250, 16))).accepted is False
    assert validator.validate(window * 1000).accepted is False
    clean = preprocess_window(window)
    assert clean.shape == (250, 16) and np.all(np.isfinite(clean))
    with pytest.raises(ValueError):
        preprocess_window(np.zeros((100, 16)))
    with pytest.raises(ValueError):
        preprocess_window(np.full((250, 16), np.nan))


def test_04_game_window_and_authorization():
    coord = GameInferenceCoordinator(window_size=250, channels=16, window_duration_ms=2000)
    coord.start_window(started_at_ms=1000.0, task_hand="left")
    result = None
    for i in range(250):
        result = coord.add_sample([float(i % 7)] * 16, now_ms=1000.0 + i * 8)
    assert result.status == GameInferenceCoordinator.STATUS_READY
    assert len(result.window) == 250
    assert coord.claim_prediction(result.generation) is True
    assert coord.claim_prediction(result.generation) is False
    assert coord.allows_movement(result.generation, "left", 0, 0.9, now_ms=1500.0) is True
    assert coord.allows_movement(result.generation, "right", 0, 0.9, now_ms=1500.0) is False
    assert coord.allows_movement(result.generation, "left", 1, 0.9, now_ms=1500.0) is False
    assert UnityCommandMapper.from_prediction(0).direction == "esquerda"
    assert UnityCommandMapper.from_prediction(1).direction == "direita"
    with pytest.raises(ValueError):
        UnityCommandMapper.from_prediction(7)
    coord2 = GameInferenceCoordinator(window_size=250, channels=16,
                                      window_duration_ms=2000, collection_margin_ms=500)
    coord2.start_window(started_at_ms=0.0, task_hand="right")
    expired = coord2.add_sample([1.0] * 16, now_ms=99999.0)
    assert expired.status == GameInferenceCoordinator.STATUS_EXPIRED


def test_05_unity_session_flow_real_socket():
    _cleanup_unity()
    comm = UnityCommunicator()
    assert comm.start_server()
    time.sleep(0.25)
    try:
        vr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vr.settimeout(5.0)
        vr.connect(("127.0.0.1", 12345))
        vr.settimeout(1.0)
        time.sleep(0.2)
        comm.set_pending_session(PatientData(nome="Demo VR", nivel=5, lado="Esquerdo"), TaskType.TREINO)
        vr.sendall(b"B;HEADER;1;hand;E\n")
        time.sleep(0.6)
        try:
            data = vr.recv(8192).decode("utf-8", errors="ignore")
        except socket.timeout:
            data = ""
        assert "Demo VR" in data and "Treino" in data
        assert comm.session.phase == SessionPhase.SETUP
        assert comm.send_trigger() is False
        vr.sendall(b"Confirm\n")
        time.sleep(0.3)
        assert comm.session.phase == SessionPhase.READY
        assert comm.send_trigger() is True
        assert comm.session.phase == SessionPhase.ACTIVE
        flowers = []
        comm.on_flower_action = flowers.append
        vr.sendall(b"LEFT_FLOWER\nRIGHT_FLOWER\n")
        time.sleep(0.5)
        assert [a.value for a in flowers] == ["LEFT_FLOWER", "RIGHT_FLOWER"]
        assert comm.end_task("ok") is True
        assert comm.end_session("Fim!") is True
        vr.sendall(b"confirm_end\n")
        time.sleep(0.3)
        assert comm.session.phase == SessionPhase.IDLE
        vr.close()
    finally:
        comm.stop_server()
        _cleanup_unity()


def test_06_esp32_flow():
    gateway = FakeESP32Gateway()
    controller = ESP32Controller.from_gateway(gateway)
    assert controller.connect() is True
    assert controller.send_direction("esquerda") is True
    assert controller.send_direction("direita") is True
    with pytest.raises(ValueError):
        controller.send_direction("cima")
    with pytest.raises(ValueError):
        controller.send_direction("   ")
    controller.disconnect()
    assert controller.send_direction("esquerda") is False


def test_07_inference_use_cases():
    gateway = FakeInferenceGateway(forced_index=1)
    result = RunInferenceUseCase(gateway).execute(_synthetic_window().tolist())
    assert result.predicted_index == 1 and 0.0 <= result.confidence <= 1.0
    result.validate()
    with pytest.raises(ValueError):
        RunInferenceUseCase(gateway).execute([])
    meta = ModelMetadata(path="/m/paciente.keras", name="paciente",
                         backend="tensorflow", input_shape=(None, 250, 16),
                         expected_time_steps=250, expected_channels=16,
                         modified_at_epoch=time.time())
    meta.validate()


def test_08_accuracy_with_real_vr_messages():
    assert AccuracyPresenter.parse_message("CORRECT").is_correct is True
    assert AccuracyPresenter.parse_message("WRONG").is_correct is False
    assert AccuracyPresenter.parse_message("LEFT_FLOWER") is None
    assert AccuracyPresenter.parse_message("RED_FLOWER,TRIGGER_ACTION_LEFT").is_correct is True
    assert AccuracyPresenter.parse_message("lixo") is None
    trials = [AccuracyPresenter.parse_message(m) for m in ("CORRECT", "WRONG", "CORRECT")]
    view = AccuracyPresenter.present([t for t in trials if t is not None])
    assert view.correct_count == 2 and view.total_count == 3
    assert "66.7%" in view.summary_text


def test_09_full_demo_two_rounds():
    td = tempfile.mkdtemp(prefix="bb_demo_")
    try:
        telemetry = PipelineTelemetry()
        _, patient_controller, recording_controller, session_controller, marker_controller = _make_controllers(td)
        pid = patient_controller.register_patient({
            "name": "Demo Ministra", "age": 60, "sex": "F",
            "affected_hand": "left", "time_since_event": 8,
        })
        telemetry.record("PATIENT_CREATED", patient_id=pid)
        rid = recording_controller.start_recording(StartRecordingRequest(
            patient_id=pid, filename="demo_ministra.csv", task_type="jogo"))
        sess = session_controller.start_session(StartSessionRequest(
            patient_id=pid, task_type="jogo", recording_id=rid, started_at_epoch=time.time()))
        assert sess.game_mode is True
        telemetry.record("SESSION_STARTED", recording_id=rid)
        unity = UnityController.from_gateway(FakeUnityGateway())
        esp32 = ESP32Controller.from_gateway(FakeESP32Gateway())
        assert unity.set_pending_session("Demo Ministra", 5, "Esquerdo", "jogo") is None
        assert unity.send_trigger() is True
        telemetry.record("VR_TRIGGERED")
        quality = EEGWindowQualityValidator()
        trials = []
        for rnd, (cue, hand, idx, vr_answer) in enumerate([
            ("T1", "left", 0, "CORRECT"),
            ("T2", "right", 1, "WRONG"),
        ]):
            reg = marker_controller.register_marker(cue, "jogo")
            assert reg.accepted and reg.external_signal in ("trigger_left", "trigger_right")
            unity.send_action(reg.external_signal)
            telemetry.record("TASK_SENT", marker=cue, round=rnd)
            coord = GameInferenceCoordinator(window_size=250, channels=16, window_duration_ms=2000)
            coord.start_window(task_hand=hand)
            ready = None
            window_data = _synthetic_window(seed=rnd)
            for i, sample in enumerate(window_data):
                ready = coord.add_sample(sample, now_ms=1000.0 + i * 8)
            assert ready.status == GameInferenceCoordinator.STATUS_READY
            assert quality.validate(ready.window).accepted is True
            clean = preprocess_window(np.asarray(ready.window))
            fake_model = FakeInferenceGateway(forced_index=idx)
            prediction = RunInferenceUseCase(fake_model).execute(clean.tolist())
            assert coord.claim_prediction(ready.generation) is True
            runtime_action = UnityCommandMapper.from_prediction(prediction.predicted_index)
            assert coord.allows_movement(ready.generation, hand, prediction.predicted_index,
                                         prediction.confidence, now_ms=1500.0) is True
            assert unity.send_action(runtime_action.direction) is True
            assert esp32.send_direction(runtime_action.direction) is True
            telemetry.record("PREDICTION_DONE", predicted_index=idx, round=rnd)
            trial = AccuracyPresenter.parse_message(vr_answer)
            assert trial is not None
            trials.append(trial)
            telemetry.record("UNITY_RESPONSE", message=vr_answer, round=rnd)
        view = AccuracyPresenter.present(trials)
        assert view.total_count == 2 and view.correct_count == 1
        assert "50.0%" in view.summary_text
        assert unity.end_task() is True
        assert unity.end_session("Parabens! Sessao finalizada com sucesso!") is True
        assert session_controller.end_session() is not None
        recording_controller.stop_recording(rid, 60)
        telemetry.record("SESSION_ENDED")
        names = [e.name for e in telemetry.latest_events(50)]
        for expected in ("PATIENT_CREATED", "SESSION_STARTED", "VR_TRIGGERED",
                         "TASK_SENT", "PREDICTION_DONE", "UNITY_RESPONSE", "SESSION_ENDED"):
            assert expected in names
        recs = recording_controller.list_patient_recordings(pid)
        assert any(r.filename == "demo_ministra.csv" for r in recs)
    finally:
        shutil.rmtree(td, ignore_errors=True)
