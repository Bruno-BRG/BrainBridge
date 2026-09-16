"""Smoke tests do backend FastAPI (sem hardware, sem TF, DB temporario)."""

import csv
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brainbridge_v2.application.runtime_config import reset_runtime
from brainbridge_v2.bootstrap.container import AppContainer
from brainbridge_v2.infrastructure.acquisition.eeg_stream_gateway_adapter import (
    EEGStreamGatewayAdapter)
from brainbridge_v2.infrastructure.communication.esp32_gateway_adapter import (
    ESP32GatewayAdapter)
from brainbridge_v2.infrastructure.communication.unity_gateway_adapter import (
    UnityGatewayAdapter)
from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.ml.model_catalog_gateway_adapter import (
    FileSystemModelCatalogGatewayAdapter)
from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter)
from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
    ModelTrainingGatewayAdapter)
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import (
    SQLitePatientRepository)
from brainbridge_v2.infrastructure.repositories.sqlite_recording_repository import (
    SQLiteRecordingRepository)
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import (
    InMemoryMarkerStateStore)
from brainbridge_v2.infrastructure.state.in_memory_session_store import (
    InMemorySessionStore)
from brainbridge_v2.interface_adapters.controllers.eeg_stream_controller import (
    EEGStreamController)
from brainbridge_v2.interface_adapters.controllers.esp32_controller import (
    ESP32Controller)
from brainbridge_v2.interface_adapters.controllers.inference_controller import (
    InferenceController)
from brainbridge_v2.interface_adapters.controllers.marker_controller import (
    MarkerController)
from brainbridge_v2.interface_adapters.controllers.patient_controller import (
    PatientController)
from brainbridge_v2.interface_adapters.controllers.recording_controller import (
    RecordingController)
from brainbridge_v2.interface_adapters.controllers.session_controller import (
    SessionController)
from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController)
from brainbridge_v2.interface_adapters.controllers.unity_controller import (
    UnityController)
from brainbridge_v2.server.app import create_app
from brainbridge_v2.server.eeg_hub import EEGHub, extract_eeg_samples


def build_test_container(db_path: Path) -> AppContainer:
    db_manager = DatabaseManager(db_path=db_path)
    return AppContainer(
        db_manager=db_manager,
        eeg_stream_controller=EEGStreamController.from_gateway(EEGStreamGatewayAdapter()),
        inference_controller=InferenceController.from_gateways(
            FileSystemModelCatalogGatewayAdapter(),
            TensorFlowInferenceGatewayAdapter()),
        training_controller=TrainingController.from_gateways(
            ModelTrainingGatewayAdapter(), TensorFlowInferenceGatewayAdapter()),
        patient_controller=PatientController.from_repository(
            SQLitePatientRepository(db_manager)),
        recording_controller=RecordingController.from_repository(
            SQLiteRecordingRepository(db_manager)),
        session_controller=SessionController.from_store(InMemorySessionStore()),
        marker_controller=MarkerController.from_store(InMemoryMarkerStateStore()),
        unity_controller=UnityController.from_gateway(UnityGatewayAdapter()),
        esp32_controller=ESP32Controller.from_gateway(ESP32GatewayAdapter()),
    )


@pytest.fixture
def client():
    reset_runtime()
    with tempfile.TemporaryDirectory() as temp_dir:
        container = build_test_container(Path(temp_dir) / "test.db")
        with TestClient(create_app(container)) as test_client:
            yield test_client
    reset_runtime()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_patients_recordings_sessions(client):
    created = client.post("/api/patients", json={
        "name": "Teste API", "age": 60, "sex": "M",
        "affected_hand": "left", "time_since_event": 6})
    assert created.status_code == 201
    patient_id = created.json()["data"]["id"]

    listed = client.get("/api/patients")
    assert any(p["id"] == patient_id for p in listed.json()["data"])

    patched = client.patch(f"/api/patients/{patient_id}/affected-hand",
                           json={"affected_hand": "right"})
    assert patched.status_code == 200

    rec = client.post("/api/recordings/start", json={
        "patient_id": patient_id, "filename": "api_test.csv", "task_type": "livre"})
    assert rec.status_code == 201
    recording_id = rec.json()["data"]["id"]

    sess = client.post("/api/sessions/start", json={
        "patient_id": patient_id, "task_type": "livre", "recording_id": recording_id})
    assert sess.status_code == 201
    assert client.get("/api/sessions/current").json()["data"]["task_type"] == "livre"

    recs = client.get(f"/api/patients/{patient_id}/recordings")
    assert len(recs.json()["data"]) == 1

    assert client.post(f"/api/recordings/{recording_id}/stop",
                       json={"duration_seconds": 5}).status_code == 200
    assert client.post("/api/sessions/end").status_code == 200
    assert client.get("/api/sessions/current").json()["data"] is None

    cal = client.get(f"/api/patients/{patient_id}/calibration")
    assert cal.json()["data"]["calibrated"] is False


def test_markers_and_baseline(client):
    reg = client.post("/api/markers", json={"marker_type": "T1", "task_type": "treino"})
    assert reg.status_code == 201
    assert reg.json()["data"]["accepted"] is True
    state = client.get("/api/markers/state")
    assert state.json()["data"]["t1_count"] == 1
    assert client.post("/api/markers/reset").status_code == 200
    assert client.get("/api/markers/state").json()["data"]["t1_count"] == 0
    assert client.post("/api/baseline/start", json={"duration_seconds": 5}).status_code == 200
    assert client.post("/api/baseline/tick").status_code == 200


def test_models_and_predict_without_model(client):
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/models/loaded").json()["data"] is None
    assert client.get("/api/inference/ea").status_code == 200
    response = client.post("/api/inference/predict",
                           json={"window": [[0.0] * 16] * 250})
    assert response.status_code == 400
    assert client.post("/api/models/load",
                       json={"path": "/nao/existe.keras"}).status_code in (400, 500)


def test_config_roundtrip(client):
    current = client.get("/api/config")
    assert current.json()["data"]["rl_batch_k"] == 5
    updated = client.put("/api/config", json={"values": {"rl_enabled": True, "rl_batch_k": 3}})
    assert updated.json()["data"]["rl_batch_k"] == 3
    assert client.put("/api/config", json={"values": {"nope": 1}}).status_code == 400
    rl = client.get("/api/rl/status")
    assert rl.json()["data"]["enabled"] is True


def test_devices_status_without_hardware(client):
    status = client.get("/api/devices/status")
    assert status.status_code == 200
    data = status.json()["data"]
    assert data["esp32"]["connected"] is False
    assert data["unity"]["server_active"] is False
    assert client.post("/api/devices/esp32/action",
                       json={"direction": "girafa"}).status_code == 400


def test_training_check_and_missing_csv_job(client):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "calib.csv"
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["%Sample Rate = 125 Hz"])
            writer.writerow(["%Signal Stage = raw"])
            writer.writerow(["Sample Index"] + [f"EXG Channel {i}" for i in range(16)] + ["Annotations"])
            import numpy as np
            rng = np.random.default_rng(0)
            for i in range(600):
                marker = {0: "T1", 300: "T2"}.get(i, "")
                writer.writerow([i] + rng.normal(size=16).tolist() + [marker])
        check = client.post("/api/training/check",
                            json={"csv_file_path": str(csv_path), "required": 2})
        assert check.json()["data"] == {"trials": 1, "required": 2, "ok": False}

    started = client.post("/api/training", json={
        "csv_file_path": "/nao/existe.csv", "patient_id": 1, "auto_load": False})
    assert started.status_code == 202
    job_id = started.json()["data"]["job_id"]
    import time
    deadline = time.time() + 10
    status = "running"
    while status == "running" and time.time() < deadline:
        import time as _t
        _t.sleep(0.2)
        status = client.get(f"/api/training/{job_id}").json()["data"]["status"]
    assert status == "error"
    assert client.get("/api/training/nope").status_code == 404


def test_extract_eeg_samples_formats():
    assert extract_eeg_samples([1.0] * 16) is not None
    assert extract_eeg_samples({"Ch1": 1.0, **{f"Ch{i}": 0.0 for i in range(2, 17)}}) is not None
    assert extract_eeg_samples({"type": "timeSeriesRaw",
                                "data": [[0.0] * 4] * 16}) is not None
    assert extract_eeg_samples({"channels": [2.0] * 16}) is not None
    assert extract_eeg_samples("nao-json") is None
    assert extract_eeg_samples([1.0] * 8) is None


def test_eeg_hub_synth_and_websocket(client):
    hub_status = client.post("/api/devices/eeg/connect", json={"simulate": True})
    assert hub_status.json()["data"]["mode"] == "synth"
    try:
        with client.websocket_connect("/ws/eeg") as websocket:
            start = websocket.receive_json()
            assert start["type"] == "status"
            websocket.send_json({"cmd": "ping"})
            got_eeg = False
            for _ in range(50):
                message = websocket.receive_json()
                if message.get("type") == "eeg" and message.get("count", 0) > 0:
                    assert len(message["batch"][0]) == 16
                    got_eeg = True
                    break
            assert got_eeg
            websocket.send_json({"cmd": "disconnect"})
    finally:
        client.post("/api/devices/eeg/disconnect")
    assert EEGHub().status()["mode"] == "idle"


def test_recording_writes_csv_with_markers(client):
    import time as _time

    created = client.post("/api/patients", json={"name": "Gravacao API"})
    patient_id = created.json()["data"]["id"]
    client.post("/api/devices/eeg/connect", json={"simulate": True})
    try:
        started = client.post("/api/recordings/start", json={
            "patient_id": patient_id, "filename": "web_test.csv", "task_type": "treino"})
        assert started.status_code == 201
        recording_id = started.json()["data"]["id"]
        csv_path = started.json()["data"]["csv_path"]
        assert csv_path and csv_path.endswith(".csv")
        _time.sleep(0.6)
        marker = client.post("/api/markers", json={"marker_type": "T1", "task_type": "treino"})
        assert marker.status_code == 201
        _time.sleep(0.6)
        stopped = client.post(f"/api/recordings/{recording_id}/stop",
                              json={"duration_seconds": 1})
        assert stopped.json()["data"]["csv_path"] == csv_path
    finally:
        client.post("/api/devices/eeg/disconnect")
    with open(csv_path) as handle:
        content = handle.read()
    assert "%Sample Rate = 125 Hz" in content
    assert ",T1" in content
    data_rows = [line for line in content.splitlines()
                 if line and not line.startswith("%") and not line.startswith("Sample Index")]
    assert len(data_rows) > 50


def test_unity_verdicts_roundtrip(client):
    empty = client.get("/api/devices/unity/verdicts")
    assert empty.status_code == 200
    assert empty.json()["data"]["verdicts"] == []
    assert empty.json()["data"]["last_seq"] == 0

    state = client.app.state.brainbridge
    state._on_unity_message("TRIAL 3 CORRECT")
    state._on_unity_message("algo sem veredicto")
    state._on_unity_message("WRONG")

    first = client.get("/api/devices/unity/verdicts", params={"after": 0})
    verdicts = first.json()["data"]["verdicts"]
    last_seq = first.json()["data"]["last_seq"]
    assert [v["verdict"] for v in verdicts] == ["correct", "wrong"]
    assert last_seq == 2

    second = client.get("/api/devices/unity/verdicts", params={"after": last_seq})
    assert second.json()["data"]["verdicts"] == []


def test_eeg_stream_filter_raw_vs_filtered():
    from brainbridge_v2.server.eeg_hub import extract_eeg_samples
    raw = {"type": "timeSeriesRaw", "data": [[float(c)] * 4 for c in range(16)]}
    filt = {"type": "timeSeries", "data": [[float(c)] * 4 for c in range(16)]}
    assert extract_eeg_samples(raw, "raw") is not None
    assert extract_eeg_samples(filt, "raw") is None
    assert extract_eeg_samples(filt, "filtered") is not None
    assert extract_eeg_samples(raw, "filtered") is None
    assert extract_eeg_samples({"Ch1": 1.0, **{f"Ch{c}": 0.0 for c in range(2, 17)}}, "raw") is not None


def test_eeg_connect_loud_failure_on_busy_port():
    import socket
    from brainbridge_v2.server.eeg_hub import EEGHub
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("127.0.0.1", 0))
    busy_port = blocker.getsockname()[1]
    hub = EEGHub()
    try:
        with pytest.raises(RuntimeError):
            hub.connect("127.0.0.1", busy_port, simulate=False)
        assert hub.status()["running"] is False
    finally:
        hub.disconnect()
        blocker.close()


def test_unity_publish_session(client):
    created = client.post("/api/patients", json={
        "name": "Jogador VR", "age": 40, "sex": "M",
        "affected_hand": "right", "time_since_event": 3})
    patient_id = created.json()["data"]["id"]
    rec = client.post("/api/recordings/start", json={
        "patient_id": patient_id, "filename": "vr_test.csv", "task_type": "jogo"})
    assert rec.status_code == 201
    pub = client.post("/api/devices/unity/publish-session", json={
        "patient_id": patient_id, "task_type": "jogo"})
    assert pub.status_code == 200
    data = pub.json()["data"]
    assert data["published"] is True
    assert data["nome"] == "Jogador VR"
    assert data["lado"] == "Direito"
    assert data["nivel"] == 0
    missing = client.post("/api/devices/unity/publish-session", json={
        "patient_id": 999999, "task_type": "jogo"})
    assert missing.status_code == 404


def test_esp32_ports_and_connect_report(client):
    ports = client.get("/api/devices/esp32/ports")
    assert ports.status_code == 200
    assert isinstance(ports.json()["data"]["ports"], list)
    rep = client.post("/api/devices/esp32/connect", json={"port": "/dev/ttyINEXISTENTE"})
    assert rep.status_code == 200
    data = rep.json()["data"]
    assert data["connected"] is False
    assert "reason" in data and data["reason"]
