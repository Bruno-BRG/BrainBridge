"""Rotas REST espelhando os controllers (sem Qt)."""

import asyncio
import dataclasses
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from brainbridge_v2.application.runtime_config import (
    _CALIBRATION_DEFAULTS,
    get_runtime,
    set_runtime,
)
from brainbridge_v2.server import schemas
from brainbridge_v2.server._helpers import asdict, ok, run_controller
from brainbridge_v2.server.eeg_hub import get_hub
from brainbridge_v2.server.state import get_state

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"ok": True, "service": "brainbridge", "time": time.time()}


# -- pacientes --------------------------------------------------------------
@router.get("/patients")
def list_patients(request: Request):
    container = get_state(request.app).container
    return ok(run_controller(container.patient_controller.list_patients))


@router.post("/patients", status_code=201)
def create_patient(body: schemas.PatientCreate, request: Request):
    container = get_state(request.app).container
    patient_id = run_controller(
        lambda: container.patient_controller.register_patient(body.model_dump()))
    return JSONResponse({"ok": True, "data": {"id": patient_id}}, status_code=201)


@router.patch("/patients/{patient_id}/affected-hand")
def update_affected_hand(patient_id: int, body: schemas.AffectedHandUpdate, request: Request):
    container = get_state(request.app).container
    run_controller(lambda: container.patient_controller.update_patient_affected_hand(
        patient_id, body.affected_hand))
    return ok({"id": patient_id, "affected_hand": body.affected_hand})


@router.get("/patients/{patient_id}/recordings")
def list_recordings(patient_id: int, request: Request):
    container = get_state(request.app).container
    return ok(run_controller(
        lambda: container.recording_controller.list_patient_recordings(patient_id)))


@router.get("/patients/{patient_id}/calibration")
def calibration_status(patient_id: int, request: Request):
    container = get_state(request.app).container
    gateway = getattr(container.training_controller, "_training_gateway", None)
    available = False
    if gateway is not None and hasattr(gateway, "patient_model_available"):
        try:
            available = bool(gateway.patient_model_available(patient_id))
        except Exception:
            available = False
    return ok({"patient_id": patient_id, "calibrated": available,
               "trials_required": get_runtime("calib_trials_required", 10)})


# -- gravacoes ---------------------------------------------------------------
@router.post("/recordings/start", status_code=201)
def start_recording(body: schemas.RecordingStart, request: Request):
    from brainbridge_v2.server.recorder import get_recorder

    container = get_state(request.app).container
    recording_id = run_controller(
        lambda: container.recording_controller.start_recording(body.model_dump()))
    csv_path = None
    try:
        patients = container.patient_controller.list_patients()
        match = next((p for p in patients if p["id"] == int(body.patient_id)), {})
        name = (match.get("name") if isinstance(match, dict) else None) or "Unknown"
        csv_path = get_recorder().start(
            recording_id, f"P{int(body.patient_id):03d}", body.task_type,
            patient_name=str(name))
    except Exception as exc:
        csv_path = None
        print(f"[RECORDER] Falha ao iniciar CSV: {exc}")
    return JSONResponse({"ok": True, "data": {"id": recording_id, "csv_path": csv_path}},
                        status_code=201)


@router.post("/recordings/{recording_id}/stop")
def stop_recording(recording_id: int, body: schemas.RecordingStop, request: Request):
    from brainbridge_v2.server.recorder import get_recorder

    container = get_state(request.app).container
    run_controller(lambda: container.recording_controller.stop_recording(
        recording_id, body.duration_seconds))
    return ok({"id": recording_id, "csv_path": get_recorder().stop(recording_id)})


# -- sessoes ------------------------------------------------------------------
@router.post("/sessions/start", status_code=201)
def start_session(body: schemas.SessionStart, request: Request):
    import time as _time

    container = get_state(request.app).container
    payload = body.model_dump()
    if payload.get("started_at_epoch") is None:
        payload["started_at_epoch"] = _time.time()
    session = run_controller(
        lambda: container.session_controller.start_session(payload))
    return JSONResponse({"ok": True, "data": asdict(session)}, status_code=201)


@router.get("/sessions/current")
def current_session(request: Request):
    container = get_state(request.app).container
    session = container.session_controller.get_current_session()
    return ok(asdict(session) if session is not None else None)


@router.post("/sessions/end")
def end_session(request: Request):
    container = get_state(request.app).container
    session = container.session_controller.end_session()
    return ok(asdict(session) if session is not None else None)


# -- marcadores ---------------------------------------------------------------
@router.get("/markers/state")
def marker_state(request: Request):
    container = get_state(request.app).container
    return ok(container.marker_controller.get_state())


@router.post("/markers", status_code=201)
def register_marker(body: schemas.MarkerRegister, request: Request):
    from brainbridge_v2.server.recorder import get_recorder

    container = get_state(request.app).container
    registration = run_controller(lambda: container.marker_controller.register_marker(
        body.marker_type, body.task_type))
    try:
        data = asdict(registration)
        if data.get("accepted") and str(data.get("marker_type") or "").upper() in (
                "T1", "T2", "BASELINE"):
            get_recorder().broadcast_marker(str(data["marker_type"]).upper())
    except Exception:
        pass
    return ok(registration, status_code=201)


@router.post("/markers/reset")
def reset_markers(request: Request):
    container = get_state(request.app).container
    return ok(container.marker_controller.reset_state())


@router.post("/baseline/start")
def baseline_start(body: schemas.BaselineStart, request: Request):
    from brainbridge_v2.server.recorder import get_recorder

    container = get_state(request.app).container
    state = run_controller(
        lambda: container.marker_controller.start_baseline(body.duration_seconds))
    try:
        get_recorder().broadcast_marker("BASELINE")
    except Exception:
        pass
    return ok(state)


@router.post("/baseline/tick")
def baseline_tick(request: Request):
    container = get_state(request.app).container
    return ok(container.marker_controller.tick_baseline())


# -- modelos / inferencia ------------------------------------------------------
@router.get("/models")
def list_models(request: Request):
    container = get_state(request.app).container
    return ok(run_controller(container.inference_controller.list_models))


@router.get("/models/loaded")
def loaded_model(request: Request):
    container = get_state(request.app).container
    model = container.inference_controller.get_loaded_model()
    return ok(asdict(model) if model is not None else None)


@router.post("/models/load")
def load_model(body: schemas.ModelLoad, request: Request):
    container = get_state(request.app).container
    return ok(run_controller(
        lambda: container.inference_controller.load_model(body.path)))


@router.post("/models/load-latest")
def load_latest_model(request: Request):
    container = get_state(request.app).container
    return ok(run_controller(container.inference_controller.load_latest_model))


@router.post("/inference/predict")
async def predict(body: schemas.InferencePredict, request: Request):
    container = get_state(request.app).container

    def _run():
        return container.inference_controller.predict(body.window, input_fs=body.input_fs)

    try:
        result = await asyncio.to_thread(_run)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ok(result)


@router.get("/inference/ea")
def ea_status(request: Request):
    container = get_state(request.app).container
    ctrl = container.inference_controller
    try:
        required = bool(ctrl.ea_required())
    except Exception:
        required = False
    try:
        calibrated = bool(ctrl.ea_calibrated())
    except Exception:
        calibrated = True
    try:
        done, total = ctrl.ea_progress()
    except Exception:
        done, total = 1, 1
    return ok({"required": required, "calibrated": calibrated,
               "done": done, "total": total})


# -- RL online ------------------------------------------------------------------
@router.post("/rl/update")
async def rl_update(body: schemas.RLUpdate, request: Request):
    import asyncio as _asyncio

    container = get_state(request.app).container
    epochs = body.epochs if body.epochs is not None else int(get_runtime("rl_epochs", 3))
    lr = body.lr if body.lr is not None else float(get_runtime("rl_lr", 5e-5))

    def _run():
        return container.inference_controller.rl_online_update(
            body.windows, body.labels, sample_weights=body.weights,
            epochs=epochs, lr=lr, freeze_backbone=False)

    try:
        result = await _asyncio.to_thread(_run)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ok(result)


@router.post("/rl/snapshot")
def rl_snapshot(request: Request):
    container = get_state(request.app).container
    return ok({"snapshot": bool(container.inference_controller.rl_snapshot())})


@router.post("/rl/restore")
def rl_restore(request: Request):
    container = get_state(request.app).container
    restored = bool(container.inference_controller.rl_restore())
    if not restored:
        raise HTTPException(status_code=404, detail="Sem checkpoint pre-RL.")
    return ok({"restored": True})


@router.get("/rl/status")
def rl_status(request: Request):
    container = get_state(request.app).container
    ctrl = container.inference_controller
    try:
        updates = int(ctrl.rl_updates_count())
    except Exception:
        updates = 0
    try:
        has_snapshot = bool(ctrl.rl_has_snapshot())
    except Exception:
        has_snapshot = False
    return ok({"enabled": bool(get_runtime("rl_enabled", False)),
               "updates_applied": updates, "has_snapshot": has_snapshot,
               "max_updates": int(get_runtime("rl_max_updates", 20)),
               "batch_k": int(get_runtime("rl_batch_k", 5))})


# -- dispositivos ---------------------------------------------------------------
@router.get("/devices/status")
def devices_status(request: Request):
    container = get_state(request.app).container
    hub = get_hub()
    return ok({
        "eeg": hub.status(),
        "unity": {
            "server_active": bool(container.unity_controller.is_server_active()),
            "client_connected": bool(container.unity_controller.is_client_connected()),
        },
        "esp32": {"connected": bool(container.esp32_controller.is_connected())},
        "esp32": {
            "connected": bool(container.esp32_controller.is_connected()),
            "port": container.esp32_controller.get_port(),
        },
    })


@router.post("/devices/eeg/connect")
def eeg_connect(body: schemas.EEGConnect, request: Request):
    hub = get_hub()
    return ok(hub.connect(body.host, body.port, simulate=body.simulate,
                          stream=body.stream))


@router.post("/devices/eeg/disconnect")
def eeg_disconnect():
    return ok(get_hub().disconnect())


@router.post("/devices/unity/start")
def unity_start(request: Request):
    container = get_state(request.app).container
    return ok({"server_active": bool(run_controller(container.unity_controller.start_server))})


@router.post("/devices/unity/stop")
def unity_stop(request: Request):
    container = get_state(request.app).container
    container.unity_controller.stop_server()
    return ok({"server_active": False})


@router.post("/devices/unity/action")
def unity_action(body: schemas.UnityAction, request: Request):
    container = get_state(request.app).container
    return ok({"sent": bool(run_controller(
        lambda: container.unity_controller.send_action(body.direction)))})


@router.post("/devices/unity/session")
def unity_session(body: schemas.UnitySession, request: Request):
    container = get_state(request.app).container
    container.unity_controller.set_pending_session(
        body.nome, body.nivel, body.lado, body.tarefa, body.sessoes)
    return ok({"published": True})


@router.post("/devices/unity/trigger")
def unity_trigger(request: Request):
    container = get_state(request.app).container
    return ok({"sent": bool(container.unity_controller.send_trigger())})


@router.post("/devices/unity/end-task")
def unity_end_task(request: Request):
    container = get_state(request.app).container
    return ok({"sent": bool(container.unity_controller.end_task())})


@router.post("/devices/unity/end-session")
def unity_end_session(body: schemas.UnityEndSession, request: Request):
    container = get_state(request.app).container
    return ok({"sent": bool(container.unity_controller.end_session(body.message))})


@router.post("/devices/unity/publish-session")
def unity_publish_session(body: schemas.UnityPublishSession, request: Request):
    """Publica paciente/nivel/lado/tarefa no VR em uma chamada."""
    from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
        ProgressionPresenter)
    container = get_state(request.app).container
    patients = run_controller(container.patient_controller.list_patients)
    patient = next((p for p in patients if int(p.get("id", -1)) == int(body.patient_id)), None)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Paciente {body.patient_id} nao encontrado.")
    recordings = container.recording_controller.list_patient_recordings(int(body.patient_id))
    previous_count = max(0, len(recordings or []) - 1)
    nivel = ProgressionPresenter.level_for_session_count(previous_count)
    hand = str(patient.get("affected_hand") or "left").lower()
    lado = "Direito" if hand == "right" else "Esquerdo"
    task = str(body.task_type or "jogo")
    container.unity_controller.set_pending_session(
        str(patient.get("name") or "Paciente"), nivel, lado, task, previous_count)
    return ok({"published": True, "nome": patient.get("name"), "nivel": nivel,
               "lado": lado, "tarefa": task, "sessoes": previous_count})


@router.get("/devices/unity/verdicts")
def unity_verdicts(request: Request, after: int = 0):
    """Veredictos CORRECT/WRONG enviados pelo VR desde `after` (seq)."""
    state = get_state(request.app)
    pending, last_seq = state.drain_vr_verdicts(after)
    return ok({"verdicts": pending, "last_seq": last_seq})


@router.post("/devices/esp32/connect")
def esp32_connect(request: Request, body: schemas.ESP32Connect | None = None):
    container = get_state(request.app).container
    port = (body.port or "").strip() if body and body.port else None
    report = run_controller(lambda: container.esp32_controller.connect_report(port))
    return ok(report)


@router.get("/devices/esp32/ports")
def esp32_ports(request: Request):
    container = get_state(request.app).container
    return ok({"ports": container.esp32_controller.list_ports()})


@router.post("/devices/esp32/disconnect")
def esp32_disconnect(request: Request):
    container = get_state(request.app).container
    container.esp32_controller.disconnect()
    return ok({"connected": False})


@router.post("/devices/esp32/action")
def esp32_action(body: schemas.ESP32Action, request: Request):
    container = get_state(request.app).container
    return ok({"sent": bool(run_controller(
        lambda: container.esp32_controller.send_direction(body.direction)))})


# -- treino ----------------------------------------------------------------------
@router.post("/training", status_code=202)
async def start_training(body: schemas.TrainingStart, request: Request):
    import asyncio as _asyncio

    state = get_state(request.app)
    container = state.container
    try:
        patient_id = int(body.patient_id)
        if patient_id <= 0:
            raise ValueError("Paciente invalido para treinamento.")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Paciente invalido para treinamento.")
    job = state.new_job(body.csv_file_path, patient_id, auto_load=body.auto_load)
    job.status = "running"

    def _run():
        try:
            if job.auto_load:
                result = container.training_controller.train_and_load_model(
                    job.csv_file_path, job.patient_id,
                    progress_callback=job.emit)
            else:
                result = container.training_controller.train_model(
                    job.csv_file_path, job.patient_id,
                    progress_callback=job.emit)
            job.result = asdict(result)
            job.status = "done"
            job.emit("Concluído com sucesso.")
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.emit(f"Erro: {job.error}")
        finally:
            import time as _time
            job.finished_at_epoch = _time.time()

    _asyncio.get_running_loop().run_in_executor(None, _run)
    return JSONResponse({"ok": True, "data": {"job_id": job.job_id}}, status_code=202)


@router.get("/training/{job_id}")
def training_status(job_id: str, request: Request):
    job = get_state(request.app).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nao encontrado.")
    return ok({"job_id": job.job_id, "status": job.status,
               "messages": list(job.messages), "result": job.result,
               "error": job.error})


@router.get("/training/{job_id}/events")
async def training_events(job_id: str, request: Request):
    import asyncio as _asyncio

    state = get_state(request.app)
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nao encontrado.")

    async def _stream():
        yield f"data: {json.dumps({'status': job.status})}\n\n"
        while True:
            try:
                message = await _asyncio.to_thread(job._queue.get, True, 5.0)
                yield f"data: {json.dumps({'message': message})}\n\n"
            except Exception:
                if job.status in ("done", "error"):
                    break
                yield ": ping\n\n"
            if job.status in ("done", "error") and job._queue.empty():
                break
        yield f"data: {json.dumps({'status': job.status, 'error': job.error})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/training/check")
def training_check(body: schemas.TrialsCheck, request: Request):
    from brainbridge_v2.infrastructure.ml.trainer import count_labeled_trials_total

    required = body.required
    if required is None:
        required = int(get_runtime("calib_trials_required", 10))
    try:
        trials = int(count_labeled_trials_total(body.csv_file_path))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ok({"trials": trials, "required": int(required), "ok": trials >= int(required)})


# -- config -----------------------------------------------------------------------
@router.get("/config")
def get_config():
    return ok({key: get_runtime(key) for key in sorted(_CALIBRATION_DEFAULTS)})


@router.put("/config")
def update_config(body: schemas.ConfigUpdate):
    updated = {}
    for key, value in body.values.items():
        try:
            set_runtime(key, value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        updated[key] = get_runtime(key)
    return ok(updated)
