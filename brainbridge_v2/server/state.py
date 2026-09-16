"""Estado compartilhado do servidor: container + jobs de treino."""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingJob:
    job_id: str
    csv_file_path: str
    patient_id: int
    auto_load: bool = True
    status: str = "queued"  # queued | running | done | error
    messages: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at_epoch: float = field(default_factory=time.time)
    finished_at_epoch: Optional[float] = None
    _queue: Any = field(default_factory=queue.Queue, repr=False)

    def emit(self, message: str) -> None:
        text = str(message)
        self.messages.append(text)
        try:
            self._queue.put_nowait(text)
        except Exception:
            pass


class ServerState:
    """ Guarda container (injeção) + jobs. Instância única via app.state."""

    def __init__(self, container=None):
        self.container = container
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

        # Veredictos do VR (CORRECT/WRONG via TCP da Unity). O comunicador
        # já recebe essas mensagens; aqui elas viram fila consumível pela UI.
        self._vr_verdicts: List[Dict[str, Any]] = []
        self._vr_seq = 0
        self._vr_lock = threading.Lock()
        self._wire_vr_verdicts()

    def _wire_vr_verdicts(self):
        try:
            controller = getattr(self.container, "unity_controller", None)
            if controller is None:
                return
            controller.set_message_callback(self._on_unity_message)
        except Exception:
            pass

    def _on_unity_message(self, message) -> None:
        text_msg = str(message or "")
        upper = text_msg.upper()
        kind = None
        if "CORRECT" in upper:
            kind = "correct"
        elif "WRONG" in upper:
            kind = "wrong"
        if kind is None:
            return
        with self._vr_lock:
            self._vr_seq += 1
            self._vr_verdicts.append({
                "seq": self._vr_seq,
                "verdict": kind,
                "message": text_msg[:200],
                "at_epoch": time.time(),
            })
            del self._vr_verdicts[:-200]

    def drain_vr_verdicts(self, after_seq: int = 0):
        with self._vr_lock:
            pending = [v for v in self._vr_verdicts if v["seq"] > int(after_seq or 0)]
            last = self._vr_seq
        return pending, last

    def new_job(self, csv_file_path: str, patient_id: int, auto_load: bool = True) -> TrainingJob:
        job = TrainingJob(job_id=uuid.uuid4().hex, csv_file_path=csv_file_path,
                          patient_id=patient_id, auto_load=auto_load)
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self.jobs.get(job_id)


def get_state(app) -> ServerState:
    return app.state.brainbridge
