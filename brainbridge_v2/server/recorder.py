"""Gravacao CSV OpenBCI no servidor (paridade com o logger do desktop).

Um RecorderManager global liga/desliga a gravacao por recording_id:
- assina o EEGHub via callback sincrono (thread de ingestao);
- escreve via OpenBCICSVLogger (16ch, 125 Hz, Annotations T1/T2/T0);
- marcadores externos (POST /api/markers) entram como pendentes.
"""

import threading
from typing import Dict, Optional


class RecorderManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: Dict[int, dict] = {}

    def start(self, recording_id: int, patient_id: str, task: str,
              patient_name: str = "Unknown", base_path: Optional[str] = None) -> str:
        from brainbridge_v2.infrastructure.acquisition.data_logger import OpenBCICSVLogger
        from brainbridge_v2.infrastructure.config.settings import RECORDINGS_DIR
        from brainbridge_v2.server.eeg_hub import get_hub

        with self._lock:
            if recording_id in self._active:
                raise ValueError("Gravacao ja ativa para este recording_id.")
            logger = OpenBCICSVLogger(
                patient_id=str(patient_id), task=str(task),
                patient_name=patient_name,
                base_path=base_path or str(RECORDINGS_DIR))
            pending = {"marker": None}

            def _on_sample(message) -> None:
                data = (message or {}).get("data")
                if not data:
                    return
                marker = pending["marker"]
                pending["marker"] = None
                try:
                    logger.log_sample([float(v) for v in data[:16]], marker)
                except Exception:
                    pass

            off = get_hub().subscribe_sync(_on_sample)
            self._active[recording_id] = {
                "logger": logger, "off": off, "pending": pending,
            }
            return str(logger.get_full_path())

    def add_marker(self, recording_id: int, marker: Optional[str]) -> bool:
        with self._lock:
            entry = self._active.get(recording_id)
            if entry is None or not marker:
                return False
            entry["pending"]["marker"] = str(marker)
            return True

    def broadcast_marker(self, marker: Optional[str]) -> int:
        """Marca pendente em todas as gravacoes ativas (uso via /api/markers)."""
        with self._lock:
            ids = list(self._active)
        count = 0
        for recording_id in ids:
            if self.add_marker(recording_id, marker):
                count += 1
        return count

    def stop(self, recording_id: int) -> Optional[str]:
        with self._lock:
            entry = self._active.pop(recording_id, None)
        if entry is None:
            return None
        try:
            entry["off"]()
        except Exception:
            pass
        logger = entry["logger"]
        try:
            path = str(logger.get_full_path())
        except Exception:
            path = None
        try:
            logger.stop_logging()
        except Exception:
            pass
        return path

    def active_ids(self):
        with self._lock:
            return list(self._active)


_manager: Optional[RecorderManager] = None
_manager_lock = threading.Lock()


def get_recorder() -> RecorderManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = RecorderManager()
        return _manager
