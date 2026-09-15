"""
Marker and baseline-related use cases.
"""

from dataclasses import replace
from typing import Dict

from brainbridge_v2.application.ports.marker_state_store import MarkerStateStore
from brainbridge_v2.domain.entities.marker_state import MarkerState


class GetMarkerStateUseCase:
    def __init__(self, store: MarkerStateStore):
        self._store = store

    def execute(self) -> MarkerState:
        return self._store.get()


class ResetMarkerStateUseCase:
    def __init__(self, store: MarkerStateStore):
        self._store = store

    def execute(self) -> MarkerState:
        state = MarkerState()
        state.validate()
        return self._store.save(state)


class RegisterMarkerUseCase:
    def __init__(self, store: MarkerStateStore):
        self._store = store

    def execute(self, marker_type: str, task_type: str) -> Dict[str, object]:
        state = self._store.get()
        if state.baseline_active:
            return {
                "accepted": False,
                "reason": "baseline_active",
                "state": state,
                "external_signal": None,
                "esp32_direction": None,
            }

        normalized_marker = marker_type.strip().upper()
        normalized_task = task_type.strip().lower()
        if normalized_marker not in {"T1", "T2"}:
            raise ValueError("Marcador invalido.")

        if normalized_marker == "T1":
            new_state = replace(state, t1_count=state.t1_count + 1)
            external_signal = "trigger_left" if normalized_task in {"teste", "treino", "jogo"} else None
        else:
            new_state = replace(state, t2_count=state.t2_count + 1)
            external_signal = "trigger_right" if normalized_task in {"teste", "treino", "jogo"} else None

        new_state.validate()
        saved = self._store.save(new_state)
        return {
            "accepted": True,
            "reason": None,
            "state": saved,
            "external_signal": external_signal,
            "esp32_direction": None,  # Cues never authorize orthosis movement.
            "marker_type": normalized_marker,
        }


class StartBaselineUseCase:
    def __init__(self, store: MarkerStateStore):
        self._store = store

    def execute(self, duration_seconds: int = 300) -> MarkerState:
        if duration_seconds <= 0:
            raise ValueError("Duracao de baseline deve ser positiva.")
        state = self._store.get()
        new_state = replace(state, baseline_remaining_seconds=duration_seconds)
        new_state.validate()
        return self._store.save(new_state)


class TickBaselineUseCase:
    def __init__(self, store: MarkerStateStore):
        self._store = store

    def execute(self) -> Dict[str, object]:
        state = self._store.get()
        if state.baseline_remaining_seconds <= 0:
            return {
                "state": state,
                "finished": True,
            }

        new_remaining = max(0, state.baseline_remaining_seconds - 1)
        new_state = replace(state, baseline_remaining_seconds=new_remaining)
        new_state.validate()
        saved = self._store.save(new_state)
        return {
            "state": saved,
            "finished": saved.baseline_remaining_seconds == 0,
        }
