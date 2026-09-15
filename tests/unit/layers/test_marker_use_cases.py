import pytest

from brainbridge_v2.application.use_cases.marker_use_cases import (
    RegisterMarkerUseCase,
    ResetMarkerStateUseCase,
    StartBaselineUseCase,
    TickBaselineUseCase,
)
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import (
    InMemoryMarkerStateStore,
)


def test_register_marker_updates_t1_and_returns_external_signal():
    store = InMemoryMarkerStateStore()
    result = RegisterMarkerUseCase(store).execute("T1", "jogo")

    assert result["accepted"] is True
    assert result["state"].t1_count == 1
    assert result["external_signal"] == "trigger_left"
    assert result["esp32_direction"] is None


@pytest.mark.parametrize("task", ["jogo", "treino", "teste"])
@pytest.mark.parametrize("marker,cue", [("T1", "trigger_left"), ("T2", "trigger_right")])
def test_cues_preserve_unity_payload_but_never_authorize_serial(task, marker, cue):
    result = RegisterMarkerUseCase(InMemoryMarkerStateStore()).execute(marker, task)
    assert result["external_signal"] == cue
    assert result["esp32_direction"] is None


def test_register_marker_rejects_marker_during_baseline():
    store = InMemoryMarkerStateStore()
    StartBaselineUseCase(store).execute(10)

    result = RegisterMarkerUseCase(store).execute("T2", "jogo")

    assert result["accepted"] is False
    assert result["reason"] == "baseline_active"
    assert result["state"].t2_count == 0


def test_tick_baseline_finishes_countdown():
    store = InMemoryMarkerStateStore()
    StartBaselineUseCase(store).execute(2)
    tick = TickBaselineUseCase(store)

    first = tick.execute()
    second = tick.execute()

    assert first["finished"] is False
    assert first["state"].baseline_remaining_seconds == 1
    assert second["finished"] is True
    assert second["state"].baseline_remaining_seconds == 0


def test_reset_marker_state_clears_counters_and_baseline():
    store = InMemoryMarkerStateStore()
    register = RegisterMarkerUseCase(store)
    register.execute("T1", "treino")
    StartBaselineUseCase(store).execute(8)

    reset = ResetMarkerStateUseCase(store).execute()

    assert reset.t1_count == 0
    assert reset.t2_count == 0
    assert reset.baseline_remaining_seconds == 0


def test_start_baseline_requires_positive_duration():
    store = InMemoryMarkerStateStore()

    with pytest.raises(ValueError):
        StartBaselineUseCase(store).execute(0)
