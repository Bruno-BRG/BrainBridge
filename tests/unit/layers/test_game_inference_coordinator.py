from brainbridge_v2.application.game_inference_coordinator import (
    GameInferenceCoordinator,
)
import pytest


def test_game_inference_waits_for_fresh_window_after_task_start():
    coordinator = GameInferenceCoordinator(
        window_size=3,
        channels=2,
        window_duration_ms=2000,
    )

    coordinator.eeg_buffer.extend([[99.0, 99.0], [98.0, 98.0], [97.0, 97.0]])
    coordinator.start_window(started_at_ms=1000.0)

    first = coordinator.add_sample([1.0, 10.0], now_ms=1100.0)
    second = coordinator.add_sample([2.0, 20.0], now_ms=1200.0)
    third = coordinator.add_sample([3.0, 30.0], now_ms=1300.0)

    assert first.status == GameInferenceCoordinator.STATUS_COLLECTING
    assert second.status == GameInferenceCoordinator.STATUS_COLLECTING
    assert third.status == GameInferenceCoordinator.STATUS_READY
    assert third.window == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_game_inference_locks_after_prediction_is_used():
    coordinator = GameInferenceCoordinator(window_size=1, channels=2)
    coordinator.start_window(started_at_ms=1000.0)

    ready = coordinator.add_sample([1.0, 2.0], now_ms=1001.0)
    coordinator.mark_prediction_used()
    locked = coordinator.add_sample([3.0, 4.0], now_ms=1002.0)

    assert ready.status == GameInferenceCoordinator.STATUS_READY
    assert locked.status == GameInferenceCoordinator.STATUS_LOCKED


def test_game_inference_expires_two_second_window():
    coordinator = GameInferenceCoordinator(
        window_size=3,
        channels=2,
        window_duration_ms=2000,
    )
    coordinator.start_window(started_at_ms=1000.0)

    result = coordinator.add_sample([1.0, 2.0], now_ms=3501.0)

    assert result.status == GameInferenceCoordinator.STATUS_EXPIRED
    assert coordinator.is_window_open is False


def test_game_inference_normalizes_channel_count():
    coordinator = GameInferenceCoordinator(window_size=2, channels=3)
    coordinator.start_window()

    coordinator.add_sample([1.0], now_ms=1.0)
    result = coordinator.add_sample([2.0, 3.0, 4.0, 5.0], now_ms=2.0)

    assert result.status == GameInferenceCoordinator.STATUS_READY
    assert result.window == [[1.0, 0.0, 0.0], [2.0, 3.0, 4.0]]


def test_game_inference_stays_inactive_until_window_starts():
    coordinator = GameInferenceCoordinator(window_size=1, channels=2)

    result = coordinator.add_sample([1.0, 2.0], now_ms=1000.0)

    assert result.status == GameInferenceCoordinator.STATUS_INACTIVE
    assert result.window is None


def test_game_inference_restart_window_discards_partial_samples():
    coordinator = GameInferenceCoordinator(window_size=3, channels=2)
    coordinator.start_window(started_at_ms=1000.0)
    coordinator.add_sample([1.0, 10.0], now_ms=1100.0)
    coordinator.add_sample([2.0, 20.0], now_ms=1200.0)

    coordinator.start_window(started_at_ms=1300.0)
    first = coordinator.add_sample([3.0, 30.0], now_ms=1400.0)
    second = coordinator.add_sample([4.0, 40.0], now_ms=1500.0)
    third = coordinator.add_sample([5.0, 50.0], now_ms=1600.0)

    assert first.status == GameInferenceCoordinator.STATUS_COLLECTING
    assert second.status == GameInferenceCoordinator.STATUS_COLLECTING
    assert third.status == GameInferenceCoordinator.STATUS_READY
    assert third.window == [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]


def test_game_inference_allows_sample_on_exact_expiration_boundary():
    coordinator = GameInferenceCoordinator(
        window_size=1,
        channels=2,
        window_duration_ms=2000,
    )
    coordinator.start_window(started_at_ms=1000.0)

    result = coordinator.add_sample([1.0, 2.0], now_ms=3000.0)

    assert result.status == GameInferenceCoordinator.STATUS_READY
    assert result.window == [[1.0, 2.0]]


def test_game_inference_reset_returns_to_cold_state():
    coordinator = GameInferenceCoordinator(window_size=2, channels=2)
    coordinator.start_window(started_at_ms=1000.0)
    coordinator.add_sample([1.0, 2.0], now_ms=1100.0)
    coordinator.mark_prediction_used()

    coordinator.reset()

    assert coordinator.is_window_open is False
    assert coordinator.prediction_locked is False
    assert coordinator.samples_since_window_start == 0
    assert coordinator.window_started_at_ms is None
    assert list(coordinator.eeg_buffer) == []


def test_game_inference_configure_rebuilds_buffer_and_resets_state():
    coordinator = GameInferenceCoordinator(window_size=3, channels=2)
    coordinator.start_window(started_at_ms=1000.0)
    coordinator.add_sample([1.0, 2.0], now_ms=1100.0)

    coordinator.configure(window_size=2, channels=3, window_duration_ms=1500)

    assert coordinator.window_size == 2
    assert coordinator.channels == 3
    assert coordinator.window_duration_ms == 1500
    assert coordinator.is_window_open is False
    assert coordinator.eeg_buffer.maxlen == 2
    assert list(coordinator.eeg_buffer) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"channels": 0},
        {"window_duration_ms": 0},
    ],
)
def test_game_inference_rejects_invalid_initial_config(kwargs):
    with pytest.raises(ValueError):
        GameInferenceCoordinator(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"channels": 0},
        {"window_duration_ms": 0},
    ],
)
def test_game_inference_rejects_invalid_runtime_config(kwargs):
    coordinator = GameInferenceCoordinator()

    with pytest.raises(ValueError):
        coordinator.configure(**kwargs)


@pytest.mark.parametrize("hand,task,index,allowed", [
    ("left", "left", 0, True), ("right", "right", 1, True),
    (None, "left", 0, False), ("left", "right", 0, False),
    ("right", "left", 1, False), ("left", "left", 1, False),
    ("right", "right", 0, False), ("left", "left", 0.5, False),
])
def test_movement_requires_affected_attempt_and_prediction(hand, task, index, allowed):
    c = GameInferenceCoordinator(window_size=1)
    generation = c.start_window(task_hand=task)
    assert not c.allows_movement(generation, hand, index, 0.9)
    c.add_sample(range(16))
    assert c.claim_prediction(generation)
    assert c.allows_movement(generation, hand, index, 0.9) is allowed
    assert not c.claim_prediction(generation)
    assert not c.allows_movement(generation, hand, index, float("nan"))
    c.reset()
    assert not c.allows_movement(generation, hand, index, 0.9)


def test_full_cap_window_has_explicit_collection_margin_and_one_ready_result():
    c = GameInferenceCoordinator()
    generation = c.start_window(0, task_hand="right")
    for i in range(250):
        result = c.add_sample(range(16), now_ms=100 + (i + 1) * 8)
    assert result.status == c.STATUS_READY
    assert result.generation == generation
    assert len(result.window) == 250
    assert result.window[0] == list(range(16))
    assert c.add_sample(range(16), now_ms=2110).status == c.STATUS_LOCKED
    c.start_window(3000, task_hand="left")
    assert not c.close_window(generation)
    assert c.is_window_open
    assert not c.claim_prediction(generation)


def test_movement_fails_after_deadline_even_when_timer_has_not_run():
    c = GameInferenceCoordinator(window_size=1)
    generation = c.start_window(1000, task_hand="left")
    c.add_sample(range(16), now_ms=1010)
    c.claim_prediction(generation)
    assert c.allows_movement(generation, "left", 0, 0.9, now_ms=3499)
    assert not c.allows_movement(generation, "left", 0, 0.9, now_ms=3501)
