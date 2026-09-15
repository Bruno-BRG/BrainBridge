"""Calibracao obrigatoria ponderada + RL online (sem TensorFlow real)."""

import json
import numpy as np
import pytest

from brainbridge_v2.application.runtime_config import (
    get_runtime, reset_runtime, set_runtime)


@pytest.fixture(autouse=True)
def _clean_runtime():
    reset_runtime()
    yield
    reset_runtime()


def test_runtime_overrides_and_unknown_key():
    assert get_runtime("calib_trials_required") == 10
    assert get_runtime("rl_enabled") is False
    assert get_runtime("rl_batch_k") == 5
    set_runtime("rl_enabled", True)
    set_runtime("rl_batch_k", 3)
    set_runtime("calib_trials_required", 6)
    assert get_runtime("rl_enabled") is True
    assert get_runtime("rl_batch_k") == 3
    assert get_runtime("calib_trials_required") == 6
    with pytest.raises(ValueError):
        set_runtime("nope", 1)


def _write_openbci_csv(path, n_samples=600, markers=None):
    import csv
    markers = markers or {}
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["%Sample Rate = 125 Hz"])
        w.writerow(["%Signal Stage = raw"])
        w.writerow(["Sample Index"] + [f"EXG Channel {i}" for i in range(16)] + ["Annotations"])
        for i in range(n_samples):
            row = [i] + rng.normal(size=16).tolist() + [markers.get(i, "")]
            w.writerow(row)


def test_count_labeled_trials(tmp_path):
    from brainbridge_v2.infrastructure.ml.trainer import (
        count_labeled_trials, count_labeled_trials_total)

    path = tmp_path / "s.csv"
    # T1 em 0 (segmento 0..300), T2 em 300 (300..550, T0 fecha): 2 trials.
    _write_openbci_csv(path, n_samples=600, markers={0: "T1", 300: "T2", 550: "T0"})
    counts = count_labeled_trials(path)
    assert counts == {"T1": 1, "T2": 1}
    assert count_labeled_trials_total(path) == 2
    # Segmento curto (<250) nao conta.
    _write_openbci_csv(path, n_samples=300, markers={0: "T1", 100: "T2"})
    assert count_labeled_trials_total(path) == 0


def test_patient_model_available(tmp_path):
    from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
        ModelTrainingGatewayAdapter)
    from brainbridge_v2.infrastructure.ml.eeg_pipeline import write_pipeline_manifest

    gw = ModelTrainingGatewayAdapter(trainer_module=object(), models_dir=tmp_path)
    assert gw.patient_model_available(1) is False
    assert gw.patient_model_available(-3) is False
    model = tmp_path / "patient_1_candidate_abc.keras"
    model.write_bytes(b"fake-model")
    write_pipeline_manifest(model, training_source_stages=["raw"])
    (tmp_path / "patient_1.json").write_text(
        json.dumps({"model_path": model.name}))
    assert gw.patient_model_available(1) is True
    assert gw.patient_model_available(2) is False


class _StubModel:
    def __init__(self):
        self._weights = [np.zeros((4, 2)), np.zeros(2)]

    def get_weights(self):
        return [w.copy() for w in self._weights]

    def set_weights(self, weights):
        self._weights = [np.array(w, copy=True) for w in weights]


class _StubAdapter:
    def __init__(self, model):
        self.model = model


def _gateway_with_stub():
    from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
        TensorFlowInferenceGatewayAdapter)
    gw = TensorFlowInferenceGatewayAdapter.__new__(
        TensorFlowInferenceGatewayAdapter)
    import threading
    gw._lock = threading.RLock()
    gw._adapter = _StubAdapter(_StubModel())
    gw._loaded_model = object()
    gw._rl_snapshot = None
    gw._rl_updates_applied = 0
    return gw


def test_rl_snapshot_restore_and_counter():
    gw = _gateway_with_stub()
    assert gw.rl_has_snapshot() is False
    assert gw.rl_updates_count() == 0
    assert gw.rl_snapshot_weights() is True
    assert gw.rl_has_snapshot() is True
    gw._adapter.model._weights = [np.ones((4, 2)), np.ones(2)]
    assert gw.rl_restore_snapshot() is True
    assert gw.rl_has_snapshot() is False
    np.testing.assert_array_equal(gw._adapter.model._weights[0], np.zeros((4, 2)))
    assert gw.rl_restore_snapshot() is False


def test_inference_controller_rl_passthrough():
    from brainbridge_v2.interface_adapters.controllers.inference_controller import (
        InferenceController)

    class _FakeGW:
        def __init__(self):
            self.calls = []

        def rl_online_update(self, windows, labels, **kwargs):
            self.calls.append((len(windows), list(labels), kwargs))
            return {"n": len(windows), "loss": 0.5}

        def rl_snapshot_weights(self):
            return True

        def rl_has_snapshot(self):
            return True

        def rl_restore_snapshot(self):
            return True

        def rl_updates_count(self):
            return 7

    fake = _FakeGW()
    ctrl = InferenceController.__new__(InferenceController)
    ctrl._gateway = fake
    out = ctrl.rl_online_update([np.zeros((250, 16))], [1],
                                sample_weights=[3.0], epochs=2, lr=1e-4)
    assert out == {"n": 1, "loss": 0.5}
    assert fake.calls[0][2]["sample_weights"] == [3.0]
    assert ctrl.rl_snapshot() is True
    assert ctrl.rl_restore() is True
    assert ctrl.rl_updates_count() == 7


def test_training_controller_patient_model_available(tmp_path):
    from brainbridge_v2.interface_adapters.controllers.training_controller import (
        TrainingController)
    from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
        ModelTrainingGatewayAdapter)

    gw = ModelTrainingGatewayAdapter(trainer_module=object(), models_dir=tmp_path)
    ctrl = TrainingController.__new__(TrainingController)
    ctrl._training_gateway = gw
    assert ctrl.patient_model_available(9) is False


def test_run_inference_legacy_gateway_without_input_fs():
    from brainbridge_v2.application.use_cases.inference_use_cases import (
        RunInferenceUseCase)
    from brainbridge_v2.domain.entities.prediction_result import PredictionResult

    class _Legacy:
        def predict(self, window):  # sem input_fs
            return PredictionResult(predicted_index=0, confidence=0.9,
                                    probabilities=(0.9, 0.1))

    out = RunInferenceUseCase(_Legacy()).execute([[0.0] * 16] * 250, input_fs=128.0)
    assert out.predicted_index == 0
