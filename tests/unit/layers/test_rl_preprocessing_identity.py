"""RL online usa o mesmo pre-processamento do treino/inferencia (sem TF real)."""

import sys
import types

import numpy as np
import pytest

from brainbridge_v2.infrastructure.ml.eeg_pipeline import preprocess_window
from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter,
)


class _Weights:
    input_shape = (None, 250, 16)
    output_shape = (None, 2)

    def __init__(self):
        self._w = [np.zeros((2, 2), dtype=np.float64)]

    def get_weights(self):
        return [w.copy() for w in self._w]

    def set_weights(self, weights):
        self._w = [np.array(w, copy=True) for w in weights]


class _Clone(_Weights):
    layers = []

    def __init__(self, captured):
        super().__init__()
        self._captured = captured

    def compile(self, **kwargs):
        return None

    def fit(self, X, y, **kwargs):
        self._captured["X"] = np.asarray(X)
        self._captured["y"] = np.asarray(y)
        self._captured["kwargs"] = kwargs

        class _History:
            history = {"loss": [0.42]}

        return _History()


class _FakeAdapter:
    def __init__(self, model):
        self.model = model

    def load_model(self, model_path):
        return self.model

    def predict(self, batch):
        return np.array([[0.5, 0.5]], dtype=np.float32)


def _gateway_with_fake_tf(monkeypatch, tmp_path):
    captured = {}
    models_mod = types.ModuleType("tensorflow.keras.models")
    models_mod.clone_model = lambda base: _Clone(captured)
    optim_mod = types.ModuleType("tensorflow.keras.optimizers")
    optim_mod.Adam = lambda learning_rate: ("adam", learning_rate)
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras", types.ModuleType("tensorflow.keras"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", optim_mod)

    from brainbridge_v2.infrastructure.ml.eeg_pipeline import write_pipeline_manifest
    path = tmp_path / "rl_base.keras"
    write_pipeline_manifest(path, training_source_stages=["raw"])
    gateway = TensorFlowInferenceGatewayAdapter(
        adapter_factory=lambda: _FakeAdapter(_Weights()), warmup_enabled=False)
    gateway.load_model(str(path))
    return gateway, captured


def test_rl_online_update_preprocesses_raw_windows_like_training(monkeypatch, tmp_path):
    gateway, captured = _gateway_with_fake_tf(monkeypatch, tmp_path)
    rng = np.random.default_rng(11)
    # Escala RAW (dezenas de uV), como o frontend envia pelo WebSocket.
    raw = [rng.normal(0.0, 40.0, size=(250, 16)) for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    out = gateway.rl_online_update(raw, labels, sample_weights=[1.0, 3.0, 1.0, 3.0, 1.0],
                                   epochs=2, lr=1e-4)
    assert out == {"n": 5, "loss": 0.42}
    assert captured["X"].shape == (5, 250, 16)
    assert captured["X"].dtype == np.float32
    np.testing.assert_array_equal(captured["y"], np.asarray(labels, dtype=np.int32))
    for i in range(5):
        np.testing.assert_allclose(captured["X"][i], preprocess_window(raw[i]), rtol=1e-5)


def test_rl_online_update_maps_channels_like_inference(monkeypatch, tmp_path):
    gateway, captured = _gateway_with_fake_tf(monkeypatch, tmp_path)
    rng = np.random.default_rng(12)
    raw = [rng.normal(0.0, 40.0, size=(250, 8)) for _ in range(3)]
    out = gateway.rl_online_update(raw, [0, 1, 1])
    assert out["n"] == 3
    assert captured["X"].shape == (3, 250, 16)


def test_rl_online_update_rejects_invalid_batches(monkeypatch, tmp_path):
    gateway, _ = _gateway_with_fake_tf(monkeypatch, tmp_path)
    rng = np.random.default_rng(13)
    good = [rng.normal(size=(250, 16)) for _ in range(2)]
    with pytest.raises(ValueError):
        gateway.rl_online_update([], [])
    with pytest.raises(ValueError):
        gateway.rl_online_update(good, [0])
    with pytest.raises(ValueError):
        gateway.rl_online_update(good, [0, 2])
    with pytest.raises(ValueError):
        gateway.rl_online_update([np.zeros((4, 16))], [0])
