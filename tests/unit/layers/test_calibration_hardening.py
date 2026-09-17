"""Endurecimento da calibracao (mesmo paciente) sem mudar a UX.

FT com aumento leve + peso de classe, defaults 10ep/LR 5e-5, RL com
replica com ruido, e higiene da selecao da base (sem metrics_/candidatos).
"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from brainbridge_v2.application.runtime_config import get_runtime
from brainbridge_v2.infrastructure.ml import trainer
from brainbridge_v2.infrastructure.ml.eeg_pipeline import write_pipeline_manifest


def test_runtime_defaults_calibracao_padrao():
    assert get_runtime("calib_trials_required") == 10
    assert get_runtime("calib_epochs") == 10
    assert get_runtime("calib_lr") == 5e-5
    assert get_runtime("rl_augment") is True


def _ft_mocks(monkeypatch):
    x = np.random.default_rng(3).normal(size=(8, 250, 16)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    g = np.array(["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"])
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(trainer, "_load_openbci_csv",
                          lambda path, **kw: (np.ones((250, 16)), [""] * 250))
    monkeypatch.setattr(trainer, "_create_windows_ht",
                          lambda *a, **k: (x, y, g))
    monkeypatch.setattr(Path, "read_bytes", lambda path: str(path).encode())


class _FakeFTModel:
    input_shape = (None, 250, 16)
    output_shape = (None, 2)
    def __init__(self):
        self.fit_calls = []
        self.layers = []
    def fit(self, *a, **k):
        self.fit_calls.append((np.asarray(a[0]), np.asarray(a[1]), k))
        class _H:
            history = {"accuracy": [0.5], "loss": [0.7]}
        return _H()
    def evaluate(self, x, y, verbose=0):
        return 0.7, 0.5
    def save(self, path):
        Path(path).write_text("fake", encoding="utf-8")


def test_ft_augment_triplica_treino_e_pesa_classes(monkeypatch):
    _ft_mocks(monkeypatch)
    fake = _FakeFTModel()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.keras"
        base.write_text("b", encoding="utf-8")
        write_pipeline_manifest(base, training_source_stages=["raw"])
        monkeypatch.setattr(trainer, "MODELS_DIR", Path(td))
        res = trainer.train_from_csvs(["s.csv"], model_name="ft1",
                                        base_model_path=str(base), epochs=1,
                                        model_loader=lambda p: fake,
                                        finetune_augment=True,
                                        finetune_class_weight=True)
        assert res.model_path.endswith("ft1.keras")
    assert len(fake.fit_calls) == 1
    xf, yf, kw = fake.fit_calls[0]
    assert len(xf) == 12 and len(yf) == 12  # 4 treino x3; val intacta
    assert kw.get("class_weight") == {0: 1.0, 1: 1.0}
    assert "sample_weight" not in kw


def test_ft_sem_flags_mantem_comportamento(monkeypatch):
    _ft_mocks(monkeypatch)
    fake = _FakeFTModel()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.keras"
        base.write_text("b", encoding="utf-8")
        write_pipeline_manifest(base, training_source_stages=["raw"])
        monkeypatch.setattr(trainer, "MODELS_DIR", Path(td))
        trainer.train_from_csvs(["s.csv"], model_name="ft0",
                                  base_model_path=str(base), epochs=1,
                                  model_loader=lambda p: fake)
    xf, yf, kw = fake.fit_calls[0]
    assert len(xf) == 4
    assert "class_weight" not in kw


def test_base_ignora_metrics_e_candidatos(tmp_path, monkeypatch):
    from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
        ModelTrainingGatewayAdapter)
    import time
    good = tmp_path / "generalized_eegnet.keras"
    good.write_bytes(b"good")
    write_pipeline_manifest(good, training_source_stages=["raw"])
    time.sleep(0.02)
    pol = tmp_path / "metrics_eval.keras"
    pol.write_bytes(b"metrics")
    write_pipeline_manifest(pol, training_source_stages=["unknown"])
    cand = tmp_path / "patient_9_candidate_x.keras"
    cand.write_bytes(b"cand")
    write_pipeline_manifest(cand, training_source_stages=["raw"])
    gw = ModelTrainingGatewayAdapter(trainer_module=object(), models_dir=tmp_path)
    assert gw._latest_base_model_path() == good


def _rl_gateway(monkeypatch, tmp_path):
    from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
        TensorFlowInferenceGatewayAdapter)
    captured = {}
    class _W:
        input_shape = (None, 250, 16)
        output_shape = (None, 2)
        def __init__(self):
            self._w = [np.zeros((2, 2))]
        def get_weights(self):
            return [w.copy() for w in self._w]
        def set_weights(self, w):
            self._w = [np.array(x, copy=True) for x in w]
    class _C(_W):
        layers = []
        def compile(self, **k):
            return None
        def fit(self, x, y, **k):
            captured["X"] = np.asarray(x)
            captured["sw"] = k.get("sample_weight")
            class _H:
                history = {"loss": [0.5]}
            return _H()
    class _A:
        def __init__(self, m):
            self.model = m
        def load_model(self, p):
            return self.model
        def predict(self, b):
            return np.array([[0.5, 0.5]], dtype=np.float32)
    mods = types.ModuleType("tensorflow.keras.models")
    mods.clone_model = lambda b: _C()
    opts = types.ModuleType("tensorflow.keras.optimizers")
    opts.Adam = lambda learning_rate: ("adam", learning_rate)
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras", types.ModuleType("tensorflow.keras"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", mods)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", opts)
    p = tmp_path / "b.keras"
    write_pipeline_manifest(p, training_source_stages=["raw"])
    gw = TensorFlowInferenceGatewayAdapter(adapter_factory=lambda: _A(_W()),
                                             warmup_enabled=False)
    gw.load_model(str(p))
    return gw, captured


def test_rl_augment_duplica_lote_com_mesmo_peso(monkeypatch, tmp_path):
    gw, cap = _rl_gateway(monkeypatch, tmp_path)
    rng = np.random.default_rng(5)
    raw = [rng.normal(0.0, 40.0, size=(250, 16)) for _ in range(4)]
    out = gw.rl_online_update(raw, [0, 1, 0, 1],
                               sample_weights=np.array([1.0, 3.0, 1.0, 3.0]),
                               epochs=1, augment=True)
    assert out["n"] == 8
    assert cap["X"].shape == (8, 250, 16)
    assert list(np.asarray(cap["sw"]).reshape(-1)) == [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]


def test_rl_sem_augment_mantem_lote(monkeypatch, tmp_path):
    gw, cap = _rl_gateway(monkeypatch, tmp_path)
    rng = np.random.default_rng(6)
    raw = [rng.normal(0.0, 40.0, size=(250, 16)) for _ in range(4)]
    out = gw.rl_online_update(raw, [0, 1, 0, 1], epochs=1)
    assert out["n"] == 4
    assert cap["X"].shape == (4, 250, 16)
