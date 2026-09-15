import numpy as np
import pytest
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from brainbridge_v2.infrastructure.ml.eeg_pipeline import write_pipeline_manifest, preprocess_window

from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter,
)


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


class FakeModel:
    input_shape = (None, 250, 16)
    output_shape = (None, 2)


class FakeTensorFlowAdapter:
    def __init__(self):
        self.model = None
        self.predicted_batches = []

    def load_model(self, model_path: str):
        self.model = FakeModel()
        return self.model

    def predict(self, data):
        self.predicted_batches.append(data)
        return np.array([[0.25, 0.75]], dtype=np.float32)


def test_reload_and_warmup_wait_for_prediction_without_blocking_metadata(tmp_path):
    entered, release, loading, loaded = Event(), Event(), Event(), Event()
    class BlockingAdapter(FakeTensorFlowAdapter):
        def predict(self, data):
            entered.set()
            assert release.wait(3)
            return super().predict(data)

    adapters = iter([BlockingAdapter(), FakeTensorFlowAdapter()])
    gateway = TensorFlowInferenceGatewayAdapter(adapter_factory=lambda: next(adapters), warmup_enabled=False)
    path = tmp_path / "test.keras"
    write_pipeline_manifest(path, training_source_stages=["raw"])
    original = gateway.load_model(str(path))
    gateway._warmup_enabled = True
    def reload():
        loading.set()
        result = gateway.load_model(str(path))
        loaded.set()
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        prediction = pool.submit(gateway.predict, np.random.default_rng(1).normal(size=(250, 16)))
        try:
            assert entered.wait(3)
            load = pool.submit(reload)
            assert loading.wait(3)
            assert not loaded.wait(0.05)
            assert gateway.get_loaded_model() is original
        finally:
            release.set()
        assert prediction.result(timeout=3).predicted_index == 1
        assert load.result(timeout=3) is gateway.get_loaded_model()
    assert len(gateway._adapter.predicted_batches) == 1  # warmup


def test_tensorflow_inference_gateway_adapter_loads_model_and_predicts(tmp_path):
    created_adapters = []

    def build_adapter():
        adapter = FakeTensorFlowAdapter()
        created_adapters.append(adapter)
        return adapter

    gateway = TensorFlowInferenceGatewayAdapter(adapter_factory=build_adapter)

    path = tmp_path / "test.keras"
    write_pipeline_manifest(path, training_source_stages=["raw"])
    model = gateway.load_model(str(path))
    window = np.random.default_rng(3).normal(size=(250, 16))
    result = gateway.predict(window)

    assert model.expected_time_steps == 250
    assert model.expected_channels == 16
    assert gateway.get_loaded_model() is not None
    assert created_adapters[0].predicted_batches[0].shape == (1, 250, 16)
    np.testing.assert_array_equal(created_adapters[0].predicted_batches[1][0], preprocess_window(window))
    assert result.predicted_index == 1
    assert result.right_probability == 0.75


def test_tensorflow_inference_gateway_adapter_can_skip_warmup(tmp_path):
    created_adapters = []

    def build_adapter():
        adapter = FakeTensorFlowAdapter()
        created_adapters.append(adapter)
        return adapter

    gateway = TensorFlowInferenceGatewayAdapter(
        adapter_factory=build_adapter,
        warmup_enabled=False,
    )

    path = tmp_path / "test.keras"
    write_pipeline_manifest(path, training_source_stages=["raw"])
    gateway.load_model(str(path))

    assert created_adapters[0].predicted_batches == []


def test_legacy_load_fails_closed(tmp_path):
    gateway = TensorFlowInferenceGatewayAdapter(adapter_factory=FakeTensorFlowAdapter)
    with pytest.raises(ValueError, match="retrain"):
        gateway.load_model(str(tmp_path / "legacy.keras"))
    assert gateway.get_loaded_model() is None


def test_reload_warmup_failure_clears_previous_and_new_model(tmp_path):
    class FailingAdapter(FakeTensorFlowAdapter):
        def predict(self, data):
            raise RuntimeError("warmup failed")

    adapters = iter([FakeTensorFlowAdapter(), FailingAdapter()])
    gateway = TensorFlowInferenceGatewayAdapter(adapter_factory=lambda: next(adapters), warmup_enabled=True)
    path = tmp_path / "test.keras"
    write_pipeline_manifest(path, training_source_stages=["raw"])
    gateway.load_model(str(path))
    assert gateway.get_loaded_model() is not None
    with pytest.raises(RuntimeError, match="warmup failed"):
        gateway.load_model(str(path))
    assert gateway.get_loaded_model() is None
    assert gateway._adapter is None
    with pytest.raises(RuntimeError, match="Nenhum modelo"):
        gateway.predict(np.zeros((250, 16)))
