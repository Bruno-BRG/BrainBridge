import pytest

from brainbridge_v2.application.use_cases.training_use_cases import (
    AutoLoadTrainedModelUseCase,
    TrainModelUseCase,
)
from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.training_result import TrainingResult


class FakeTrainingGateway:
    def __init__(self):
        self.calls = []

    def train(self, csv_file_path, patient_id, progress_callback=None):
        self.calls.append((csv_file_path, patient_id))
        if progress_callback is not None:
            progress_callback("Preparando dados...")
        return TrainingResult(
            model_path=f"C:/models/patient_{patient_id}.keras",
            training_time_seconds=1.5,
            final_accuracy=0.8,
        )


class FakeInferenceGateway:
    def __init__(self):
        self.loaded_paths = []

    def load_model(self, model_path: str):
        self.loaded_paths.append(model_path)
        return ModelMetadata(path=model_path, name=model_path.split("/")[-1])

    def get_loaded_model(self):
        if not self.loaded_paths:
            return None
        latest = self.loaded_paths[-1]
        return ModelMetadata(path=latest, name=latest.split("/")[-1])

    def predict(self, eeg_window):
        raise NotImplementedError


def test_training_use_cases_train_and_auto_load():
    training_gateway = FakeTrainingGateway()
    inference_gateway = FakeInferenceGateway()
    progress_messages = []

    training_result = TrainModelUseCase(training_gateway).execute(
        "C:/recordings/train.csv",
        7,
        progress_callback=progress_messages.append,
    )
    auto_loaded_result = AutoLoadTrainedModelUseCase(
        training_gateway,
        inference_gateway,
    ).execute("C:/recordings/train.csv", 7)

    assert training_result.model_path.endswith("patient_7.keras")
    assert auto_loaded_result.model_path.endswith("patient_7.keras")
    assert inference_gateway.loaded_paths == ["C:/models/patient_7.keras"]
    assert progress_messages == ["Preparando dados..."]


@pytest.mark.parametrize("auto_load", [False, True])
def test_training_failure_propagates_without_loading_model(auto_load):
    error = RuntimeError("TensorFlow indisponivel")

    class FailingTrainingGateway:
        def train(self, csv_file_path, patient_id, progress_callback=None):
            raise error

    inference_gateway = FakeInferenceGateway()
    inference_gateway.load_model("existing.keras")
    gateway = FailingTrainingGateway()
    use_case = (
        AutoLoadTrainedModelUseCase(gateway, inference_gateway)
        if auto_load else TrainModelUseCase(gateway)
    )

    with pytest.raises(RuntimeError, match="TensorFlow indisponivel") as raised:
        use_case.execute("train.csv", 7)

    assert raised.value is error
    assert inference_gateway.loaded_paths == ["existing.keras"]
