"""
Inference-related use cases.
"""

from typing import List, Optional, Sequence

from brainbridge_v2.application.ports.inference_gateway import InferenceGateway
from brainbridge_v2.application.ports.model_catalog_gateway import ModelCatalogGateway
from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult


class ListAvailableModelsUseCase:
    """
    Returns the available TensorFlow models from the configured catalog.
    """

    def __init__(self, model_catalog_gateway: ModelCatalogGateway):
        self._model_catalog_gateway = model_catalog_gateway

    def execute(self) -> List[ModelMetadata]:
        models = self._model_catalog_gateway.list_models()
        for model in models:
            model.validate()
        return models


class LoadModelUseCase:
    """
    Loads a model directly by path.
    """

    def __init__(self, inference_gateway: InferenceGateway):
        self._inference_gateway = inference_gateway

    def execute(self, model_path: str) -> ModelMetadata:
        normalized_path = str(model_path).strip()
        if not normalized_path:
            raise ValueError("Caminho do modelo eh obrigatorio.")
        model = self._inference_gateway.load_model(normalized_path)
        model.validate()
        return model


class SelectModelUseCase:
    """
    Selects a discovered model and loads it for inference.
    """

    def __init__(
        self,
        model_catalog_gateway: ModelCatalogGateway,
        inference_gateway: InferenceGateway,
    ):
        self._model_catalog_gateway = model_catalog_gateway
        self._inference_gateway = inference_gateway

    def execute(self, model_path: str) -> ModelMetadata:
        normalized_path = str(model_path).strip()
        if not normalized_path:
            raise ValueError("Caminho do modelo selecionado eh obrigatorio.")

        available_models = {
            model.path: model for model in self._model_catalog_gateway.list_models()
        }
        if normalized_path not in available_models:
            raise ValueError("Modelo selecionado nao foi encontrado no catalogo.")

        selected_model = self._inference_gateway.load_model(normalized_path)
        selected_model.validate()
        return selected_model


class GetLoadedModelUseCase:
    """
    Returns the currently loaded model, if any.
    """

    def __init__(self, inference_gateway: InferenceGateway):
        self._inference_gateway = inference_gateway

    def execute(self) -> Optional[ModelMetadata]:
        model = self._inference_gateway.get_loaded_model()
        if model is not None:
            model.validate()
        return model


class RunInferenceUseCase:
    """
    Runs inference on an EEG window using the currently loaded model.
    """

    def __init__(self, inference_gateway: InferenceGateway):
        self._inference_gateway = inference_gateway

    def execute(self, eeg_window: Sequence[Sequence[float]], *,
                input_fs: Optional[float] = None) -> PredictionResult:
        if len(eeg_window) == 0:
            raise ValueError("Janela EEG nao pode ser vazia.")
        try:
            result = self._inference_gateway.predict(eeg_window, input_fs=input_fs)
        except TypeError:
            # Gateways legados/fakes sem input_fs.
            result = self._inference_gateway.predict(eeg_window)
        result.validate()
        return result
