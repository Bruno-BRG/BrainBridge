"""
Controller that adapts presentation requests to inference use cases.
"""

from typing import List, Optional, Sequence

from brainbridge_v2.application.ports.inference_gateway import InferenceGateway
from brainbridge_v2.application.ports.model_catalog_gateway import ModelCatalogGateway
from brainbridge_v2.application.use_cases.inference_use_cases import (
    GetLoadedModelUseCase,
    ListAvailableModelsUseCase,
    LoadModelUseCase,
    RunInferenceUseCase,
    SelectModelUseCase,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    InferencePresenter,
    ModelViewModel,
    PredictionViewModel,
)


class InferenceController:
    """
    Presentation-facing controller for model discovery, loading and inference.
    """

    def __init__(
        self,
        list_available_models_use_case: ListAvailableModelsUseCase,
        load_model_use_case: LoadModelUseCase,
        select_model_use_case: SelectModelUseCase,
        get_loaded_model_use_case: GetLoadedModelUseCase,
        run_inference_use_case: RunInferenceUseCase,
        inference_gateway: Optional[InferenceGateway] = None,
    ):
        self._list_available_models_use_case = list_available_models_use_case
        self._load_model_use_case = load_model_use_case
        self._select_model_use_case = select_model_use_case
        self._get_loaded_model_use_case = get_loaded_model_use_case
        self._run_inference_use_case = run_inference_use_case
        self._gateway = inference_gateway

    @classmethod
    def from_gateways(
        cls,
        model_catalog_gateway: ModelCatalogGateway,
        inference_gateway: InferenceGateway,
    ) -> "InferenceController":
        return cls(
            list_available_models_use_case=ListAvailableModelsUseCase(
                model_catalog_gateway
            ),
            load_model_use_case=LoadModelUseCase(inference_gateway),
            select_model_use_case=SelectModelUseCase(
                model_catalog_gateway,
                inference_gateway,
            ),
            get_loaded_model_use_case=GetLoadedModelUseCase(inference_gateway),
            run_inference_use_case=RunInferenceUseCase(inference_gateway),
            inference_gateway=inference_gateway,
        )

    def list_models(self) -> List[ModelViewModel]:
        models = self._list_available_models_use_case.execute()
        return [InferencePresenter.present_model(model) for model in models]

    def load_model(self, model_path: str) -> ModelViewModel:
        model = self._load_model_use_case.execute(model_path)
        return InferencePresenter.present_model(model)

    def select_model(self, model_path: str) -> ModelViewModel:
        model = self._select_model_use_case.execute(model_path)
        return InferencePresenter.present_model(model)

    def load_latest_model(self) -> ModelViewModel:
        models = self.list_models()
        if not models:
            raise ValueError("Nenhum modelo TensorFlow (.keras/.h5) foi encontrado.")
        return self.select_model(models[0].path)

    def get_loaded_model(self) -> Optional[ModelViewModel]:
        model = self._get_loaded_model_use_case.execute()
        if model is None:
            return None
        return InferencePresenter.present_model(model)

    def has_loaded_model(self) -> bool:
        return self.get_loaded_model() is not None

    def predict(self, eeg_window: Sequence[Sequence[float]], *,
                input_fs: Optional[float] = None) -> PredictionViewModel:
        normalized_window: Sequence[Sequence[float]]
        if hasattr(eeg_window, "tolist"):
            normalized_window = eeg_window.tolist()
        else:
            normalized_window = eeg_window

        result = self._run_inference_use_case.execute(normalized_window,
                                                      input_fs=input_fs)
        return InferencePresenter.present_prediction(result)

    # -- Euclidean Alignment (calibracao online) ---------------------------
    def ea_required(self) -> bool:
        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "ea_enabled"):
            return False
        try:
            return bool(gateway.ea_enabled())
        except Exception:
            return False

    def ea_calibrated(self) -> bool:
        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "ea_is_calibrated"):
            return True
        try:
            return bool(gateway.ea_is_calibrated())
        except Exception:
            return True

    def ea_progress(self) -> tuple[int, int]:
        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "ea_progress"):
            return (1, 1)
        try:
            return tuple(gateway.ea_progress())
        except Exception:
            return (1, 1)

    def ea_observe_sample(self, sample) -> bool:
        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "ea_observe_sample"):
            return True
        try:
            return bool(gateway.ea_observe_sample(sample))
        except Exception:
            return True

    def ea_reset(self) -> None:
        gateway = self._gateway
        if gateway is not None and hasattr(gateway, "ea_reset"):
            try:
                gateway.ea_reset()
            except Exception:
                pass

    # -- RL online ---------------------------------------------------------
    def _rl_gateway(self):
        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "rl_online_update"):
            raise RuntimeError("Gateway sem suporte a RL.")
        return gateway

    def rl_snapshot(self) -> bool:
        try:
            return bool(self._rl_gateway().rl_snapshot_weights())
        except Exception:
            return False

    def rl_has_snapshot(self) -> bool:
        try:
            return bool(self._rl_gateway().rl_has_snapshot())
        except Exception:
            return False

    def rl_restore(self) -> bool:
        try:
            return bool(self._rl_gateway().rl_restore_snapshot())
        except Exception:
            return False

    def rl_updates_count(self) -> int:
        try:
            return int(self._rl_gateway().rl_updates_count())
        except Exception:
            return 0

    def rl_online_update(self, windows, labels, *, sample_weights=None,
                         epochs: int = 3, lr: float = 5e-5,
                         freeze_backbone: bool = True) -> dict:
        return self._rl_gateway().rl_online_update(
            windows, labels, sample_weights=sample_weights,
            epochs=epochs, lr=lr, freeze_backbone=freeze_backbone)
