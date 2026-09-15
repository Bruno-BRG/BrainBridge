"""
Controller that adapts presentation requests to training use cases.
"""

from typing import Callable, Optional

from brainbridge_v2.application.ports.inference_gateway import InferenceGateway
from brainbridge_v2.application.ports.training_gateway import TrainingGateway
from brainbridge_v2.application.use_cases.inference_use_cases import (
    GetLoadedModelUseCase,
)
from brainbridge_v2.application.use_cases.training_use_cases import (
    AutoLoadTrainedModelUseCase,
    TrainModelUseCase,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    TrainingPresenter,
    TrainingResultViewModel,
)


class TrainingController:
    """
    Presentation-facing controller for model training workflows.
    """

    def __init__(
        self,
        train_model_use_case: TrainModelUseCase,
        auto_load_trained_model_use_case: AutoLoadTrainedModelUseCase,
        get_loaded_model_use_case: GetLoadedModelUseCase,
        training_gateway=None,
    ):
        self._train_model_use_case = train_model_use_case
        self._auto_load_trained_model_use_case = auto_load_trained_model_use_case
        self._get_loaded_model_use_case = get_loaded_model_use_case
        self._training_gateway = training_gateway

    @classmethod
    def from_gateways(
        cls,
        training_gateway: TrainingGateway,
        inference_gateway: InferenceGateway,
    ) -> "TrainingController":
        return cls(
            train_model_use_case=TrainModelUseCase(training_gateway),
            auto_load_trained_model_use_case=AutoLoadTrainedModelUseCase(
                training_gateway,
                inference_gateway,
            ),
            get_loaded_model_use_case=GetLoadedModelUseCase(inference_gateway),
            training_gateway=training_gateway,
        )

    def patient_model_available(self, patient_id: int) -> bool:
        """True se o paciente ja tem calibracao (modelo publicado e valido)."""
        gateway = self._training_gateway
        if gateway is None or not hasattr(gateway, "patient_model_available"):
            return False
        try:
            return bool(gateway.patient_model_available(patient_id))
        except Exception:
            return False

    def train_model(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResultViewModel:
        result = self._train_model_use_case.execute(
            csv_file_path,
            patient_id,
            progress_callback=progress_callback,
        )
        return TrainingPresenter.present(result, auto_loaded=False)

    def train_and_load_model(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResultViewModel:
        result = self._auto_load_trained_model_use_case.execute(
            csv_file_path,
            patient_id,
            progress_callback=progress_callback,
        )
        loaded_model = self._get_loaded_model_use_case.execute()
        return TrainingPresenter.present(
            result,
            auto_loaded=True,
            loaded_model=loaded_model,
        )
