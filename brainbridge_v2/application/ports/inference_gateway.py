"""
Port definition for inference operations.
"""

from typing import Optional, Protocol, Sequence

from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult


class InferenceGateway(Protocol):
    def load_model(self, model_path: str) -> ModelMetadata:
        ...

    def get_loaded_model(self) -> Optional[ModelMetadata]:
        ...

    def predict(self, eeg_window: Sequence[Sequence[float]], *,
                input_fs: Optional[float] = None) -> PredictionResult:
        ...
