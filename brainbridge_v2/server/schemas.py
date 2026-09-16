"""Schemas Pydantic da API (contrato REST/WebSocket <-> frontend)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    name: str
    age: int = 0
    sex: str = ""
    affected_hand: Optional[str] = None
    time_since_event: int = 0
    notes: str = ""


class AffectedHandUpdate(BaseModel):
    affected_hand: str


class RecordingStart(BaseModel):
    patient_id: int
    filename: str
    task_type: str
    notes: str = ""


class RecordingStop(BaseModel):
    duration_seconds: int = 0


class SessionStart(BaseModel):
    patient_id: int
    task_type: str
    recording_id: int
    started_at_epoch: Optional[float] = None


class MarkerRegister(BaseModel):
    marker_type: str = Field(description="T1, T2, BASELINE...")
    task_type: str = "livre"


class BaselineStart(BaseModel):
    duration_seconds: int = 300


class EEGConnect(BaseModel):
    host: str = "localhost"
    port: int = 12345
    simulate: bool = False
    stream: str = Field(default="raw", description="raw (timeSeriesRaw) | filtered (timeSeries)")


class UnityAction(BaseModel):
    direction: str = Field(description="esquerda | direita")


class UnitySession(BaseModel):
    nome: str
    nivel: int
    lado: str
    tarefa: str
    sessoes: int = 0


class UnityEndSession(BaseModel):
    message: str = ""


class UnityPublishSession(BaseModel):
    patient_id: int
    task_type: str = "jogo"


class ESP32Connect(BaseModel):
    port: str | None = Field(default=None, description="Porta serial; vazio = autodetectar")


class ESP32Action(BaseModel):
    direction: str = Field(description="esquerda | direita")


class ModelLoad(BaseModel):
    path: str


class InferencePredict(BaseModel):
    window: List[List[float]] = Field(description="Janela (T, C), RAW")
    input_fs: Optional[float] = None


class RLUpdate(BaseModel):
    windows: List[List[List[float]]] = Field(description="Lote (N, T, C)")
    labels: List[int]
    weights: Optional[List[float]] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None


class TrainingStart(BaseModel):
    csv_file_path: str
    patient_id: int
    auto_load: bool = True


class ConfigUpdate(BaseModel):
    values: Dict[str, Any] = Field(description="Chaves de runtime_config")


class TrialsCheck(BaseModel):
    csv_file_path: str
    required: Optional[int] = None
