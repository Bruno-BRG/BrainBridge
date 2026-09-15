"""
Session entity for the domain layer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    """
    Runtime session entity independent from presentation details.
    """

    patient_id: int
    task_type: str
    recording_id: int
    started_at_epoch: float

    def validate(self) -> None:
        if self.patient_id <= 0:
            raise ValueError("Paciente da sessao deve ser valido.")
        if self.recording_id <= 0:
            raise ValueError("Gravacao da sessao deve ser valida.")
        if not self.task_type or not self.task_type.strip():
            raise ValueError("Tipo de tarefa da sessao eh obrigatorio.")
        if self.started_at_epoch <= 0:
            raise ValueError("Horario de inicio da sessao deve ser valido.")

    @property
    def game_mode(self) -> bool:
        return (self.task_type or "").strip().lower() == "jogo"

    @property
    def free_mode(self) -> bool:
        return (self.task_type or "").strip().lower() == "livre"
