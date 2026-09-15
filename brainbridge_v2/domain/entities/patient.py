"""
Patient entity for the domain layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Patient:
    """
    Core patient entity independent from infrastructure details.
    """

    name: str
    age: int
    sex: str
    affected_hand: Optional[str]
    time_since_event: int
    notes: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None

    def validate(self) -> None:
        self.validate_affected_hand(self.affected_hand)
        if not self.name or not self.name.strip():
            raise ValueError("Nome do paciente eh obrigatorio.")
        if self.age < 0 or self.age > 150:
            raise ValueError("Idade invalida.")
        if self.time_since_event < 0:
            raise ValueError("Tempo desde evento nao pode ser negativo.")

    @staticmethod
    def validate_affected_hand(affected_hand: Optional[str]) -> None:
        if affected_hand not in (None, "left", "right"):
            raise ValueError("Mao afetada deve ser 'left' ou 'right'.")
