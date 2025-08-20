"""
DTOs para transferência de dados entre camadas
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CreatePatientDTO:
    """DTO para criação de paciente"""
    name: str
    age: int
    gender: str
    time_since_brain_event: int
    brain_event_type: str
    affected_side: str
    notes: str = ""


@dataclass
class PatientDTO:
    """DTO para representação de paciente"""
    id: Optional[int]
    name: str
    age: int
    gender: str
    time_since_brain_event: int
    brain_event_type: str
    affected_side: str
    notes: str
    created_at: Optional[datetime] = None
    
    @property
    def age_group(self) -> str:
        """Retorna a faixa etária"""
        if self.age < 18:
            return "Infantil"
        elif self.age < 65:
            return "Adulto"
        else:
            return "Idoso"
    
    @property
    def is_recent_event(self) -> bool:
        """Verifica se o evento foi recente"""
        return self.time_since_brain_event < 6


@dataclass
class UpdatePatientNotesDTO:
    """DTO para atualização de observações do paciente"""
    patient_id: int
    notes: str
