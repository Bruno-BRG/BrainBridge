"""
Entidade Patient - representa um paciente no sistema BCI
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class PatientId:
    """Value Object para identificação do paciente"""
    value: int
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("PatientId deve ser um número positivo")


@dataclass
class Patient:
    """
    Entidade Patient - representa um paciente no sistema BCI
    
    Invariantes de domínio:
    - Nome não pode ser vazio ou apenas espaços
    - Idade deve estar entre 0 e 150 anos
    - Tempo desde evento cerebral deve ser não-negativo
    """
    id: Optional[PatientId]
    name: str
    age: int
    gender: str
    time_since_brain_event: int  # meses
    brain_event_type: str
    affected_side: str
    notes: str
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Valida invariantes da entidade"""
        if not self.name or self.name.strip() == "":
            raise ValueError("Nome do paciente não pode ser vazio")
        
        if self.age < 0 or self.age > 150:
            raise ValueError("Idade deve estar entre 0 e 150 anos")
        
        if self.time_since_brain_event < 0:
            raise ValueError("Tempo desde evento cerebral deve ser não-negativo")
    
    def get_age_group(self) -> str:
        """Retorna a faixa etária do paciente"""
        if self.age < 18:
            return "Infantil"
        elif self.age < 65:
            return "Adulto"
        else:
            return "Idoso"
    
    def is_recent_event(self) -> bool:
        """Verifica se o evento cerebral foi recente (menos de 6 meses)"""
        return self.time_since_brain_event < 6
    
    def update_notes(self, new_notes: str) -> 'Patient':
        """Atualiza as observações do paciente"""
        return Patient(
            id=self.id,
            name=self.name,
            age=self.age,
            gender=self.gender,
            time_since_brain_event=self.time_since_brain_event,
            brain_event_type=self.brain_event_type,
            affected_side=self.affected_side,
            notes=new_notes,
            created_at=self.created_at
        )
