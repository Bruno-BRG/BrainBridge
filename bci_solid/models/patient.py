"""
Modelo de Paciente - Seguindo SRP

Esta classe representa um paciente e contém apenas as informações
e validações relacionadas a um paciente.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class Gender(Enum):
    """Enum para gênero"""
    MALE = "Masculino"
    FEMALE = "Feminino" 
    OTHER = "Outro"


class AffectedHand(Enum):
    """Enum para mão afetada"""
    LEFT = "Esquerda"
    RIGHT = "Direita"
    BOTH = "Ambas"
    NONE = "Nenhuma"


@dataclass
class Patient:
    """
    Modelo de paciente seguindo SRP
    
    Responsabilidade única: Representar e validar dados de um paciente
    """
    name: str
    age: int
    gender: Gender
    affected_hand: AffectedHand
    time_since_event: int  # em meses
    notes: Optional[str] = None
    patient_id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validação após inicialização"""
        self.validate()
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def validate(self) -> None:
        """
        Valida os dados do paciente
        
        Raises:
            ValueError: Se algum dado for inválido
        """
        if not self.name or len(self.name.strip()) < 2:
            raise ValueError("Nome deve ter pelo menos 2 caracteres")
        
        if not isinstance(self.age, int) or self.age < 0 or self.age > 150:
            raise ValueError("Idade deve ser um número entre 0 e 150")
        
        if not isinstance(self.gender, Gender):
            raise ValueError("Gênero deve ser um valor válido")
        
        if not isinstance(self.affected_hand, AffectedHand):
            raise ValueError("Mão afetada deve ser um valor válido")
        
        if not isinstance(self.time_since_event, int) or self.time_since_event < 0:
            raise ValueError("Tempo desde evento deve ser um número não-negativo")
    
    def to_dict(self) -> dict:
        """Converte o paciente para dicionário"""
        return {
            'id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender.value,
            'affected_hand': self.affected_hand.value,
            'time_since_event': self.time_since_event,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Patient':
        """Cria um paciente a partir de dicionário"""
        return cls(
            name=data['name'],
            age=data['age'],
            gender=Gender(data['gender']),
            affected_hand=AffectedHand(data['affected_hand']),
            time_since_event=data['time_since_event'],
            notes=data.get('notes'),
            patient_id=data.get('id'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )
    
    def update_from_dict(self, data: dict) -> None:
        """Atualiza o paciente com dados de um dicionário"""
        if 'name' in data:
            self.name = data['name']
        if 'age' in data:
            self.age = data['age']
        if 'gender' in data:
            self.gender = Gender(data['gender'])
        if 'affected_hand' in data:
            self.affected_hand = AffectedHand(data['affected_hand'])
        if 'time_since_event' in data:
            self.time_since_event = data['time_since_event']
        if 'notes' in data:
            self.notes = data['notes']
        
        # Revalidar após atualização
        self.validate()
    
    def __str__(self) -> str:
        """Representação string do paciente"""
        return f"Patient(id={self.patient_id}, name='{self.name}', age={self.age})"
    
    def __repr__(self) -> str:
        """Representação detalhada do paciente"""
        return (f"Patient(patient_id={self.patient_id}, name='{self.name}', "
                f"age={self.age}, gender={self.gender.value}, "
                f"affected_hand={self.affected_hand.value})")
