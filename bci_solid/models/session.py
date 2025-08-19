"""
Modelo de Sessão - Seguindo SRP

Esta classe representa uma sessão de gravação e contém apenas as informações
e validações relacionadas a uma sessão.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class SessionType(Enum):
    """Tipos de sessão"""
    BASELINE = "baseline"
    TRAINING = "treino"
    TEST = "teste"
    GAME = "jogo"


class SessionStatus(Enum):
    """Status da sessão"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Session:
    """
    Modelo de sessão seguindo SRP
    
    Responsabilidade única: Representar e gerenciar dados de uma sessão
    """
    patient_id: int
    session_type: SessionType
    filename: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: SessionStatus = SessionStatus.CREATED
    notes: Optional[str] = None
    session_id: Optional[int] = None
    duration_seconds: Optional[int] = None
    marker_count: dict = None  # Contadores de marcadores {'T1': 0, 'T2': 0, 'T0': 0}
    
    def __post_init__(self):
        """Validação e inicialização após criação"""
        self.validate()
        if self.marker_count is None:
            self.marker_count = {'T1': 0, 'T2': 0, 'T0': 0, 'BASELINE': 0}
    
    def validate(self) -> None:
        """
        Valida os dados da sessão
        
        Raises:
            ValueError: Se algum dado for inválido
        """
        if not isinstance(self.patient_id, int) or self.patient_id <= 0:
            raise ValueError("ID do paciente deve ser um inteiro positivo")
        
        if not isinstance(self.session_type, SessionType):
            raise ValueError("Tipo de sessão deve ser um valor válido")
        
        if not self.filename or not self.filename.strip():
            raise ValueError("Nome do arquivo é obrigatório")
        
        if not isinstance(self.status, SessionStatus):
            raise ValueError("Status da sessão deve ser um valor válido")
        
        if self.start_time and self.end_time:
            if self.end_time <= self.start_time:
                raise ValueError("Horário de fim deve ser posterior ao de início")
    
    def start_session(self) -> None:
        """Inicia a sessão"""
        if self.status != SessionStatus.CREATED:
            raise ValueError("Sessão só pode ser iniciada quando está no status CREATED")
        
        self.start_time = datetime.now()
        self.status = SessionStatus.ACTIVE
    
    def pause_session(self) -> None:
        """Pausa a sessão"""
        if self.status != SessionStatus.ACTIVE:
            raise ValueError("Sessão só pode ser pausada quando está ACTIVE")
        
        self.status = SessionStatus.PAUSED
    
    def resume_session(self) -> None:
        """Resume a sessão"""
        if self.status != SessionStatus.PAUSED:
            raise ValueError("Sessão só pode ser resumida quando está PAUSED")
        
        self.status = SessionStatus.ACTIVE
    
    def complete_session(self) -> None:
        """Completa a sessão"""
        if self.status not in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            raise ValueError("Sessão só pode ser completada quando está ACTIVE ou PAUSED")
        
        self.end_time = datetime.now()
        self.status = SessionStatus.COMPLETED
        
        if self.start_time:
            self.duration_seconds = int((self.end_time - self.start_time).total_seconds())
    
    def cancel_session(self) -> None:
        """Cancela a sessão"""
        if self.status == SessionStatus.COMPLETED:
            raise ValueError("Sessão já completada não pode ser cancelada")
        
        self.status = SessionStatus.CANCELLED
        if not self.end_time:
            self.end_time = datetime.now()
    
    def set_error(self, error_message: str) -> None:
        """Define status de erro"""
        self.status = SessionStatus.ERROR
        self.notes = f"ERRO: {error_message}"
        if not self.end_time:
            self.end_time = datetime.now()
    
    def add_marker(self, marker_type: str) -> None:
        """Adiciona contador de marcador"""
        if marker_type in self.marker_count:
            self.marker_count[marker_type] += 1
        else:
            self.marker_count[marker_type] = 1
    
    def get_duration_string(self) -> str:
        """Retorna duração formatada como string"""
        if not self.duration_seconds:
            return "00:00:00"
        
        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        seconds = self.duration_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def is_active(self) -> bool:
        """Verifica se a sessão está ativa"""
        return self.status == SessionStatus.ACTIVE
    
    def to_dict(self) -> dict:
        """Converte a sessão para dicionário"""
        return {
            'id': self.session_id,
            'patient_id': self.patient_id,
            'session_type': self.session_type.value,
            'filename': self.filename,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'notes': self.notes,
            'duration_seconds': self.duration_seconds,
            'marker_count': self.marker_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """Cria uma sessão a partir de dicionário"""
        return cls(
            patient_id=data['patient_id'],
            session_type=SessionType(data['session_type']),
            filename=data['filename'],
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            status=SessionStatus(data.get('status', 'created')),
            notes=data.get('notes'),
            session_id=data.get('id'),
            duration_seconds=data.get('duration_seconds'),
            marker_count=data.get('marker_count', {'T1': 0, 'T2': 0, 'T0': 0, 'BASELINE': 0})
        )
    
    def __str__(self) -> str:
        """Representação string da sessão"""
        return f"Session(id={self.session_id}, patient_id={self.patient_id}, type={self.session_type.value})"
    
    def __repr__(self) -> str:
        """Representação detalhada da sessão"""
        return (f"Session(session_id={self.session_id}, patient_id={self.patient_id}, "
                f"type={self.session_type.value}, status={self.status.value})")
