"""
Interfaces de Dados - Seguindo ISP (Interface Segregation Principle)

Estas interfaces definem contratos específicos para operações de dados,
evitando interfaces grandes e monolíticas.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime


class IPatientRepository(ABC):
    """Interface para operações de repositório de pacientes"""
    
    @abstractmethod
    def create_patient(self, patient_data: Dict[str, Any]) -> int:
        """Cria um novo paciente e retorna o ID"""
        pass
    
    @abstractmethod
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Busca paciente por ID"""
        pass
    
    @abstractmethod
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Retorna todos os pacientes"""
        pass
    
    @abstractmethod
    def update_patient(self, patient_id: int, patient_data: Dict[str, Any]) -> bool:
        """Atualiza dados do paciente"""
        pass
    
    @abstractmethod
    def delete_patient(self, patient_id: int) -> bool:
        """Remove paciente do sistema"""
        pass


class ISessionRepository(ABC):
    """Interface para operações de repositório de sessões"""
    
    @abstractmethod
    def create_session(self, session_data: Dict[str, Any]) -> int:
        """Cria uma nova sessão e retorna o ID"""
        pass
    
    @abstractmethod
    def get_session_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Busca sessão por ID"""
        pass
    
    @abstractmethod
    def get_sessions_by_patient(self, patient_id: int) -> List[Dict[str, Any]]:
        """Retorna todas as sessões de um paciente"""
        pass
    
    @abstractmethod
    def update_session(self, session_id: int, session_data: Dict[str, Any]) -> bool:
        """Atualiza dados da sessão"""
        pass


class IDataLogger(ABC):
    """Interface para logging de dados"""
    
    @abstractmethod
    def start_logging(self, session_id: int, filename: str) -> bool:
        """Inicia o logging de dados"""
        pass
    
    @abstractmethod
    def log_data(self, data: Dict[str, Any]) -> bool:
        """Registra dados no log"""
        pass
    
    @abstractmethod
    def add_marker(self, marker_type: str, timestamp: datetime) -> bool:
        """Adiciona marcador aos dados"""
        pass
    
    @abstractmethod
    def stop_logging(self) -> bool:
        """Para o logging e finaliza o arquivo"""
        pass
    
    @abstractmethod
    def is_logging(self) -> bool:
        """Verifica se está fazendo logging"""
        pass
