"""
Port para persistência de pacientes - Interface que define operações de repositório
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.model.patient import Patient, PatientId


class PatientRepositoryPort(ABC):
    """
    Outbound Port para persistência de pacientes
    
    Define o contrato que adapters de persistência devem implementar.
    """
    
    @abstractmethod
    def save(self, patient: Patient) -> Patient:
        """
        Salva um paciente no repositório
        
        Args:
            patient: Paciente a ser salvo
            
        Returns:
            Paciente salvo com ID atribuído (se novo)
            
        Raises:
            PatientAlreadyExistsError: Se paciente com mesmo ID já existe
            RepositoryError: Em caso de erro de persistência
        """
        pass
    
    @abstractmethod
    def find_by_id(self, patient_id: PatientId) -> Optional[Patient]:
        """
        Busca paciente por ID
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            Paciente encontrado ou None
            
        Raises:
            RepositoryError: Em caso de erro na consulta
        """
        pass
    
    @abstractmethod
    def find_all(self) -> List[Patient]:
        """
        Retorna todos os pacientes
        
        Returns:
            Lista de pacientes
            
        Raises:
            RepositoryError: Em caso de erro na consulta
        """
        pass
    
    @abstractmethod
    def delete(self, patient_id: PatientId) -> bool:
        """
        Remove paciente do repositório
        
        Args:
            patient_id: ID do paciente a ser removido
            
        Returns:
            True se removido com sucesso, False se não encontrado
            
        Raises:
            RepositoryError: Em caso de erro na operação
        """
        pass


class PatientAlreadyExistsError(Exception):
    """Erro quando tentativa de criar paciente com ID já existente"""
    pass


class PatientNotFoundError(Exception):
    """Erro quando paciente não é encontrado"""
    pass


class RepositoryError(Exception):
    """Erro genérico de repositório"""
    pass
