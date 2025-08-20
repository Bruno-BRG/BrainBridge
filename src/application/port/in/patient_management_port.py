"""
Port para gerenciamento de pacientes - Interface do caso de uso
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.interface.dto.patient_dto import PatientDTO, CreatePatientDTO, UpdatePatientNotesDTO


class PatientManagementInPort(ABC):
    """
    Inbound Port para gerenciamento de pacientes
    
    Define o contrato da aplicação para operações com pacientes.
    """
    
    @abstractmethod
    def register_patient(self, patient_data: CreatePatientDTO) -> PatientDTO:
        """
        Registra um novo paciente no sistema
        
        Args:
            patient_data: Dados do paciente a ser registrado
            
        Returns:
            DTO do paciente registrado com ID atribuído
            
        Raises:
            InvalidPatientDataError: Se dados do paciente são inválidos
            PatientRegistrationError: Em caso de erro no registro
        """
        pass
    
    @abstractmethod
    def find_patient_by_id(self, patient_id: int) -> Optional[PatientDTO]:
        """
        Busca paciente por ID
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            DTO do paciente ou None se não encontrado
            
        Raises:
            PatientSearchError: Em caso de erro na busca
        """
        pass
    
    @abstractmethod
    def list_all_patients(self) -> List[PatientDTO]:
        """
        Lista todos os pacientes cadastrados
        
        Returns:
            Lista de DTOs dos pacientes
            
        Raises:
            PatientSearchError: Em caso de erro na consulta
        """
        pass
    
    @abstractmethod
    def update_patient_notes(self, update_data: UpdatePatientNotesDTO) -> PatientDTO:
        """
        Atualiza as observações de um paciente
        
        Args:
            update_data: Dados para atualização
            
        Returns:
            DTO do paciente atualizado
            
        Raises:
            PatientNotFoundError: Se paciente não for encontrado
            PatientUpdateError: Em caso de erro na atualização
        """
        pass
    
    @abstractmethod
    def remove_patient(self, patient_id: int) -> bool:
        """
        Remove um paciente do sistema
        
        Args:
            patient_id: ID do paciente a ser removido
            
        Returns:
            True se removido com sucesso, False se não encontrado
            
        Raises:
            PatientRemovalError: Em caso de erro na remoção
        """
        pass


class InvalidPatientDataError(Exception):
    """Erro quando dados do paciente são inválidos"""
    pass


class PatientRegistrationError(Exception):
    """Erro durante registro de paciente"""
    pass


class PatientSearchError(Exception):
    """Erro durante busca de paciente"""
    pass


class PatientUpdateError(Exception):
    """Erro durante atualização de paciente"""
    pass


class PatientRemovalError(Exception):
    """Erro durante remoção de paciente"""
    pass
