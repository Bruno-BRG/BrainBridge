"""
Testes para PatientManagementUseCase
"""
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
from src.application.usecase.patient_management_usecase import PatientManagementUseCase
from src.application.port.out.patient_repository_port import PatientRepositoryPort, RepositoryError
from src.application.port.in.patient_management_port import (
    PatientManagementInPort,
    InvalidPatientDataError,
    PatientRegistrationError,
    PatientSearchError
)
from src.interface.dto.patient_dto import CreatePatientDTO, UpdatePatientNotesDTO
from src.domain.model.patient import Patient, PatientId


class TestPatientManagementUseCase:
    """Testes para o use case de gerenciamento de pacientes"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.mock_repository = Mock(spec=PatientRepositoryPort)
        self.use_case = PatientManagementUseCase(self.mock_repository)
    
    def test_register_patient_success(self):
        """Deve registrar paciente com sucesso"""
        # Arrange
        patient_data = CreatePatientDTO(
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes="Paciente colaborativo"
        )
        
        # Mock do paciente salvo pelo repositório
        saved_patient = Patient(
            id=PatientId(1),
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes="Paciente colaborativo",
            created_at=datetime(2025, 8, 20, 10, 0, 0)
        )
        
        self.mock_repository.save.return_value = saved_patient
        
        # Act
        with patch('src.application.usecase.patient_management_usecase.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 8, 20, 10, 0, 0)
            result = self.use_case.register_patient(patient_data)
        
        # Assert
        assert result.id == 1
        assert result.name == "João Silva"
        assert result.age == 45
        assert result.created_at == datetime(2025, 8, 20, 10, 0, 0)
        
        # Verificar chamada do repositório
        self.mock_repository.save.assert_called_once()
        saved_patient_arg = self.mock_repository.save.call_args[0][0]
        assert saved_patient_arg.name == "João Silva"
        assert saved_patient_arg.created_at == datetime(2025, 8, 20, 10, 0, 0)
    
    def test_register_patient_invalid_data(self):
        """Deve rejeitar dados inválidos do paciente"""
        # Arrange
        invalid_patient_data = CreatePatientDTO(
            name="",  # Nome vazio - deve falhar
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes=""
        )
        
        # Act & Assert
        with pytest.raises(InvalidPatientDataError, match="Dados inválidos"):
            self.use_case.register_patient(invalid_patient_data)
        
        # Repositório não deve ser chamado
        self.mock_repository.save.assert_not_called()
    
    def test_register_patient_repository_error(self):
        """Deve tratar erro do repositório"""
        # Arrange
        patient_data = CreatePatientDTO(
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes=""
        )
        
        self.mock_repository.save.side_effect = RepositoryError("Erro de conexão")
        
        # Act & Assert
        with pytest.raises(PatientRegistrationError, match="Erro ao registrar paciente"):
            self.use_case.register_patient(patient_data)
    
    def test_find_patient_by_id_success(self):
        """Deve encontrar paciente por ID"""
        # Arrange
        patient = Patient(
            id=PatientId(1),
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes="",
            created_at=datetime(2025, 8, 20, 10, 0, 0)
        )
        
        self.mock_repository.find_by_id.return_value = patient
        
        # Act
        result = self.use_case.find_patient_by_id(1)
        
        # Assert
        assert result is not None
        assert result.id == 1
        assert result.name == "João Silva"
        
        # Verificar chamada do repositório
        self.mock_repository.find_by_id.assert_called_once_with(PatientId(1))
    
    def test_find_patient_by_id_not_found(self):
        """Deve retornar None quando paciente não encontrado"""
        # Arrange
        self.mock_repository.find_by_id.return_value = None
        
        # Act
        result = self.use_case.find_patient_by_id(1)
        
        # Assert
        assert result is None
    
    def test_find_patient_by_id_invalid_id(self):
        """Deve rejeitar ID inválido"""
        # Act & Assert
        with pytest.raises(InvalidPatientDataError, match="ID do paciente deve ser positivo"):
            self.use_case.find_patient_by_id(0)
        
        with pytest.raises(InvalidPatientDataError, match="ID do paciente deve ser positivo"):
            self.use_case.find_patient_by_id(-1)
    
    def test_list_all_patients_success(self):
        """Deve listar todos os pacientes"""
        # Arrange
        patients = [
            Patient(PatientId(1), "João", 45, "M", 12, "AVC", "Esquerdo", ""),
            Patient(PatientId(2), "Maria", 30, "F", 6, "TCE", "Direito", "")
        ]
        
        self.mock_repository.find_all.return_value = patients
        
        # Act
        result = self.use_case.list_all_patients()
        
        # Assert
        assert len(result) == 2
        assert result[0].id == 1
        assert result[0].name == "João"
        assert result[1].id == 2
        assert result[1].name == "Maria"
    
    def test_list_all_patients_empty(self):
        """Deve retornar lista vazia quando não há pacientes"""
        # Arrange
        self.mock_repository.find_all.return_value = []
        
        # Act
        result = self.use_case.list_all_patients()
        
        # Assert
        assert result == []
