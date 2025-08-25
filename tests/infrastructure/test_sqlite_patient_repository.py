"""
Testes de integração para SqlitePatientRepository
"""
import sys
import os
import tempfile
import pytest
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.infrastructure.adapter.out.sqlite_patient_repository import SqlitePatientRepository
from src.application.port.out.patient_repository_port import RepositoryError, PatientNotFoundError
from src.domain.model.patient import Patient, PatientId


class TestSqlitePatientRepository:
    """Testes de integração para o repositório SQLite"""
    
    def setup_method(self):
        """Setup para cada teste"""
        # Usar banco em memória para testes
        self.repository = SqlitePatientRepository(":memory:")
    
    def test_save_new_patient(self):
        """Deve salvar novo paciente e atribuir ID"""
        # Arrange
        patient = Patient(
            id=None,
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes="Paciente colaborativo",
            created_at=datetime(2025, 8, 20, 10, 0, 0)
        )
        
        # Act
        saved_patient = self.repository.save(patient)
        
        # Assert
        assert saved_patient.id is not None
        assert saved_patient.id.value > 0
        assert saved_patient.name == "João Silva"
        assert saved_patient.age == 45
        assert saved_patient.created_at == datetime(2025, 8, 20, 10, 0, 0)
    
    def test_find_by_id_existing(self):
        """Deve encontrar paciente existente por ID"""
        # Arrange
        patient = Patient(
            id=None,
            name="Maria Santos",
            age=30,
            gender="F",
            time_since_brain_event=6,
            brain_event_type="TCE",
            affected_side="Direito",
            notes="",
            created_at=datetime.now()
        )
        
        saved_patient = self.repository.save(patient)
        
        # Act
        found_patient = self.repository.find_by_id(saved_patient.id)
        
        # Assert
        assert found_patient is not None
        assert found_patient.id == saved_patient.id
        assert found_patient.name == "Maria Santos"
        assert found_patient.age == 30
    
    def test_find_by_id_non_existing(self):
        """Deve retornar None para paciente inexistente"""
        # Act
        found_patient = self.repository.find_by_id(PatientId(999))
        
        # Assert
        assert found_patient is None
    
    def test_find_all_empty(self):
        """Deve retornar lista vazia quando não há pacientes"""
        # Act
        patients = self.repository.find_all()
        
        # Assert
        assert patients == []
    
    def test_find_all_with_patients(self):
        """Deve retornar todos os pacientes salvos"""
        # Arrange
        patient1 = Patient(None, "João", 45, "M", 12, "AVC", "Esquerdo", "")
        patient2 = Patient(None, "Maria", 30, "F", 6, "TCE", "Direito", "")
        
        self.repository.save(patient1)
        self.repository.save(patient2)
        
        # Act
        patients = self.repository.find_all()
        
        # Assert
        assert len(patients) == 2
        # Verificar se os nomes estão presentes (ordem pode variar)
        names = [p.name for p in patients]
        assert "João" in names
        assert "Maria" in names
    
    def test_delete_existing(self):
        """Deve deletar paciente existente"""
        # Arrange
        patient = Patient(None, "João", 45, "M", 12, "AVC", "Esquerdo", "")
        saved_patient = self.repository.save(patient)
        
        # Act
        deleted = self.repository.delete(saved_patient.id)
        
        # Assert
        assert deleted is True
        
        # Verificar que foi removido
        found_patient = self.repository.find_by_id(saved_patient.id)
        assert found_patient is None
    
    def test_delete_non_existing(self):
        """Deve retornar False para paciente inexistente"""
        # Act
        deleted = self.repository.delete(PatientId(999))
        
        # Assert
        assert deleted is False
    
    def test_update_existing_patient(self):
        """Deve atualizar paciente existente"""
        # Arrange
        patient = Patient(None, "João", 45, "M", 12, "AVC", "Esquerdo", "Notas originais")
        saved_patient = self.repository.save(patient)
        
        # Criar versão atualizada
        updated_patient = Patient(
            id=saved_patient.id,
            name=saved_patient.name,
            age=saved_patient.age,
            gender=saved_patient.gender,
            time_since_brain_event=saved_patient.time_since_brain_event,
            brain_event_type=saved_patient.brain_event_type,
            affected_side=saved_patient.affected_side,
            notes="Notas atualizadas",
            created_at=saved_patient.created_at
        )
        
        # Act
        result = self.repository.save(updated_patient)
        
        # Assert
        assert result.notes == "Notas atualizadas"
        
        # Verificar persistência
        found_patient = self.repository.find_by_id(saved_patient.id)
        assert found_patient.notes == "Notas atualizadas"
