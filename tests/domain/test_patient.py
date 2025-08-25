"""
Testes para a entidade Patient e PatientId
"""
import pytest
from datetime import datetime
from src.domain.model.patient import Patient, PatientId


class TestPatientId:
    """Testes para o Value Object PatientId"""
    
    def test_create_valid_patient_id(self):
        """Deve criar PatientId válido"""
        patient_id = PatientId(1)
        assert patient_id.value == 1
    
    def test_invalid_patient_id_zero(self):
        """Deve rejeitar PatientId zero"""
        with pytest.raises(ValueError, match="PatientId deve ser um número positivo"):
            PatientId(0)
    
    def test_invalid_patient_id_negative(self):
        """Deve rejeitar PatientId negativo"""
        with pytest.raises(ValueError, match="PatientId deve ser um número positivo"):
            PatientId(-1)


class TestPatient:
    """Testes para a entidade Patient"""
    
    def test_create_valid_patient(self):
        """Deve criar paciente válido"""
        patient = Patient(
            id=PatientId(1),
            name="João Silva",
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes="Paciente colaborativo",
            created_at=datetime.now()
        )
        
        assert patient.name == "João Silva"
        assert patient.age == 45
        assert patient.id.value == 1
    
    def test_create_patient_without_id(self):
        """Deve criar paciente sem ID (para novos pacientes)"""
        patient = Patient(
            id=None,
            name="Maria Santos",
            age=30,
            gender="F",
            time_since_brain_event=6,
            brain_event_type="TCE",
            affected_side="Direito",
            notes=""
        )
        
        assert patient.id is None
        assert patient.name == "Maria Santos"
    
    def test_invalid_empty_name(self):
        """Deve rejeitar nome vazio"""
        with pytest.raises(ValueError, match="Nome do paciente não pode ser vazio"):
            Patient(
                id=None,
                name="",
                age=30,
                gender="F",
                time_since_brain_event=6,
                brain_event_type="TCE",
                affected_side="Direito",
                notes=""
            )
    
    def test_invalid_whitespace_name(self):
        """Deve rejeitar nome apenas com espaços"""
        with pytest.raises(ValueError, match="Nome do paciente não pode ser vazio"):
            Patient(
                id=None,
                name="   ",
                age=30,
                gender="F",
                time_since_brain_event=6,
                brain_event_type="TCE",
                affected_side="Direito",
                notes=""
            )
    
    def test_invalid_age_negative(self):
        """Deve rejeitar idade negativa"""
        with pytest.raises(ValueError, match="Idade deve estar entre 0 e 150 anos"):
            Patient(
                id=None,
                name="João",
                age=-1,
                gender="M",
                time_since_brain_event=6,
                brain_event_type="AVC",
                affected_side="Esquerdo",
                notes=""
            )
    
    def test_invalid_age_too_high(self):
        """Deve rejeitar idade muito alta"""
        with pytest.raises(ValueError, match="Idade deve estar entre 0 e 150 anos"):
            Patient(
                id=None,
                name="João",
                age=200,
                gender="M",
                time_since_brain_event=6,
                brain_event_type="AVC",
                affected_side="Esquerdo",
                notes=""
            )
    
    def test_invalid_negative_time_since_event(self):
        """Deve rejeitar tempo negativo desde evento"""
        with pytest.raises(ValueError, match="Tempo desde evento cerebral deve ser não-negativo"):
            Patient(
                id=None,
                name="João",
                age=45,
                gender="M",
                time_since_brain_event=-1,
                brain_event_type="AVC",
                affected_side="Esquerdo",
                notes=""
            )
    
    def test_get_age_group(self):
        """Deve retornar faixa etária correta"""
        child = Patient(None, "Criança", 10, "F", 1, "TCE", "Direito", "")
        adult = Patient(None, "Adulto", 30, "M", 1, "AVC", "Esquerdo", "")
        elder = Patient(None, "Idoso", 70, "M", 1, "AVC", "Esquerdo", "")
        
        assert child.get_age_group() == "Infantil"
        assert adult.get_age_group() == "Adulto"
        assert elder.get_age_group() == "Idoso"
    
    def test_is_recent_event(self):
        """Deve identificar eventos recentes"""
        recent = Patient(None, "João", 45, "M", 3, "AVC", "Esquerdo", "")
        old = Patient(None, "Maria", 50, "F", 12, "TCE", "Direito", "")
        
        assert recent.is_recent_event() is True
        assert old.is_recent_event() is False
    
    def test_update_notes(self):
        """Deve atualizar observações mantendo imutabilidade"""
        original = Patient(
            PatientId(1), "João", 45, "M", 12, "AVC", "Esquerdo", "Notas originais"
        )
        
        updated = original.update_notes("Novas observações")
        
        # Original não mudou
        assert original.notes == "Notas originais"
        # Novo objeto com notas atualizadas
        assert updated.notes == "Novas observações"
        assert updated.id == original.id
        assert updated.name == original.name
