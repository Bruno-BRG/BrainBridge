"""
Use Case simplificado para demonstração
"""
from typing import List, Optional
from datetime import datetime


# Exceções personalizadas
class InvalidPatientDataError(Exception):
    pass

class PatientRegistrationError(Exception):
    pass

class PatientSearchError(Exception):
    pass

class PatientUpdateError(Exception):
    pass

class PatientRemovalError(Exception):
    pass


class SimplePatientManagementUseCase:
    """Use Case simplificado para gerenciamento de pacientes"""
    
    def __init__(self, patient_repository):
        self._patient_repository = patient_repository
    
    def register_patient(self, patient_data):
        """Registra um novo paciente no sistema"""
        try:
            # Importar aqui para evitar problemas de import circular
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            sys.path.insert(0, project_root)
            
            from src.interface.mapper.patient_mapper import PatientMapper
            from src.domain.model.patient import Patient
            
            # Converter DTO para entidade de domínio
            patient = PatientMapper.to_domain(patient_data)
            
            # Adicionar timestamp de criação
            patient_with_timestamp = Patient(
                id=patient.id,
                name=patient.name,
                age=patient.age,
                gender=patient.gender,
                time_since_brain_event=patient.time_since_brain_event,
                brain_event_type=patient.brain_event_type,
                affected_side=patient.affected_side,
                notes=patient.notes,
                created_at=datetime.now()
            )
            
            # Salvar no repositório
            saved_patient = self._patient_repository.save(patient_with_timestamp)
            
            # Converter de volta para DTO
            return PatientMapper.to_dto(saved_patient)
            
        except ValueError as e:
            raise InvalidPatientDataError(f"Dados inválidos: {str(e)}") from e
        except Exception as e:
            raise PatientRegistrationError(f"Erro ao registrar paciente: {str(e)}") from e
    
    def find_patient_by_id(self, patient_id: int):
        """Busca paciente por ID"""
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            sys.path.insert(0, project_root)
            
            from src.domain.model.patient import PatientId
            from src.interface.mapper.patient_mapper import PatientMapper
            
            if patient_id <= 0:
                raise InvalidPatientDataError("ID do paciente deve ser positivo")
            
            domain_patient_id = PatientId(patient_id)
            patient = self._patient_repository.find_by_id(domain_patient_id)
            
            if patient:
                return PatientMapper.to_dto(patient)
            return None
            
        except ValueError as e:
            raise InvalidPatientDataError(f"ID inválido: {str(e)}") from e
        except Exception as e:
            raise PatientSearchError(f"Erro ao buscar paciente: {str(e)}") from e
    
    def list_all_patients(self):
        """Lista todos os pacientes cadastrados"""
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            sys.path.insert(0, project_root)
            
            from src.interface.mapper.patient_mapper import PatientMapper
            
            patients = self._patient_repository.find_all()
            return [PatientMapper.to_dto(patient) for patient in patients]
            
        except Exception as e:
            raise PatientSearchError(f"Erro ao listar pacientes: {str(e)}") from e
    
    def update_patient_notes(self, update_data):
        """Atualiza as observações de um paciente"""
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            sys.path.insert(0, project_root)
            
            from src.domain.model.patient import PatientId
            from src.interface.mapper.patient_mapper import PatientMapper
            
            if update_data.patient_id <= 0:
                raise InvalidPatientDataError("ID do paciente deve ser positivo")
            
            domain_patient_id = PatientId(update_data.patient_id)
            
            # Buscar paciente existente
            existing_patient = self._patient_repository.find_by_id(domain_patient_id)
            if not existing_patient:
                raise PatientUpdateError(f"Paciente com ID {update_data.patient_id} não encontrado")
            
            # Atualizar observações usando método do domínio
            updated_patient = existing_patient.update_notes(update_data.notes)
            
            # Salvar no repositório
            saved_patient = self._patient_repository.save(updated_patient)
            
            # Converter para DTO
            return PatientMapper.to_dto(saved_patient)
            
        except ValueError as e:
            raise InvalidPatientDataError(f"Dados inválidos: {str(e)}") from e
        except Exception as e:
            raise PatientUpdateError(f"Erro ao atualizar paciente: {str(e)}") from e
    
    def remove_patient(self, patient_id: int):
        """Remove um paciente do sistema"""
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            sys.path.insert(0, project_root)
            
            from src.domain.model.patient import PatientId
            
            if patient_id <= 0:
                raise InvalidPatientDataError("ID do paciente deve ser positivo")
            
            domain_patient_id = PatientId(patient_id)
            return self._patient_repository.delete(domain_patient_id)
            
        except ValueError as e:
            raise InvalidPatientDataError(f"ID inválido: {str(e)}") from e
        except Exception as e:
            raise PatientRemovalError(f"Erro ao remover paciente: {str(e)}") from e
