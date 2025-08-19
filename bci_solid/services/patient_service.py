"""
Serviço de Pacientes - Seguindo DIP e SRP

Este serviço depende de abstrações (interfaces) e tem uma única responsabilidade:
gerenciar operações de negócio relacionadas a pacientes.
"""

from typing import List, Dict, Optional, Any
from ..interfaces.data_interfaces import IPatientRepository
from ..interfaces.service_interfaces import IValidationService
from ..models.patient import Patient, Gender, AffectedHand


class PatientService:
    """
    Serviço de pacientes seguindo DIP e SRP
    
    Responsabilidade única: Lógica de negócio para pacientes
    Depende de abstrações (IPatientRepository, IValidationService)
    """
    
    def __init__(self, 
                 patient_repository: IPatientRepository,
                 validation_service: Optional[IValidationService] = None):
        """
        Inicializa o serviço com dependências injetadas
        
        Args:
            patient_repository: Repositório de dados de pacientes
            validation_service: Serviço de validação (opcional)
        """
        self._patient_repository = patient_repository
        self._validation_service = validation_service
    
    def create_patient(self, patient_data: Dict[str, Any]) -> tuple[bool, str, Optional[int]]:
        """
        Cria um novo paciente
        
        Args:
            patient_data: Dados do paciente
            
        Returns:
            Tupla (sucesso, mensagem, patient_id)
        """
        try:
            # Validar dados se serviço de validação estiver disponível
            if self._validation_service:
                validation_result = self._validation_service.validate_patient_data(patient_data)
                if validation_result:
                    errors = []
                    for field, field_errors in validation_result.items():
                        errors.extend(field_errors)
                    return False, "; ".join(errors), None
            
            # Criar modelo de paciente (validação automática)
            patient = self._create_patient_model(patient_data)
            
            # Verificar se já existe paciente com mesmo nome
            existing_patients = self._patient_repository.get_all_patients()
            if any(p['name'].lower() == patient.name.lower() for p in existing_patients):
                return False, "Já existe um paciente com este nome", None
            
            # Salvar no repositório
            patient_id = self._patient_repository.create_patient(patient.to_dict())
            
            return True, "Paciente criado com sucesso", patient_id
            
        except ValueError as e:
            return False, f"Dados inválidos: {str(e)}", None
        except Exception as e:
            return False, f"Erro interno: {str(e)}", None
    
    def get_patient(self, patient_id: int) -> Optional[Patient]:
        """
        Busca paciente por ID
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            Paciente ou None se não encontrado
        """
        try:
            patient_data = self._patient_repository.get_patient_by_id(patient_id)
            if patient_data:
                return Patient.from_dict(patient_data)
            return None
        except Exception:
            return None
    
    def get_all_patients(self) -> List[Patient]:
        """
        Retorna todos os pacientes
        
        Returns:
            Lista de pacientes
        """
        try:
            patients_data = self._patient_repository.get_all_patients()
            return [Patient.from_dict(data) for data in patients_data]
        except Exception:
            return []
    
    def update_patient(self, patient_id: int, updates: Dict[str, Any]) -> tuple[bool, str]:
        """
        Atualiza dados do paciente
        
        Args:
            patient_id: ID do paciente
            updates: Dados para atualizar
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Buscar paciente existente
            current_patient = self.get_patient(patient_id)
            if not current_patient:
                return False, "Paciente não encontrado"
            
            # Validar atualizações se serviço de validação estiver disponível
            if self._validation_service:
                validation_result = self._validation_service.validate_patient_data(updates)
                if validation_result:
                    errors = []
                    for field, field_errors in validation_result.items():
                        errors.extend(field_errors)
                    return False, "; ".join(errors)
            
            # Aplicar atualizações
            current_patient.update_from_dict(updates)
            
            # Salvar no repositório
            success = self._patient_repository.update_patient(patient_id, current_patient.to_dict())
            
            if success:
                return True, "Paciente atualizado com sucesso"
            else:
                return False, "Erro ao atualizar paciente"
                
        except ValueError as e:
            return False, f"Dados inválidos: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def delete_patient(self, patient_id: int) -> tuple[bool, str]:
        """
        Remove paciente do sistema
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Verificar se paciente existe
            patient = self.get_patient(patient_id)
            if not patient:
                return False, "Paciente não encontrado"
            
            # Remover do repositório
            success = self._patient_repository.delete_patient(patient_id)
            
            if success:
                return True, "Paciente removido com sucesso"
            else:
                return False, "Erro ao remover paciente"
                
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def search_patients(self, query: str) -> List[Patient]:
        """
        Busca pacientes por nome
        
        Args:
            query: Termo de busca
            
        Returns:
            Lista de pacientes que correspondem à busca
        """
        all_patients = self.get_all_patients()
        query_lower = query.lower()
        
        return [
            patient for patient in all_patients
            if query_lower in patient.name.lower()
        ]
    
    def get_patients_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo estatístico dos pacientes
        
        Returns:
            Dicionário com estatísticas
        """
        patients = self.get_all_patients()
        
        if not patients:
            return {
                'total': 0,
                'by_gender': {},
                'by_affected_hand': {},
                'age_stats': {}
            }
        
        # Calcular estatísticas
        ages = [p.age for p in patients]
        
        return {
            'total': len(patients),
            'by_gender': self._count_by_attribute(patients, 'gender'),
            'by_affected_hand': self._count_by_attribute(patients, 'affected_hand'),
            'age_stats': {
                'min': min(ages),
                'max': max(ages),
                'avg': sum(ages) / len(ages)
            }
        }
    
    def _create_patient_model(self, patient_data: Dict[str, Any]) -> Patient:
        """
        Cria modelo de paciente a partir dos dados
        
        Args:
            patient_data: Dados do paciente
            
        Returns:
            Instância de Patient
        """
        # Converter strings para enums
        gender = Gender(patient_data['gender']) if isinstance(patient_data['gender'], str) else patient_data['gender']
        affected_hand = AffectedHand(patient_data['affected_hand']) if isinstance(patient_data['affected_hand'], str) else patient_data['affected_hand']
        
        return Patient(
            name=patient_data['name'],
            age=patient_data['age'],
            gender=gender,
            affected_hand=affected_hand,
            time_since_event=patient_data['time_since_event'],
            notes=patient_data.get('notes')
        )
    
    def _count_by_attribute(self, patients: List[Patient], attribute: str) -> Dict[str, int]:
        """
        Conta pacientes por atributo
        
        Args:
            patients: Lista de pacientes
            attribute: Nome do atributo
            
        Returns:
            Dicionário com contagens
        """
        counts = {}
        for patient in patients:
            value = getattr(patient, attribute)
            if hasattr(value, 'value'):  # Enum
                value = value.value
            counts[value] = counts.get(value, 0) + 1
        return counts
