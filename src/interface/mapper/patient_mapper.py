"""
Mappers para conversão entre DTOs e entidades de domínio
"""
from src.domain.model.patient import Patient, PatientId
from src.interface.dto.patient_dto import PatientDTO, CreatePatientDTO
from typing import Optional
from datetime import datetime


class PatientMapper:
    """Mapper para conversão entre Patient e DTOs"""
    
    @staticmethod
    def to_domain(dto: CreatePatientDTO) -> Patient:
        """
        Converte CreatePatientDTO para entidade Patient
        
        Args:
            dto: DTO de criação de paciente
            
        Returns:
            Entidade Patient sem ID (para novos pacientes)
        """
        return Patient(
            id=None,
            name=dto.name,
            age=dto.age,
            gender=dto.gender,
            time_since_brain_event=dto.time_since_brain_event,
            brain_event_type=dto.brain_event_type,
            affected_side=dto.affected_side,
            notes=dto.notes,
            created_at=None
        )
    
    @staticmethod
    def to_dto(patient: Patient) -> PatientDTO:
        """
        Converte entidade Patient para DTO
        
        Args:
            patient: Entidade Patient
            
        Returns:
            DTO do paciente
        """
        return PatientDTO(
            id=patient.id.value if patient.id else None,
            name=patient.name,
            age=patient.age,
            gender=patient.gender,
            time_since_brain_event=patient.time_since_brain_event,
            brain_event_type=patient.brain_event_type,
            affected_side=patient.affected_side,
            notes=patient.notes,
            created_at=patient.created_at
        )
    
    @staticmethod
    def to_domain_with_id(dto: PatientDTO) -> Patient:
        """
        Converte PatientDTO para entidade Patient (com ID)
        
        Args:
            dto: DTO do paciente
            
        Returns:
            Entidade Patient com ID
        """
        patient_id = PatientId(dto.id) if dto.id else None
        
        return Patient(
            id=patient_id,
            name=dto.name,
            age=dto.age,
            gender=dto.gender,
            time_since_brain_event=dto.time_since_brain_event,
            brain_event_type=dto.brain_event_type,
            affected_side=dto.affected_side,
            notes=dto.notes,
            created_at=dto.created_at
        )
