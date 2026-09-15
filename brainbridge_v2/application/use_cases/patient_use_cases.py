"""
Patient-related use cases.
"""

from typing import List

from brainbridge_v2.application.ports.patient_repository import PatientRepository
from brainbridge_v2.domain.entities.patient import Patient


class RegisterPatientUseCase:
    """
    Registers a new patient after domain validation.
    """

    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def execute(self, patient: Patient) -> int:
        patient.validate()
        return self._repository.add(patient)


class ListPatientsUseCase:
    """
    Returns all registered patients.
    """

    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def execute(self) -> List[Patient]:
        return self._repository.list_all()


class UpdatePatientAffectedHandUseCase:
    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def execute(self, patient_id: int, affected_hand: str) -> None:
        Patient.validate_affected_hand(affected_hand)
        if affected_hand is None:
            raise ValueError("Selecione a mao afetada.")
        if not self._repository.update_affected_hand(patient_id, affected_hand):
            raise ValueError("Paciente nao encontrado.")
