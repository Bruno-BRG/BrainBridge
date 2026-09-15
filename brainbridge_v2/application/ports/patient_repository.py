"""
Port definition for patient persistence.
"""

from typing import List, Protocol
from brainbridge_v2.domain.entities.patient import Patient


class PatientRepository(Protocol):
    def add(self, patient: Patient) -> int:
        """
        Persists a patient and returns the generated identifier.
        """
        ...

    def list_all(self) -> List[Patient]:
        """
        Returns all patients.
        """
        ...

    def update_affected_hand(self, patient_id: int, affected_hand: str) -> bool:
        """Updates only the affected hand; returns False if the patient is absent."""
        ...
