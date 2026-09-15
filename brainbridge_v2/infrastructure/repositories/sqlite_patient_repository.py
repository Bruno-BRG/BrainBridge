"""
SQLite repository adapter for patient operations.
"""

from typing import List

from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.domain.entities.patient import Patient


class SQLitePatientRepository:
    """
    Infrastructure adapter that maps PatientRepository port to DatabaseManager.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db_manager = db_manager

    def add(self, patient: Patient) -> int:
        patient.validate()
        return self._db_manager.add_patient(
            patient.name,
            patient.age,
            patient.sex,
            patient.affected_hand,
            patient.time_since_event,
            patient.notes,
        )

    def list_all(self) -> List[Patient]:
        rows = self._db_manager.get_all_patients()
        patients: List[Patient] = []
        for row in rows:
            patients.append(
                Patient(
                    id=row.get("id"),
                    name=row.get("name", ""),
                    age=int(row.get("age", 0)),
                    sex=row.get("sex", ""),
                    affected_hand=row.get("affected_hand"),
                    time_since_event=int(row.get("time_since_event", 0)),
                    notes=row.get("notes") or "",
                    created_at=row.get("created_at"),
                )
            )
        return patients

    def update_affected_hand(self, patient_id: int, affected_hand: str) -> bool:
        return self._db_manager.update_patient_affected_hand(patient_id, affected_hand)
