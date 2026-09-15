from typing import List

from brainbridge_v2.domain.entities.patient import Patient
from brainbridge_v2.interface_adapters.controllers.patient_controller import PatientController


class FakePatientRepository:
    def __init__(self):
        self._patients: List[Patient] = []
        self._next_id = 1

    def add(self, patient: Patient) -> int:
        new_id = self._next_id
        self._next_id += 1
        stored = Patient(
            id=new_id,
            name=patient.name,
            age=patient.age,
            sex=patient.sex,
            affected_hand=patient.affected_hand,
            time_since_event=patient.time_since_event,
            notes=patient.notes,
            created_at="2026-03-02T00:00:00",
        )
        self._patients.append(stored)
        return new_id

    def list_all(self) -> List[Patient]:
        return list(self._patients)


def test_patient_controller_register_and_list():
    controller = PatientController.from_repository(FakePatientRepository())

    patient_id = controller.register_patient(
        {
            "name": "Murilo",
            "age": 29,
            "sex": "Masculino",
            "affected_hand": "left",
            "time_since_event": 4,
            "notes": "Paciente ativo",
        }
    )
    patients = controller.list_patients()

    assert patient_id == 1
    assert len(patients) == 1
    assert patients[0]["name"] == "Murilo"
    assert patients[0]["id"] == 1
    assert patients[0]["created_at"] == "2026-03-02T00:00:00"
