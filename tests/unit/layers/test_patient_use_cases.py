from typing import List

import pytest

from brainbridge_v2.application.use_cases.patient_use_cases import (
    ListPatientsUseCase,
    RegisterPatientUseCase,
)
from brainbridge_v2.domain.entities.patient import Patient


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
        )
        self._patients.append(stored)
        return new_id

    def list_all(self) -> List[Patient]:
        return list(self._patients)


def test_register_patient_use_case_valid_input():
    repo = FakePatientRepository()
    use_case = RegisterPatientUseCase(repo)

    patient_id = use_case.execute(
        Patient(
            name="Ana",
            age=34,
            sex="Feminino",
            affected_hand="left",
            time_since_event=12,
            notes="Teste",
        )
    )

    assert patient_id == 1
    assert len(repo.list_all()) == 1


def test_register_patient_use_case_invalid_name():
    repo = FakePatientRepository()
    use_case = RegisterPatientUseCase(repo)

    with pytest.raises(ValueError):
        use_case.execute(
            Patient(
                name="  ",
                age=30,
                sex="Masculino",
                affected_hand="right",
                time_since_event=6,
            )
        )


def test_list_patients_use_case_returns_entities():
    repo = FakePatientRepository()
    repo.add(
        Patient(
            name="Carlos",
            age=40,
            sex="Masculino",
            affected_hand="right",
            time_since_event=24,
        )
    )
    use_case = ListPatientsUseCase(repo)

    patients = use_case.execute()

    assert len(patients) == 1
    assert patients[0].name == "Carlos"
