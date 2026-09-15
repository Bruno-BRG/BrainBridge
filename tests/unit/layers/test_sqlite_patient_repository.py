from pathlib import Path
import shutil
import uuid

from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.domain.entities.patient import Patient
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import SQLitePatientRepository


def test_sqlite_patient_repository_add_and_list():
    temp_dir = Path("brainbridge_v2") / "data" / f"tmp_test_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        db_path = temp_dir / "patients_test.db"
        manager = DatabaseManager(db_path=db_path)
        repository = SQLitePatientRepository(manager)

        patient_id = repository.add(
            Patient(
                name="Joana",
                age=52,
                sex="Feminino",
                affected_hand="right",
                time_since_event=10,
                notes="Caso de teste",
            )
        )
        patients = repository.list_all()

        assert patient_id > 0
        assert len(patients) == 1
        assert patients[0].id == patient_id
        assert patients[0].name == "Joana"
        assert patients[0].affected_hand == "right"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
