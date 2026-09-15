import sqlite3

import pytest

from brainbridge_v2.domain.entities.patient import Patient
from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import SQLitePatientRepository
from brainbridge_v2.interface_adapters.controllers.patient_controller import PatientController


@pytest.mark.parametrize('hand', ['left', 'right', None])
def test_domain_accepts_canonical_and_legacy_unknown(hand):
    Patient('Ana', 30, 'Feminino', hand, 3).validate()


@pytest.mark.parametrize('hand', ['', 'Esquerda', 'Direita', 'both', 'none', 'LEFT', 1, [], {}])
def test_domain_rejects_invalid_hand(hand):
    with pytest.raises(ValueError):
        Patient('Ana', 30, 'Feminino', hand, 3).validate()


@pytest.mark.parametrize('old_column', [True, False])
def test_non_destructive_idempotent_migration_and_crud(tmp_path, old_column):
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        hand_sql = 'affected_hand TEXT NOT NULL,' if old_column else ''
        conn.execute(f'''CREATE TABLE patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            age INTEGER NOT NULL, sex TEXT NOT NULL, {hand_sql}
            time_since_event INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, notes TEXT)''')
        for name in ['Esquerda', 'Direita', 'Ambas', 'Nenhuma', '']:
            fields = 'name, age, sex, time_since_event, notes'
            values = [name or 'Unknown', 42, 'Outro', 9, 'keep notes']
            if old_column:
                fields += ', affected_hand'
                values.append(name)
            conn.execute(f"INSERT INTO patients ({fields}) VALUES ({','.join('?' for _ in values)})", values)
        conn.execute('''CREATE TABLE recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL,
            filename TEXT NOT NULL, task_type TEXT NOT NULL, start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP, duration INTEGER, notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(id))''')
        conn.execute("INSERT INTO recordings VALUES (12, 1, 'keep.csv', 'task', '2026-01-01', NULL, NULL, 'recording notes')")
        conn.execute('CREATE INDEX patient_name_idx ON patients(name)')
    manager = DatabaseManager(path)
    recording_before = manager.get_patient_recordings(1)
    before = manager.get_all_patients()
    expected = ['left', 'right', None, None, None] if old_column else [None] * 5
    assert [p['affected_hand'] for p in sorted(before, key=lambda p: p['id'])] == expected
    manager = DatabaseManager(path)
    assert manager.get_all_patients() == before
    controller = PatientController.from_repository(SQLitePatientRepository(manager))
    controller.update_patient_affected_hand(1, 'right')
    after = next(p for p in controller.list_patients() if p['id'] == 1)
    original = next(p for p in before if p['id'] == 1)
    assert after == dict(original, affected_hand='right')
    assert manager.get_patient_recordings(1) == recording_before
    assert recording_before[0]['id'] == 12
    assert DatabaseManager(path).get_all_patients() == controller.list_patients()
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='patient_name_idx'").fetchone()
    if old_column:
        with sqlite3.connect(path) as conn:
            assert conn.execute('SELECT affected_hand_legacy FROM patients WHERE id=1').fetchone()[0] == 'Esquerda'
    new_id = controller.register_patient(dict(name='New', age=20, sex='Outro', affected_hand='left'))
    assert new_id > 5
    assert manager.delete_patient(new_id)
    assert len(controller.list_patients()) == 5


@pytest.mark.parametrize('hand', ['left', 'right', None])
def test_fresh_database_round_trip_and_delete(tmp_path, hand):
    path = tmp_path / 'fresh.db'
    manager = DatabaseManager(path)
    repository = SQLitePatientRepository(manager)
    patient_id = repository.add(Patient('Ana', 30, 'Outro', hand, 6, 'Notes'))
    patient = SQLitePatientRepository(DatabaseManager(path)).list_all()[0]
    assert patient.id == patient_id
    assert patient.affected_hand == hand
    assert patient.notes == 'Notes'
    manager.add_recording(patient_id, 'record.csv', 'task')
    assert manager.delete_patient(patient_id)
    assert repository.list_all() == []
    assert manager.get_patient_recordings(patient_id) == []
    assert not manager.delete_patient(patient_id)


def test_controller_legacy_unknown_and_invalid_updates(tmp_path):
    manager = DatabaseManager(tmp_path / 'patients.db')
    controller = PatientController.from_repository(SQLitePatientRepository(manager))
    patient_id = controller.register_patient({'name': 'Legacy import'})
    assert controller.list_patients()[0]['affected_hand'] is None
    for hand in ['Ambas', '', None, 'LEFT']:
        with pytest.raises(ValueError):
            controller.update_patient_affected_hand(patient_id, hand)
    with pytest.raises(ValueError, match='nao encontrado'):
        controller.update_patient_affected_hand(999, 'right')
    with pytest.raises(ValueError):
        controller.register_patient({'name': 'Invalid', 'affected_hand': 'Direita'})
    assert len(controller.list_patients()) == 1
    controller.update_patient_affected_hand(patient_id, 'left')
    assert controller.list_patients()[0]['affected_hand'] == 'left'
