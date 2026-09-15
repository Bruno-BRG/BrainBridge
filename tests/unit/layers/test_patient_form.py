import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PyQt5')
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox

from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import SQLitePatientRepository
from brainbridge_v2.interface_adapters.controllers.patient_controller import PatientController
from brainbridge_v2.presentation.gui.widgets.patient_form import PatientRegistrationWidget


@pytest.fixture
def form(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    for method in ['warning', 'information', 'critical']:
        monkeypatch.setattr(QMessageBox, method, lambda *args: QMessageBox.Ok)
    controller = PatientController.from_repository(
        SQLitePatientRepository(DatabaseManager(tmp_path / 'form.db'))
    )
    widget = PatientRegistrationWidget(controller)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_form_requires_explicit_hand_and_resets(form, hand):
    form.name_edit.setText('Ana')
    assert form.hand_combo.currentData() is None
    form.register_patient()
    assert form.patient_controller.list_patients() == []
    form.hand_combo.setCurrentIndex(form.hand_combo.findData(hand))
    form.register_patient()
    assert form.patient_controller.list_patients()[0]['affected_hand'] == hand
    assert form.hand_combo.currentData() is None


def test_form_updates_legacy_patient_without_changing_other_fields(form, monkeypatch):
    patient_id = form.patient_controller.register_patient({'name': 'Legacy', 'notes': 'Preserve'})
    before = form.patient_controller.list_patients()[0]
    form.load_patients()
    assert form.patients_table.item(0, 4).text() == 'Não informada'
    form.patients_table.selectRow(0)
    assert form.get_selected_patient() == patient_id
    assert form.update_hand_btn.isEnabled()
    for label, accepted in [('Direita', False), ('Selecione...', True)]:
        monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (label, accepted))
        form.update_selected_patient_hand()
        assert form.patient_controller.list_patients()[0] == before
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('Direita', True))
    form.update_selected_patient_hand()
    assert form.patient_controller.list_patients() == [dict(before, affected_hand='right')]
    assert form.patients_table.item(0, 4).text() == 'Direita'
