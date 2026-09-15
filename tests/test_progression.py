"""
Progressão do paciente: contagem de sessões, nível VR e JSON completo.
"""
import json

import pytest

from brainbridge_v2.infrastructure.communication.unity import PatientData
from brainbridge_v2.infrastructure.communication.unity_gateway_adapter import UnityGatewayAdapter
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import ProgressionPresenter


def test_patient_data_sessoes_no_json_e_legivel():
    p = PatientData(nome="Maria", nivel=3, lado="Direito", sessoes=4)
    payload = json.loads(p.to_json())
    assert payload == {"nome": "Maria", "nivel": 3, "lado": "Direito", "sessoes": 4}
    msg = p.format_message()
    assert "Sessoes: 4" in msg
    assert "Nome: Maria" in msg and "Nivel: 3" in msg and "Lado: Direito" in msg


def test_patient_data_sessoes_default_e_validacao():
    assert PatientData(nome="X", nivel=0, lado="Esquerdo").sessoes == 0
    with pytest.raises(ValueError):
        PatientData(nome="X", nivel=0, lado="Esquerdo", sessoes=-1)
    with pytest.raises(ValueError):
        PatientData(nome="X", nivel=0, lado="Esquerdo", sessoes="tres")


def test_progression_level():
    assert ProgressionPresenter.level_for_session_count(0) == 0
    assert ProgressionPresenter.level_for_session_count(1) == 1
    assert ProgressionPresenter.level_for_session_count(5) == 5
    assert ProgressionPresenter.level_for_session_count(11) == 11
    assert ProgressionPresenter.level_for_session_count(12) == 11
    assert ProgressionPresenter.level_for_session_count(100) == 11
    assert ProgressionPresenter.level_for_session_count(-3) == 0
    assert ProgressionPresenter.level_for_session_count("abc") == 0


def test_progression_summary_text():
    assert ProgressionPresenter.summary_text(0) == "0 sessões • Nível 0"
    assert ProgressionPresenter.summary_text(1) == "1 sessão • Nível 1"
    assert ProgressionPresenter.summary_text(15) == "15 sessões • Nível 11"


def test_gateway_adapter_repassa_sessoes():
    sent = {}

    class FakeComm:
        def start_session(self, patient, task):
            sent["patient"] = patient
            sent["task"] = task
            return True

        def set_pending_session(self, patient, task):
            sent["patient"] = patient
            sent["task"] = task

    adapter = UnityGatewayAdapter(communicator=FakeComm())
    assert adapter.set_pending_session("Ana", 2, "Direito", "jogo", 7) is None
    assert sent["patient"].sessoes == 7
    assert sent["patient"].nivel == 2
    assert sent["task"].value == "Jogo"
    # Compat: chamada antiga com 4 args continua valendo (sessoes=0)
    adapter.set_pending_session("Bia", 1, "Esquerdo", "treino")
    assert sent["patient"].sessoes == 0
