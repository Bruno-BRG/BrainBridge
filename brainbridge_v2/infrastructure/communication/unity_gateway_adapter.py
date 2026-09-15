"""
Unity gateway adapter backed by the concrete Unity communicator.
"""

from typing import Callable

from brainbridge_v2.infrastructure.communication.unity import (
    PatientData,
    TaskType,
    UDP_sender,
    UnityCommunicator,
)


class UnityGatewayAdapter:
    """
    Infrastructure adapter that maps UnityGateway to UnityCommunicator.
    """

    def __init__(self, communicator: UnityCommunicator | None = None):
        self._communicator = communicator or UnityCommunicator()

    def start_server(self) -> bool:
        return self._communicator.start_server()

    def stop_server(self) -> None:
        self._communicator.stop_server()

    @staticmethod
    def _build_session_args(nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0):
        clean_nome = (nome or "").strip() or "Paciente"
        try:
            clean_nivel = int(nivel)
        except (TypeError, ValueError):
            clean_nivel = 5
        clean_nivel = max(0, min(11, clean_nivel))
        clean_lado = (lado or "").strip().capitalize()
        if clean_lado not in ("Direito", "Esquerdo"):
            lowered = (lado or "").strip().lower()
            if lowered in ("right", "direita", "direito"):
                clean_lado = "Direito"
            else:
                clean_lado = "Esquerdo"
        task = TaskType.JOGO if (tarefa or "").strip().lower() == "jogo" else TaskType.TREINO
        try:
            clean_sessoes = int(sessoes)
        except (TypeError, ValueError):
            clean_sessoes = 0
        clean_sessoes = max(0, clean_sessoes)
        return PatientData(nome=clean_nome, nivel=clean_nivel, lado=clean_lado, sessoes=clean_sessoes), task

    def start_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> bool:
        patient, task = self._build_session_args(nome, nivel, lado, tarefa, sessoes)
        return self._communicator.start_session(patient, task)

    def set_pending_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> None:
        patient, task = self._build_session_args(nome, nivel, lado, tarefa, sessoes)
        self._communicator.set_pending_session(patient, task)

    def send_action(self, action: str) -> bool:
        return UDP_sender.enviar_sinal(action)

    def send_trigger(self) -> bool:
        return self._communicator.send_trigger()

    def end_task(self) -> bool:
        return self._communicator.end_task()

    def end_session(self, message: str) -> bool:
        return self._communicator.end_session(message)

    def is_server_active(self) -> bool:
        return bool(self._communicator.is_active)

    def is_client_connected(self) -> bool:
        return bool(self._communicator.tcp_connected)

    def set_message_callback(self, callback: Callable[[str], None]) -> None:
        self._communicator.set_message_callback(callback)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._communicator.set_connection_callback(callback)
