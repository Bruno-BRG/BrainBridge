"""
ESP32 gateway adapter backed by the concrete ESP32 communicator.
"""

from typing import Callable

from brainbridge_v2.infrastructure.communication.esp32 import (
    ESP32SerialCommunicator,
    get_esp32_communicator,
)


class ESP32GatewayAdapter:
    """
    Infrastructure adapter that maps ESP32Gateway to ESP32SerialCommunicator.
    """

    def __init__(self, communicator: ESP32SerialCommunicator | None = None):
        self._communicator = communicator or get_esp32_communicator()

    def connect(self) -> bool:
        return self._communicator.connect()

    def connect_report(self, port: str | None = None) -> dict:
        comm = self._communicator
        if port:
            comm.port = port
        connected = bool(comm.connect())
        report = {"connected": connected, "port": comm.port}
        if not connected:
            try:
                available = [p[0] for p in comm.list_available_ports()]
            except Exception:
                available = []
            if comm.port not in (available or []):
                reason = f"Porta {comm.port} nao encontrada."
                if available:
                    reason += f" Disponiveis: {', '.join(available)}."
                else:
                    reason += " Nenhuma porta serial detectada."
            else:
                reason = f"Falha ao abrir {comm.port}. Verifique cabo, permissao e se outro programa usa a porta."
            report["reason"] = reason
            report["available_ports"] = available
        return report

    def get_port(self) -> str:
        return str(self._communicator.port)

    def list_ports(self) -> list:
        try:
            return [{"port": p[0], "description": p[1] if len(p) > 1 else ""}
                    for p in self._communicator.list_available_ports()]
        except Exception:
            return []

    def disconnect(self) -> None:
        self._communicator.disconnect()

    def send_direction(self, direction: str) -> bool:
        if direction == "esquerda":
            return self._communicator.send_trigger_left()
        if direction == "direita":
            return self._communicator.send_trigger_right()
        raise ValueError(f"Direcao ESP32 invalida: {direction}")

    def is_connected(self) -> bool:
        return bool(self._communicator.is_connected)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._communicator.set_connection_callback(callback)
