"""
Port definition for Unity communication operations.
"""

from typing import Callable, Protocol


class UnityGateway(Protocol):
    def start_server(self) -> bool:
        ...

    def stop_server(self) -> None:
        ...

    def start_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> bool:
        ...

    def set_pending_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> None:
        ...

    def send_action(self, action: str) -> bool:
        ...

    def send_trigger(self) -> bool:
        ...

    def end_task(self) -> bool:
        ...

    def end_session(self, message: str) -> bool:
        ...

    def is_server_active(self) -> bool:
        ...

    def is_client_connected(self) -> bool:
        ...

    def set_message_callback(self, callback: Callable[[str], None]) -> None:
        ...

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        ...
