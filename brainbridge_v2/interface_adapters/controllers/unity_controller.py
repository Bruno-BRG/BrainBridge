"""
Controller that adapts presentation requests to Unity communication use cases.
"""

from typing import Callable

from brainbridge_v2.application.ports.unity_gateway import UnityGateway
from brainbridge_v2.application.use_cases.unity_use_cases import (
    EndUnitySessionUseCase,
    EndUnityTaskUseCase,
    SendUnityActionUseCase,
    SendUnityTriggerUseCase,
    SetPendingUnitySessionUseCase,
    StartUnityServerUseCase,
    StartUnitySessionUseCase,
    StopUnityServerUseCase,
)


class UnityController:
    """
    Presentation-facing controller for Unity workflows.
    """

    def __init__(
        self,
        gateway: UnityGateway,
        start_server_use_case: StartUnityServerUseCase,
        stop_server_use_case: StopUnityServerUseCase,
        send_action_use_case: SendUnityActionUseCase,
        send_trigger_use_case: SendUnityTriggerUseCase,
        end_task_use_case: EndUnityTaskUseCase,
        end_session_use_case: EndUnitySessionUseCase,
        start_session_use_case: StartUnitySessionUseCase | None = None,
        set_pending_session_use_case: SetPendingUnitySessionUseCase | None = None,
    ):
        self._gateway = gateway
        self._start_server_use_case = start_server_use_case
        self._stop_server_use_case = stop_server_use_case
        self._send_action_use_case = send_action_use_case
        self._send_trigger_use_case = send_trigger_use_case
        self._end_task_use_case = end_task_use_case
        self._end_session_use_case = end_session_use_case
        self._start_session_use_case = start_session_use_case or StartUnitySessionUseCase(gateway)
        self._set_pending_session_use_case = set_pending_session_use_case or SetPendingUnitySessionUseCase(gateway)

    @classmethod
    def from_gateway(cls, gateway: UnityGateway) -> "UnityController":
        return cls(
            gateway=gateway,
            start_server_use_case=StartUnityServerUseCase(gateway),
            stop_server_use_case=StopUnityServerUseCase(gateway),
            send_action_use_case=SendUnityActionUseCase(gateway),
            send_trigger_use_case=SendUnityTriggerUseCase(gateway),
            end_task_use_case=EndUnityTaskUseCase(gateway),
            end_session_use_case=EndUnitySessionUseCase(gateway),
        )

    def start_server(self) -> bool:
        return self._start_server_use_case.execute()

    def stop_server(self) -> None:
        self._stop_server_use_case.execute()

    def start_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> bool:
        return self._start_session_use_case.execute(nome, nivel, lado, tarefa, sessoes)

    def set_pending_session(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> None:
        self._set_pending_session_use_case.execute(nome, nivel, lado, tarefa, sessoes)

    def send_action(self, action: str) -> bool:
        return self._send_action_use_case.execute(action)

    def send_trigger(self) -> bool:
        return self._send_trigger_use_case.execute()

    def end_task(self) -> bool:
        return self._end_task_use_case.execute()

    def end_session(self, message: str) -> bool:
        return self._end_session_use_case.execute(message)

    def is_server_active(self) -> bool:
        return self._gateway.is_server_active()

    def is_client_connected(self) -> bool:
        return self._gateway.is_client_connected()

    def set_message_callback(self, callback: Callable[[str], None]) -> None:
        self._gateway.set_message_callback(callback)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._gateway.set_connection_callback(callback)
