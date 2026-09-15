"""
Unity communication use cases.
"""

from brainbridge_v2.application.ports.unity_gateway import UnityGateway


class StartUnityServerUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self) -> bool:
        return self._gateway.start_server()


class StopUnityServerUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self) -> None:
        self._gateway.stop_server()


class StartUnitySessionUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> bool:
        return self._gateway.start_session(nome, nivel, lado, tarefa, sessoes)


class SetPendingUnitySessionUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self, nome: str, nivel: int, lado: str, tarefa: str, sessoes: int = 0) -> None:
        self._gateway.set_pending_session(nome, nivel, lado, tarefa, sessoes)


class SendUnityActionUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self, action: str) -> bool:
        if not action or not action.strip():
            raise ValueError("Acao Unity eh obrigatoria.")
        return self._gateway.send_action(action.strip())


class SendUnityTriggerUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self) -> bool:
        return self._gateway.send_trigger()


class EndUnityTaskUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self) -> bool:
        return self._gateway.end_task()


class EndUnitySessionUseCase:
    def __init__(self, gateway: UnityGateway):
        self._gateway = gateway

    def execute(self, message: str) -> bool:
        return self._gateway.end_session(message)
