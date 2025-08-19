"""
Interfaces de Comunicação - Seguindo ISP (Interface Segregation Principle)

Estas interfaces definem contratos para comunicação de rede,
permitindo diferentes implementações (UDP, TCP, ZMQ, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List


class INetworkCommunicator(ABC):
    """Interface base para comunicação de rede"""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Estabelece conexão"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Encerra conexão"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        pass
    
    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """Retorna informações da conexão"""
        pass


class IDataReceiver(ABC):
    """Interface para recepção de dados"""
    
    @abstractmethod
    def start_receiving(self) -> bool:
        """Inicia recepção de dados"""
        pass
    
    @abstractmethod
    def stop_receiving(self) -> bool:
        """Para recepção de dados"""
        pass
    
    @abstractmethod
    def subscribe_to_data(self, callback: Callable[[bytes], None]) -> bool:
        """Subscreve a callback para dados recebidos"""
        pass


class ICommandSender(ABC):
    """Interface para envio de comandos"""
    
    @abstractmethod
    def send_command(self, command: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Envia comando"""
        pass
    
    @abstractmethod
    def send_bulk_commands(self, commands: List[Dict[str, Any]]) -> bool:
        """Envia múltiplos comandos"""
        pass


class IUnityBridge(ABC):
    """Interface específica para comunicação com Unity"""
    
    @abstractmethod
    def send_movement_command(self, movement_type: str, intensity: float = 1.0) -> bool:
        """Envia comando de movimento para Unity"""
        pass
    
    @abstractmethod
    def send_trigger(self, trigger_type: str) -> bool:
        """Envia trigger para Unity"""
        pass
    
    @abstractmethod
    def get_unity_status(self) -> Dict[str, Any]:
        """Obtém status do Unity"""
        pass
    
    @abstractmethod
    def register_unity_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Registra callback para eventos do Unity"""
        pass


class IConnectionManager(ABC):
    """Interface para gerenciamento de múltiplas conexões"""
    
    @abstractmethod
    def add_connection(self, name: str, communicator: INetworkCommunicator) -> bool:
        """Adiciona uma nova conexão"""
        pass
    
    @abstractmethod
    def remove_connection(self, name: str) -> bool:
        """Remove uma conexão"""
        pass
    
    @abstractmethod
    def get_connection(self, name: str) -> Optional[INetworkCommunicator]:
        """Obtém uma conexão específica"""
        pass
    
    @abstractmethod
    def get_all_connections(self) -> Dict[str, INetworkCommunicator]:
        """Retorna todas as conexões"""
        pass
