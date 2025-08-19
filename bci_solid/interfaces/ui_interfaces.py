"""
Interfaces de UI - Seguindo ISP (Interface Segregation Principle)

Estas interfaces definem contratos para componentes de interface,
seguindo o padrão MVP/MVC e permitindo testabilidade.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from PyQt5.QtWidgets import QWidget


class IView(ABC):
    """Interface base para views"""
    
    @abstractmethod
    def show(self) -> None:
        """Exibe a view"""
        pass
    
    @abstractmethod
    def hide(self) -> None:
        """Oculta a view"""
        pass
    
    @abstractmethod
    def update_view(self, data: Dict[str, Any]) -> None:
        """Atualiza a view com novos dados"""
        pass
    
    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """Habilita/desabilita a view"""
        pass


class IController(ABC):
    """Interface base para controllers"""
    
    @abstractmethod
    def handle_action(self, action: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Manipula uma ação da interface"""
        pass
    
    @abstractmethod
    def set_view(self, view: IView) -> None:
        """Define a view associada"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializa o controller"""
        pass


class IWidget(ABC):
    """Interface para widgets customizados"""
    
    @abstractmethod
    def setup_ui(self) -> None:
        """Configura a interface do widget"""
        pass
    
    @abstractmethod
    def connect_signals(self) -> None:
        """Conecta sinais e slots"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Limpa recursos do widget"""
        pass


class IPatientView(IView):
    """Interface específica para view de pacientes"""
    
    @abstractmethod
    def display_patients(self, patients: List[Dict[str, Any]]) -> None:
        """Exibe lista de pacientes"""
        pass
    
    @abstractmethod
    def show_patient_form(self, patient_data: Optional[Dict[str, Any]] = None) -> None:
        """Mostra formulário de paciente"""
        pass
    
    @abstractmethod
    def get_patient_form_data(self) -> Dict[str, Any]:
        """Obtém dados do formulário"""
        pass
    
    @abstractmethod
    def clear_form(self) -> None:
        """Limpa o formulário"""
        pass


class IStreamingView(IView):
    """Interface específica para view de streaming"""
    
    @abstractmethod
    def update_connection_status(self, status: Dict[str, Any]) -> None:
        """Atualiza status da conexão"""
        pass
    
    @abstractmethod
    def update_data_plot(self, data: Any) -> None:
        """Atualiza plot de dados"""
        pass
    
    @abstractmethod
    def set_recording_status(self, is_recording: bool) -> None:
        """Define status de gravação"""
        pass
    
    @abstractmethod
    def enable_markers(self, enabled: bool) -> None:
        """Habilita/desabilita marcadores"""
        pass


class IPlotWidget(IWidget):
    """Interface para widget de plots"""
    
    @abstractmethod
    def plot_data(self, data: Any, config: Optional[Dict[str, Any]] = None) -> None:
        """Plota dados"""
        pass
    
    @abstractmethod
    def clear_plot(self) -> None:
        """Limpa o plot"""
        pass
    
    @abstractmethod
    def set_plot_config(self, config: Dict[str, Any]) -> None:
        """Define configuração do plot"""
        pass


class IEventHandler(ABC):
    """Interface para manipulação de eventos"""
    
    @abstractmethod
    def register_event_handler(self, event: str, handler: Callable[..., None]) -> None:
        """Registra manipulador de evento"""
        pass
    
    @abstractmethod
    def emit_event(self, event: str, *args, **kwargs) -> None:
        """Emite um evento"""
        pass
    
    @abstractmethod
    def remove_event_handler(self, event: str, handler: Callable[..., None]) -> None:
        """Remove manipulador de evento"""
        pass
