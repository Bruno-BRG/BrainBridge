"""
Interfaces do Sistema BCI seguindo SOLID

Este módulo define contratos claros para todas as funcionalidades do sistema,
seguindo o princípio de Segregação de Interface (ISP).
"""

from .data_interfaces import IPatientRepository, ISessionRepository, IDataLogger
from .service_interfaces import IDataStreamingService, IDataProcessingService, IModelService
from .communication_interfaces import INetworkCommunicator, IUnityBridge
from .ui_interfaces import IView, IController, IWidget

__all__ = [
    # Data interfaces
    'IPatientRepository',
    'ISessionRepository', 
    'IDataLogger',
    
    # Service interfaces
    'IDataStreamingService',
    'IDataProcessingService',
    'IModelService',
    
    # Communication interfaces
    'INetworkCommunicator',
    'IUnityBridge',
    
    # UI interfaces
    'IView',
    'IController',
    'IWidget'
]
