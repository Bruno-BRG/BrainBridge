"""
Sistema BCI Refatorado seguindo Princípios SOLID

Este módulo principal demonstra como a nova arquitetura SOLID
resolve os problemas da implementação anterior.
"""

from .interfaces import *
from .models import *
from .services import *
from .controllers import *
from .factories import *

__version__ = "2.0.0-SOLID"
__author__ = "Projeto BCI Team - Refatoração SOLID"

__all__ = [
    # Interfaces
    'IPatientRepository',
    'ISessionRepository', 
    'IDataLogger',
    'IDataStreamingService',
    'IValidationService',
    'INetworkCommunicator',
    'IUnityBridge',
    'IView',
    'IController',
    'IWidget',
    
    # Models
    'Patient',
    'Session',
    'SessionType', 
    'SessionStatus',
    'EEGData',
    'EEGChannel',
    'MarkerType',
    'Recording',
    
    # Services
    'PatientService',
    'SessionService',
    'DataStreamingService',
    'ValidationService',
    
    # Controllers
    'PatientController',
    
    # Factories
    'ServiceFactory',
]
