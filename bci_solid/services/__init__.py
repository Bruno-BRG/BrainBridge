"""
Serviços do Sistema BCI seguindo SOLID

Implementações de serviços que seguem o DIP (Dependency Inversion Principle),
dependendo de abstrações ao invés de implementações concretas.
"""

from .patient_service import PatientService
from .session_service import SessionService
from .data_streaming_service import DataStreamingService
from .validation_service import ValidationService

__all__ = [
    'PatientService',
    'SessionService',
    'DataStreamingService', 
    'ValidationService'
]
