"""
Modelos de Domínio - Seguindo SRP (Single Responsibility Principle)

Cada modelo tem uma única responsabilidade e representa uma entidade de negócio.
"""

from .patient import Patient
from .session import Session, SessionType, SessionStatus
from .eeg_data import EEGData, EEGChannel, MarkerType
from .recording import Recording

__all__ = [
    'Patient',
    'Session',
    'SessionType', 
    'SessionStatus',
    'EEGData',
    'EEGChannel',
    'MarkerType',
    'Recording'
]
