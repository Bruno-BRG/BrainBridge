"""
Network Module - Módulos de rede e comunicação UDP

Este módulo contém classes para captura e processamento de dados via UDP.
"""

from .udp_receiver import UDPReceiver
from .csv_data_logger import CSVDataLogger

# Imports opcionais para outros módulos
try:
    from .openbci_csv_logger import *
except ImportError:
    pass

try:
    from .realtime_udp_converter import *
except ImportError:
    pass

try:
    from .simple_csv_logger import *
except ImportError:
    pass

__all__ = [
    'UDPReceiver',
    'CSVDataLogger'
]
