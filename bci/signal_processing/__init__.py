"""
Signal Processing Module - Módulos de processamento de sinais

Este módulo contém classes e funções para processamento de sinais EEG,
incluindo filtros e transformações.
"""

from .butter_filter import ButterworthFilter

__all__ = [
    'ButterworthFilter'
]
