"""
AI Module - Inteligência Artificial e Machine Learning

Este módulo contém modelos de IA para processamento de dados EEG.
"""

try:
    from .EEGNet import *
    __all__ = ['EEGNet']
except ImportError as e:
    print(f"Aviso: Dependências de AI não estão disponíveis: {e}")
    __all__ = []
