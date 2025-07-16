"""
BCI (Brain-Computer Interface) Package

Este pacote contém módulos para captura, processamento e análise de dados EEG.
"""

__version__ = "1.0.0"
__author__ = "Projeto BCI"

# Imports principais
from . import network
from . import ui
from . import configs
from . import database
from . import streaming_logic
from . import AI

__all__ = [
    'network',
    'ui', 
    'configs',
    'database',
    'streaming_logic',
    'AI'
]
