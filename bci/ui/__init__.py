"""
UI Module - Interface de usuário

Este módulo contém widgets e janelas da interface gráfica.
"""

# Imports condicionais para evitar erros se PyQt não estiver instalado
try:
    from .BCI_main_window import *
    from .EEG_plot_widget import *
    from .patient_registration_widget import *
    from .streaming_widget import *
    
    __all__ = [
        'BCI_main_window',
        'EEG_plot_widget', 
        'patient_registration_widget',
        'streaming_widget'
    ]
except ImportError as e:
    print(f"Aviso: Algumas dependências da UI não estão disponíveis: {e}")
    __all__ = []
