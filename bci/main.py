"""
Interface PyQt5 para Sistema BCI
Funcionalidades:
- Cadastro de pacientes
- Streaming de dados em tempo real com visualização
- Gravação de dados atrelada ao paciente com marcadores T1, T2 e Baseline
- Predição em tempo real durante o modo jogo
"""

import sys
import os
import time
import random
import sqlite3
import os
import threading
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QLabel, QLineEdit, 
                           QPushButton, QComboBox, QSpinBox, QTextEdit,
                           QTableWidget, QTableWidgetItem, QGroupBox,
                           QGridLayout, QMessageBox, QProgressBar, QCheckBox,
                           QSplitter, QFrame, QDateEdit, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDate
from PyQt5.QtGui import QFont, QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# Detectar se estamos executando como módulo ou script direto
if __name__ == "__main__" and __package__ is None:
    # Executando como script direto, adicionar o diretório pai ao path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Usar imports absolutos
    from bci.configs.config import get_recording_path, get_database_path, ensure_folders_exist
else:
    # Executando como módulo, usar imports relativos
    from .configs.config import get_recording_path, get_database_path, ensure_folders_exist

# Importar configuração de caminhos

# Importar o logger OpenBCI
if __name__ == "__main__" and __package__ is None:
    # Executando como script direto
    try:
        from bci.network.openbci_csv_logger import OpenBCICSVLogger
        USE_OPENBCI_LOGGER = True
    except ImportError:
        USE_OPENBCI_LOGGER = False
        print("OpenBCI Logger não encontrado, usando logger simples")
else:
    # Executando como módulo
    try:
        from .network.openbci_csv_logger import OpenBCICSVLogger
        USE_OPENBCI_LOGGER = True
    except ImportError:
        USE_OPENBCI_LOGGER = False
        print("OpenBCI Logger não encontrado, usando logger simples")


# Importar módulos do sistema existente
import csv

# Importar logger simples
if __name__ == "__main__" and __package__ is None:
    try:
        from bci.network.simple_csv_logger import SimpleCSVLogger
    except ImportError:
        SimpleCSVLogger = None
else:
    try:
        from .network.simple_csv_logger import SimpleCSVLogger
    except ImportError:
        SimpleCSVLogger = None

# Não precisa adicionar ao path pois os módulos estão na mesma pasta agora
if __name__ == "__main__" and __package__ is None:
    try:
        from bci.network.udp_receiver_BCI import UDPReceiver
        from bci.network.realtime_udp_converter import RealTimeUDPConverter
        from bci.network.csv_data_logger import CSVDataLogger
        print("Módulos do sistema carregados com sucesso")
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Usando modo de simulação")
else:
    try:
        from .network.udp_receiver_BCI import UDPReceiver
        from .network.realtime_udp_converter import RealTimeUDPConverter
        from .network.csv_data_logger import CSVDataLogger
        print("Módulos do sistema carregados com sucesso")
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Usando modo de simulação")
    # Criar classes mock para desenvolvimento
    class UDPReceiver:
        def __init__(self, host, port): 
            self.host = host
            self.port = port
            self.callback = None
            
        def set_callback(self, callback): 
            self.callback = callback
            
        def start(self): 
            # Simular erro de porta ocupada ocasionalmente
            import random
            if random.random() < 0.3:
                raise Exception("[WinError 10048] Porta já em uso")
            
        def stop(self): 
            pass
    
    class RealTimeUDPConverter:
        def __init__(self): pass
        def process_udp_data(self, data): 
            # Simular dados EEG para desenvolvimento
            return np.random.randn(16) * 50
    
    # Usar nosso logger simples se disponível
    if SimpleCSVLogger:
        CSVDataLogger = SimpleCSVLogger
    else:
        # Criar um logger mock básico
        class CSVDataLogger:
            def __init__(self, filename): 
                self.filename = filename
            def start_logging(self): pass
            def stop_logging(self): pass
            def log_data(self, data): pass


# Importar a janela principal
if __name__ == "__main__" and __package__ is None:
    try:
        from bci.ui.BCI_main_window import BCIMainWindow
    except ImportError as e:
        print(f"Erro ao importar BCIMainWindow: {e}")
        BCIMainWindow = None
else:
    try:
        from .ui.BCI_main_window import BCIMainWindow
    except ImportError as e:
        print(f"Erro ao importar BCIMainWindow: {e}")
        BCIMainWindow = None


def main():
    """Função principal do sistema BCI"""
    # Garantir que as pastas necessárias existam
    ensure_folders_exist()
    
    # Criar aplicação PyQt5
    app = QApplication(sys.argv)
    
    if BCIMainWindow:
        # Usar a interface principal modularizada
        window = BCIMainWindow()
        window.show()
    else:
        # Mostrar erro se não conseguir carregar a interface
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Erro")
        msg.setText("Não foi possível carregar a interface principal do sistema BCI.")
        msg.setDetailedText("Verifique se todos os módulos estão instalados corretamente.")
        msg.exec_()
        return
    
    # Executar aplicação
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()





