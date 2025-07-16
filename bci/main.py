"""
Interface PyQt5 para Sistema BCI
Funcionalidades:
- Cadastro de pacientes
- Streaming de dados em tempo real com visualização
- Gravação de dados atrelada ao paciente com marcadores T1, T2 e Baseline
- Predição em tempo real durante o modo jogo
"""

import sys
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

# Importar configuração de caminhos
from bci.config import get_recording_path, get_database_path, ensure_folders_exist

# Importar o logger OpenBCI
try:
    from bci.openbci_csv_logger import OpenBCICSVLogger
    USE_OPENBCI_LOGGER = True
except ImportError:
    USE_OPENBCI_LOGGER = False
    print("OpenBCI Logger não encontrado, usando logger simples")



# Importar módulos do sistema existente
import sys
import os
import csv

# Não precisa adicionar ao path pois os módulos estão na mesma pasta agora
try:
    from bci.udp_receiver import UDPReceiver
    from bci.realtime_udp_converter import RealTimeUDPConverter
    from bci.csv_data_logger import CSVDataLogger
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
    
    # Usar nosso logger simples
    CSVDataLogger = SimpleCSVLogger





