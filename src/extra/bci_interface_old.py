"""
Interface PyQt5 para Sistema BCI
Funcionalidades:
- Cadastro de pacientes
- Streaming de dados em tempo real com visualização
- Gravação de dados atrelada ao paciente com marcadores T1, T2 e Baseline
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

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QLabel, QLineEdit, 
                           QPushButton, QComboBox, QSpinBox, QTextEdit,
                           QTableWidget, QTableWidgetItem, QGroupBox,
                           QGridLayout, QMessageBox, QProgressBar, QCheckBox,
                           QSplitter, QFrame, QDateEdit, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDate
from PyQt5.QtGui import QFont, QPixmap, QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Importar configuração de caminhos
from config import get_recording_path, get_database_path, ensure_folders_exist

class SimpleCSVLogger:
    """Logger CSV simples para dados EEG com suporte a marcadores"""
    
    def __init__(self, filename):
        self.filename = filename
        self.is_logging = False
        self.data_buffer = []
        self.lock = threading.Lock()
        self.sample_count = 0
        self.pending_t0_marker = None
        self.t0_samples_remaining = 0
        
    def start_logging(self):
        """Inicia a gravação"""
        self.is_logging = True
        self.sample_count = 0
        # Criar cabeçalho
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Cabeçalho com marcador
            header = ['Timestamp'] + [f'EXG Channel {i}' for i in range(16)] + ['Marker']
            writer.writerow(header)
        print(f"Gravação iniciada: {self.filename}")
        
    def log_data(self, eeg_data, marker=None):
        """Adiciona dados ao log com marcador opcional"""
        if self.is_logging:
            with self.lock:
                timestamp = datetime.now().isoformat()
                
                # Verificar se deve inserir T0 automaticamente
                if self.t0_samples_remaining > 0:
                    self.t0_samples_remaining -= 1
                    if self.t0_samples_remaining == 0 and self.pending_t0_marker:
                        marker = self.pending_t0_marker
                        self.pending_t0_marker = None
                
                row = [timestamp] + list(eeg_data) + [marker if marker else '']
                self.data_buffer.append(row)
                self.sample_count += 1
                
                # Salvar a cada 10 amostras
                if len(self.data_buffer) >= 10:
                    self._flush_buffer()
    
    def add_marker(self, marker_type):
        """Adiciona um marcador e programa T0 se necessário"""
        if marker_type in ['T1', 'T2']:
            # Programar T0 para 400 amostras depois
            self.pending_t0_marker = 'T0'
            self.t0_samples_remaining = 400
        return marker_type
    
    def _flush_buffer(self):
        """Salva o buffer no arquivo"""
        if self.data_buffer:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_buffer)
            self.data_buffer.clear()
    
    def stop_logging(self):
        """Para a gravação"""
        self.is_logging = False
        with self.lock:
            if self.data_buffer:
                self._flush_buffer()
        print(f"Gravação finalizada: {self.filename}")


# Importar módulos do sistema existente
import sys
import os
import csv

# Não precisa adicionar ao path pois os módulos estão na mesma pasta agora
try:
    from udp_receiver import UDPReceiver
    from realtime_udp_converter import RealTimeUDPConverter
    from csv_data_logger import CSVDataLogger
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


class DatabaseManager:
    """Gerenciador do banco de dados SQLite para pacientes"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_database_path()
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                affected_hand TEXT,
                time_since_event INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                filename TEXT,
                start_time TIMESTAMP,
                duration INTEGER,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_patient(self, name: str, age: int, sex: str, affected_hand: str, 
                   time_since_event: int, notes: str = ""):
        """Adiciona um novo paciente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (name, age, sex, affected_hand, time_since_event, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, age, sex, affected_hand, time_since_event, notes))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def get_all_patients(self) -> List[Dict]:
        """Retorna todos os pacientes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        patients = cursor.fetchall()
        conn.close()
        
        return [{"id": p[0], "name": p[1], "age": p[2], "sex": p[3], 
                "affected_hand": p[4], "time_since_event": p[5], 
                "created_at": p[6], "notes": p[7]} for p in patients]
    
    def add_recording(self, patient_id: int, filename: str, notes: str = ""):
        """Adiciona uma nova gravação"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recordings (patient_id, filename, start_time, notes)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, filename, datetime.now().isoformat(), notes))
        
        conn.commit()
        conn.close()


class EEGPlotWidget(QWidget):
    """Widget para plotar dados EEG em tempo real"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_plot()
        
        # Buffer para dados
        self.data_buffer = deque(maxlen=1000)  # 8 segundos a 125 Hz
        self.time_buffer = deque(maxlen=1000)
        self.current_time = 0
        
        # Timer para atualizar plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 FPS
        
    def setup_ui(self):
        """Configura a interface do widget"""
        layout = QVBoxLayout()
        
        # Controles
        controls_layout = QHBoxLayout()
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([f"Canal {i}" for i in range(16)])
        self.channel_combo.addItem("Todos os Canais")
        self.channel_combo.currentTextChanged.connect(self.change_channel)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Auto", "±50µV", "±100µV", "±200µV", "±500µV"])
        self.scale_combo.currentTextChanged.connect(self.change_scale)
        
        controls_layout.addWidget(QLabel("Canal:"))
        controls_layout.addWidget(self.channel_combo)
        controls_layout.addWidget(QLabel("Escala:"))
        controls_layout.addWidget(self.scale_combo)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Área do plot
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def setup_plot(self):
        """Configura o plot inicial"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 8)  # 8 segundos
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('Tempo (s)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title('Dados EEG em Tempo Real')
        self.ax.grid(True, alpha=0.3)
        
        # Linhas para cada canal
        self.lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 16))
        
        for i in range(16):
            line, = self.ax.plot([], [], color=colors[i], linewidth=0.8, 
                               label=f'Canal {i}', alpha=0.7)
            self.lines.append(line)
            
        self.canvas.draw()
        
    def add_data(self, eeg_data: np.ndarray):
        """Adiciona novos dados EEG"""
        if len(eeg_data) == 16:  # 16 canais
            self.data_buffer.append(eeg_data)
            self.time_buffer.append(self.current_time)
            self.current_time += 1/125  # 125 Hz
            
    def update_plot(self):
        """Atualiza o plot com novos dados"""
        if len(self.data_buffer) == 0:
            return
            
        # Converter buffer para arrays numpy
        times = np.array(self.time_buffer)
        data = np.array(self.data_buffer)
        
        if len(times) < 2:
            return
            
        # Atualizar janela de tempo
        current_time = times[-1]
        window_start = max(0, current_time - 8)
        
        # Filtrar dados da janela
        mask = times >= window_start
        windowed_times = times[mask] - window_start
        windowed_data = data[mask]
        
        # Atualizar cada linha
        selected_channel = self.channel_combo.currentText()
        
        if selected_channel == "Todos os Canais":
            # Mostrar todos os canais com offset
            for i in range(16):
                if len(windowed_data) > 0:
                    y_data = windowed_data[:, i] + i * 100  # Offset vertical
                    self.lines[i].set_data(windowed_times, y_data)
                    self.lines[i].set_visible(True)
            
            self.ax.set_ylim(-100, 1600)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Mostrar apenas um canal
            channel_idx = int(selected_channel.split()[1])
            
            for i in range(16):
                if i == channel_idx and len(windowed_data) > 0:
                    self.lines[i].set_data(windowed_times, windowed_data[:, i])
                    self.lines[i].set_visible(True)
                else:
                    self.lines[i].set_visible(False)
            
            # Ajustar escala
            scale_text = self.scale_combo.currentText()
            if scale_text == "Auto":
                if len(windowed_data) > 0:
                    y_data = windowed_data[:, channel_idx]
                    if len(y_data) > 0:
                        y_min, y_max = np.min(y_data), np.max(y_data)
                        margin = (y_max - y_min) * 0.1
                        self.ax.set_ylim(y_min - margin, y_max + margin)
            else:
                scale_val = int(scale_text.replace("±", "").replace("µV", ""))
                self.ax.set_ylim(-scale_val, scale_val)
            
            self.ax.legend().set_visible(False)
        
        self.ax.set_xlim(0, 8)
        self.canvas.draw()
        
    def change_channel(self):
        """Callback para mudança de canal"""
        self.setup_plot()
        
    def change_scale(self):
        """Callback para mudança de escala"""
        pass


class StreamingThread(QThread):
    """Thread para streaming de dados"""
    
    data_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self.udp_receiver = None
        self.data_queue = deque(maxlen=100)
        self.is_mock_mode = False
        
    def start_streaming(self, host='localhost', port=12345):
        """Inicia o streaming"""
        self.host = host
        self.port = port
        self.is_running = True
        self.start()
        
    def stop_streaming(self):
        """Para o streaming"""
        self.is_running = False
        if self.udp_receiver:
            self.udp_receiver.stop()
        self.quit()
        self.wait()
        
    def run(self):
        """Executa o streaming"""
        try:
            # Tentar configurar receptor UDP
            self.udp_receiver = UDPReceiver(self.host, self.port)
            
            # Callback para dados recebidos
            def on_data_received(data):
                try:
                    # Processar dados UDP para extrair EEG
                    eeg_data = self.extract_eeg_from_udp(data)
                    if eeg_data is not None:
                        # Se é uma lista de amostras, processar cada uma
                        if isinstance(eeg_data, list):
                            for sample in eeg_data:
                                self.data_received.emit(sample)
                        else:
                            # Se é uma única amostra
                            self.data_received.emit(eeg_data)
                except Exception as e:
                    print(f"Erro ao processar dados: {e}")
            
            self.udp_receiver.set_callback(on_data_received)
            self.udp_receiver.start()
            
            self.connection_status.emit(True)
            self.is_mock_mode = False
            
            # Manter thread viva
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Erro no streaming UDP: {e}")
            print("Iniciando modo de simulação...")
            self.is_mock_mode = True
            self.connection_status.emit(True)
            
            # Modo simulação - gerar dados fake
            while self.is_running:
                try:
                    # Simular dados EEG (16 canais)
                    fake_data = np.random.randn(16) * 50 + np.sin(time.time() * 2 * np.pi * 0.5) * 20
                    self.data_received.emit(fake_data)
                    time.sleep(1/125)  # Simular 125 Hz
                except Exception as e:
                    print(f"Erro na simulação: {e}")
                    break
        
        finally:
            if self.udp_receiver:
                self.udp_receiver.stop()
            self.connection_status.emit(False)
    
    def extract_eeg_from_udp(self, data):
        """Extrai dados EEG do formato UDP"""
        try:
            # Se os dados são string, tentar converter para JSON
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return None
            
            # Se é um dicionário
            if isinstance(data, dict):
                # Formato timeSeriesRaw
                if 'type' in data and data['type'] == 'timeSeriesRaw' and 'data' in data:
                    timeseries = data['data']
                    if len(timeseries) >= 16:
                        # Processar todas as amostras (5 por canal)
                        all_samples = []
                        
                        # Determinar o número de amostras (assumindo que todos os canais têm o mesmo)
                        num_samples = len(timeseries[0]) if len(timeseries[0]) > 0 else 0
                        
                        # Para cada amostra temporal
                        for sample_idx in range(num_samples):
                            eeg_sample = []
                            for ch in range(16):
                                if sample_idx < len(timeseries[ch]):
                                    eeg_sample.append(timeseries[ch][sample_idx])
                                else:
                                    eeg_sample.append(0.0)
                            all_samples.append(np.array(eeg_sample))
                        
                        return all_samples  # Retorna lista de arrays
                
                # Formato direto por canais
                elif 'Ch1' in data:
                    eeg_sample = []
                    for ch in range(1, 17):
                        ch_key = f'Ch{ch}'
                        if ch_key in data:
                            value = data[ch_key]
                            if isinstance(value, list) and len(value) > 0:
                                eeg_sample.append(value[-1])
                            else:
                                eeg_sample.append(float(value) if value is not None else 0.0)
                        else:
                            eeg_sample.append(0.0)
                    return np.array(eeg_sample)
                
                # Formato com channels
                elif 'channels' in data:
                    return self.extract_eeg_from_udp(data['channels'])
            
            # Se é lista, assumir que são os 16 canais
            elif isinstance(data, list) and len(data) >= 16:
                return np.array(data[:16])
            
            return None
            
        except Exception as e:
            print(f"Erro ao extrair EEG: {e}")
            return None


class PatientRegistrationWidget(QWidget):
    """Widget para cadastro de pacientes"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        self.load_patients()
        
    def setup_ui(self):
        """Configura a interface"""
        layout = QVBoxLayout()
        
        # Formulário de cadastro
        form_group = QGroupBox("Cadastro de Novo Paciente")
        form_layout = QGridLayout()
        
        # Campos do formulário
        form_layout.addWidget(QLabel("Nome:"), 0, 0)
        self.name_edit = QLineEdit()
        form_layout.addWidget(self.name_edit, 0, 1)
        
        form_layout.addWidget(QLabel("Idade:"), 0, 2)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setValue(30)
        form_layout.addWidget(self.age_spin, 0, 3)
        
        form_layout.addWidget(QLabel("Sexo:"), 1, 0)
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Masculino", "Feminino", "Outro"])
        form_layout.addWidget(self.sex_combo, 1, 1)
        
        form_layout.addWidget(QLabel("Mão Afetada:"), 1, 2)
        self.hand_combo = QComboBox()
        self.hand_combo.addItems(["Esquerda", "Direita", "Ambas", "Nenhuma"])
        form_layout.addWidget(self.hand_combo, 1, 3)
        
        form_layout.addWidget(QLabel("Tempo desde evento (meses):"), 2, 0)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 1000)
        self.time_spin.setValue(0)
        form_layout.addWidget(self.time_spin, 2, 1)
        
        form_layout.addWidget(QLabel("Observações:"), 3, 0)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        form_layout.addWidget(self.notes_edit, 3, 1, 1, 3)
        
        # Botão de cadastro
        self.register_btn = QPushButton("Cadastrar Paciente")
        self.register_btn.clicked.connect(self.register_patient)
        form_layout.addWidget(self.register_btn, 4, 0, 1, 4)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Tabela de pacientes
        patients_group = QGroupBox("Pacientes Cadastrados")
        patients_layout = QVBoxLayout()
        
        self.patients_table = QTableWidget()
        self.patients_table.setColumnCount(7)
        self.patients_table.setHorizontalHeaderLabels([
            "ID", "Nome", "Idade", "Sexo", "Mão Afetada", "Tempo (meses)", "Data Cadastro"
        ])
        self.patients_table.setSelectionBehavior(QTableWidget.SelectRows)
        patients_layout.addWidget(self.patients_table)
        
        patients_group.setLayout(patients_layout)
        layout.addWidget(patients_group)
        
        self.setLayout(layout)
        
    def register_patient(self):
        """Registra um novo paciente"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Erro", "Nome é obrigatório!")
            return
            
        age = self.age_spin.value()
        sex = self.sex_combo.currentText()
        affected_hand = self.hand_combo.currentText()
        time_since_event = self.time_spin.value()
        notes = self.notes_edit.toPlainText()
        
        try:
            patient_id = self.db_manager.add_patient(
                name, age, sex, affected_hand, time_since_event, notes
            )
            
            QMessageBox.information(self, "Sucesso", 
                                  f"Paciente {name} cadastrado com ID {patient_id}")
            
            # Limpar formulário
            self.name_edit.clear()
            self.age_spin.setValue(30)
            self.sex_combo.setCurrentIndex(0)
            self.hand_combo.setCurrentIndex(0)
            self.time_spin.setValue(0)
            self.notes_edit.clear()
            
            # Recarregar tabela
            self.load_patients()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao cadastrar paciente: {e}")
    
    def load_patients(self):
        """Carrega a lista de pacientes"""
        try:
            patients = self.db_manager.get_all_patients()
            
            self.patients_table.setRowCount(len(patients))
            
            for row, patient in enumerate(patients):
                self.patients_table.setItem(row, 0, QTableWidgetItem(str(patient["id"])))
                self.patients_table.setItem(row, 1, QTableWidgetItem(patient["name"]))
                self.patients_table.setItem(row, 2, QTableWidgetItem(str(patient["age"])))
                self.patients_table.setItem(row, 3, QTableWidgetItem(patient["sex"]))
                self.patients_table.setItem(row, 4, QTableWidgetItem(patient["affected_hand"]))
                self.patients_table.setItem(row, 5, QTableWidgetItem(str(patient["time_since_event"])))
                self.patients_table.setItem(row, 6, QTableWidgetItem(patient["created_at"][:10]))
            
            self.patients_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")
    
    def get_selected_patient(self) -> Optional[int]:
        """Retorna o ID do paciente selecionado"""
        current_row = self.patients_table.currentRow()
        if current_row >= 0:
            return int(self.patients_table.item(current_row, 0).text())
        return None


class StreamingWidget(QWidget):
    """Widget para streaming e gravação de dados"""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.streaming_thread = None
        self.csv_logger = None
        self.is_recording = False
        self.current_patient_id = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface"""
        layout = QVBoxLayout()
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        # Conexão
        connection_group = QGroupBox("Conexão")
        connection_layout = QHBoxLayout()
        
        self.host_edit = QLineEdit("localhost")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(12345)
        
        self.connect_btn = QPushButton("Conectar")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        self.status_label = QLabel("Desconectado")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        connection_layout.addWidget(QLabel("Host:"))
        connection_layout.addWidget(self.host_edit)
        connection_layout.addWidget(QLabel("Porta:"))
        connection_layout.addWidget(self.port_spin)
        connection_layout.addWidget(self.connect_btn)
        connection_layout.addWidget(self.status_label)
        
        connection_group.setLayout(connection_layout)
        controls_layout.addWidget(connection_group)
        
        # Gravação
        recording_group = QGroupBox("Gravação")
        recording_layout = QVBoxLayout()
        
        # Primeira linha - seleção de paciente e controle de gravação
        recording_row1 = QHBoxLayout()
        
        self.patient_combo = QComboBox()
        self.refresh_patients_btn = QPushButton("Atualizar")
        self.refresh_patients_btn.clicked.connect(self.refresh_patients)
        
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        
        self.recording_label = QLabel("Não gravando")
        self.recording_label.setStyleSheet("color: gray;")
        
        recording_row1.addWidget(QLabel("Paciente:"))
        recording_row1.addWidget(self.patient_combo)
        recording_row1.addWidget(self.refresh_patients_btn)
        recording_row1.addWidget(self.record_btn)
        recording_row1.addWidget(self.recording_label)
        
        # Segunda linha - marcadores
        markers_group = QGroupBox("Marcadores")
        markers_layout = QHBoxLayout()
        
        # Botões de marcadores
        self.t1_btn = QPushButton("T1 - Movimento Real")
        self.t1_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.t1_btn.clicked.connect(lambda: self.add_marker("T1"))
        self.t1_btn.setEnabled(False)
        
        self.t2_btn = QPushButton("T2 - Movimento Imaginado")
        self.t2_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.t2_btn.clicked.connect(lambda: self.add_marker("T2"))
        self.t2_btn.setEnabled(False)
        
        self.baseline_btn = QPushButton("Baseline")
        self.baseline_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.baseline_btn.clicked.connect(self.start_baseline)
        self.baseline_btn.setEnabled(False)
        
        # Timer para baseline
        self.baseline_timer = QTimer()
        self.baseline_timer.timeout.connect(self.update_baseline_timer)
        self.baseline_time_remaining = 0
        self.baseline_label = QLabel("")
        
        markers_layout.addWidget(self.t1_btn)
        markers_layout.addWidget(self.t2_btn)
        markers_layout.addWidget(self.baseline_btn)
        markers_layout.addWidget(self.baseline_label)
        markers_layout.addStretch()
        
        markers_group.setLayout(markers_layout)
        
        recording_layout.addLayout(recording_row1)
        recording_layout.addWidget(markers_group)
        
        recording_group.setLayout(recording_layout)
        controls_layout.addWidget(recording_group)
        
        layout.addLayout(controls_layout)
        
        # Widget de plot
        self.plot_widget = EEGPlotWidget()
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
        
        # Inicializar lista de pacientes
        self.refresh_patients()
        
    def refresh_patients(self):
        """Atualiza a lista de pacientes"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecionar paciente...")
        
        try:
            patients = self.db_manager.get_all_patients()
            for patient in patients:
                self.patient_combo.addItem(
                    f"{patient['name']} (ID: {patient['id']})",
                    patient['id']
                )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")
    
    def toggle_connection(self):
        """Conecta/desconecta do streaming"""
        if self.streaming_thread is None or not self.streaming_thread.isRunning():
            # Conectar
            host = self.host_edit.text()
            port = self.port_spin.value()
            
            self.streaming_thread = StreamingThread()
            self.streaming_thread.data_received.connect(self.on_data_received)
            self.streaming_thread.connection_status.connect(self.on_connection_status)
            self.streaming_thread.start_streaming(host, port)
            
            self.connect_btn.setText("Desconectar")
            self.connect_btn.setEnabled(False)
            
        else:
            # Desconectar
            if self.is_recording:
                self.toggle_recording()
            
            self.streaming_thread.stop_streaming()
            self.connect_btn.setText("Conectar")
            self.record_btn.setEnabled(False)
    
    def toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Iniciar gravação
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Erro", "Selecione um paciente!")
                return
            
            self.current_patient_id = self.patient_combo.currentData()
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            # Gerar nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_{self.current_patient_id}_{patient_name}_{timestamp}.csv"
            full_path = get_recording_path(filename)
            
            try:
                # Usar nosso logger simples
                self.csv_logger = SimpleCSVLogger(str(full_path))
                self.csv_logger.start_logging()
                
                self.is_recording = True
                self.record_btn.setText("Parar Gravação")
                self.recording_label.setText(f"Gravando: {filename}")
                self.recording_label.setStyleSheet("color: red; font-weight: bold;")
                
                # Habilitar botões de marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                
                # Registrar gravação no banco
                self.db_manager.add_recording(self.current_patient_id, filename)
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
        else:
            # Parar gravação
            if self.csv_logger:
                self.csv_logger.stop_logging()
                self.csv_logger = None
            
            self.is_recording = False
            self.record_btn.setText("Iniciar Gravação")
            self.recording_label.setText("Não gravando")
            self.recording_label.setStyleSheet("color: gray;")
            
            # Desabilitar botões de marcadores
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_btn.setEnabled(False)
            
            # Parar timer de baseline se estiver rodando
            if self.baseline_timer.isActive():
                self.baseline_timer.stop()
                self.baseline_label.setText("")
            
            QMessageBox.information(self, "Sucesso", "Gravação finalizada!")
    
    def add_marker(self, marker_type):
        """Adiciona um marcador durante a gravação"""
        if self.is_recording and self.csv_logger:
            marker = self.csv_logger.add_marker(marker_type)
            
            # Feedback visual
            if marker_type == "T1":
                self.recording_label.setText(f"Gravando - Marcador T1 adicionado (T0 em 400 amostras)")
            elif marker_type == "T2":
                self.recording_label.setText(f"Gravando - Marcador T2 adicionado (T0 em 400 amostras)")
            
            # Resetar texto após 3 segundos
            QTimer.singleShot(3000, self.reset_recording_label)
    
    def start_baseline(self):
        """Inicia o período de baseline"""
        if self.is_recording and self.csv_logger:
            # Adicionar marcador de baseline
            self.csv_logger.add_marker("BASELINE")
            
            # Desabilitar outros botões por 5 minutos
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_btn.setEnabled(False)
            
            # Iniciar timer de 5 minutos (300 segundos)
            self.baseline_time_remaining = 300
            self.baseline_timer.start(1000)  # Atualizar a cada segundo
            
            # Feedback visual
            self.recording_label.setText("Gravando - Baseline iniciado")
    
    def update_baseline_timer(self):
        """Atualiza o timer de baseline"""
        if self.baseline_time_remaining > 0:
            minutes = self.baseline_time_remaining // 60
            seconds = self.baseline_time_remaining % 60
            self.baseline_label.setText(f"Baseline: {minutes:02d}:{seconds:02d}")
            self.baseline_time_remaining -= 1
        else:
            # Baseline terminado
            self.baseline_timer.stop()
            self.baseline_label.setText("")
            
            # Reabilitar botões se ainda estiver gravando
            if self.is_recording:
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                self.recording_label.setText("Gravando - Baseline finalizado")
                
                # Resetar texto após 3 segundos
                QTimer.singleShot(3000, self.reset_recording_label)
    
    def reset_recording_label(self):
        """Reseta o texto do label de gravação"""
        if self.is_recording:
            filename = self.csv_logger.filename if self.csv_logger else "arquivo.csv"
            self.recording_label.setText(f"Gravando: {filename}")
            self.recording_label.setStyleSheet("color: red; font-weight: bold;")
    
    def on_data_received(self, data):
        """Callback para dados recebidos"""
        # Enviar para plot
        self.plot_widget.add_data(data)
        
        # Enviar para logger se estiver gravando
        if self.is_recording and self.csv_logger:
            self.csv_logger.log_data(data)
    
    def on_connection_status(self, connected):
        """Callback para status da conexão"""
        if connected:
            if hasattr(self.streaming_thread, 'is_mock_mode') and self.streaming_thread.is_mock_mode:
                self.status_label.setText("Simulação (Dados Fake)")
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.status_label.setText("Conectado")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.record_btn.setEnabled(True)
        else:
            self.status_label.setText("Desconectado")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.record_btn.setEnabled(False)
        
        self.connect_btn.setEnabled(True)
        if connected:
            self.connect_btn.setText("Desconectar")
        else:
            self.connect_btn.setText("Conectar")


class BCIMainWindow(QMainWindow):
    """Janela principal da aplicação BCI"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface principal"""
        self.setWindowTitle("Sistema BCI - Interface de Controle")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central com abas
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("Sistema BCI - Interface de Controle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title_label)
        
        # Abas
        self.tabs = QTabWidget()
        
        # Aba de pacientes
        self.patient_widget = PatientRegistrationWidget(self.db_manager)
        self.tabs.addTab(self.patient_widget, "Cadastro de Pacientes")
        
        # Aba de streaming
        self.streaming_widget = StreamingWidget(self.db_manager)
        self.tabs.addTab(self.streaming_widget, "Streaming e Gravação")
        
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        
        # Barra de status
        self.statusBar().showMessage("Sistema BCI inicializado")
    
    def closeEvent(self, event):
        """Evento de fechamento da aplicação"""
        # Parar streaming se estiver rodando
        if hasattr(self.streaming_widget, 'streaming_thread') and \
           self.streaming_widget.streaming_thread is not None:
            self.streaming_widget.streaming_thread.stop_streaming()
        
        event.accept()


def main():
    """Função principal"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo moderno
    
    # Aplicar tema escuro (opcional)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin: 3px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """)
    
    window = BCIMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
