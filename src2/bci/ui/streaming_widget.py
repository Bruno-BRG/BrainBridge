import time
import json
import torch
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QSpinBox, QPushButton, QGroupBox, QMessageBox, QComboBox,
                           QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QColor
from bci.network import UDPReceiver
from bci.ui.plot_widget import EEGPlotWidget
from bci.data_logger import CSVDataLogger
from bci.config import get_recording_path
from bci.models.eegnet import EEGNet

class StreamingThread(QThread):
    data_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool)
    prediction_ready = pyqtSignal(float, float)  # prob_left, prob_right

    def __init__(self, host='localhost', port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.game_mode = False
        self.window_size = 400  # 3.2s @ 125Hz
        self.eeg_buffer = deque(maxlen=1000)
        
        # Inicialização do modelo
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = EEGNet().to(self.device)
            self.model.load_state_dict(torch.load('models/best_model.pth', map_location=self.device))
            self.model.eval()
        except:
            print("Erro ao carregar modelo. Modo jogo não estará disponível.")

    def run(self):
        self.running = True
        try:
            udp = UDPReceiver(self.host, self.port)
            udp.set_callback(lambda d:self._emit_data(d))
            udp.start()
            self.connection_status.emit(True)
            while self.running:
                time.sleep(0.1)
        except Exception:
            self.connection_status.emit(False)

    def stop(self):
        self.running = False

    def _emit_data(self, data):
        try:
            if isinstance(data, str): 
                data = json.loads(data)
            if isinstance(data, dict) and 'data' in data:
                series = data['data']
                for i in range(len(series[0])):
                    sample = np.array([ch[i] for ch in series])
                    self._process_sample(sample)
            elif isinstance(data, list) and len(data)>=16:
                self._process_sample(np.array(data[:16]))
        except:
            pass

    def _process_sample(self, sample):
        """Processa uma amostra de dados"""
        self.data_received.emit(sample)
        
        # Processamento para modo jogo
        if self.game_mode and self.model is not None:
            self.eeg_buffer.append(sample)
            if len(self.eeg_buffer) >= self.window_size:
                # Preparar dados para o modelo
                X = np.array(list(self.eeg_buffer))[-self.window_size:]
                X = X.reshape(1, 1, 16, self.window_size)
                X = torch.FloatTensor(X).to(self.device)
                
                # Fazer predição
                with torch.no_grad():
                    output = self.model(X)
                prob_left, prob_right = output[0].cpu().numpy()
                self.prediction_ready.emit(prob_left, prob_right)
                self.eeg_buffer.clear()

class StreamingWidget(QWidget):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.thread = None
        self.logger = None
        self.is_recording = False
        self.current_patient_id = None
        self.samples_since_marker = None
        self.baseline_active = False
        self.game_mode = False
        
        # Timer de sessão
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_session_timer)
        self.session_start_time = None
        self.session_elapsed_seconds = 0
        
        # Timer de baseline
        self.baseline_timer = QTimer()
        self.baseline_timer.timeout.connect(self._update_baseline_timer)
        self.baseline_remaining = 300  # 5 minutos
        
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        
        # Grupo de conexão UDP
        conn_group = QGroupBox("Conexão UDP")
        h = QHBoxLayout()
        self.host = QLineEdit("localhost")
        self.port = QSpinBox()
        self.port.setRange(1,65535)
        self.port.setValue(12345)
        self.btn = QPushButton("Conectar")
        self.btn.clicked.connect(self._toggle)
        self.status = QLabel("Desconectado")
        self.status.setStyleSheet("color:red")
        h.addWidget(QLabel("Host:"))
        h.addWidget(self.host)
        h.addWidget(QLabel("Porta:"))
        h.addWidget(self.port)
        h.addWidget(self.btn)
        h.addWidget(self.status)
        conn_group.setLayout(h)
        layout.addWidget(conn_group)

        # Grupo de gravação
        record_group = QGroupBox("Gravação")
        h = QHBoxLayout()
        self.patient_combo = QComboBox()
        self.patient_combo.addItem("Selecionar paciente...")
        self.refresh_btn = QPushButton("Atualizar")
        self.refresh_btn.clicked.connect(self._refresh_patients)
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        self.recording_label = QLabel("")
        h.addWidget(QLabel("Paciente:"))
        h.addWidget(self.patient_combo)
        h.addWidget(self.refresh_btn)
        h.addWidget(self.record_btn)
        h.addWidget(self.recording_label)
        record_group.setLayout(h)
        layout.addWidget(record_group)

        # Grupo de marcadores
        marker_group = QGroupBox("Marcadores")
        h = QHBoxLayout()
        self.t1_btn = QPushButton("T1 - Movimento Real")
        self.t1_btn.clicked.connect(lambda: self._add_marker("T1"))
        self.t1_btn.setEnabled(False)
        self.t2_btn = QPushButton("T2 - Movimento Imaginado")
        self.t2_btn.clicked.connect(lambda: self._add_marker("T2"))
        self.t2_btn.setEnabled(False)
        self.baseline_btn = QPushButton("Baseline")
        self.baseline_btn.clicked.connect(self._start_baseline)
        self.baseline_btn.setEnabled(False)
        self.baseline_label = QLabel("")
        h.addWidget(self.t1_btn)
        h.addWidget(self.t2_btn)
        h.addWidget(self.baseline_btn)
        h.addWidget(self.baseline_label)
        marker_group.setLayout(h)
        layout.addWidget(marker_group)

        # Grupo de jogo
        game_group = QGroupBox("Modo Jogo")
        h = QHBoxLayout()
        self.game_btn = QPushButton("Iniciar Jogo")
        self.game_btn.clicked.connect(self._toggle_game)
        self.game_btn.setEnabled(False)
        self.left_progress = QProgressBar()
        self.right_progress = QProgressBar()
        self.left_progress.setMaximum(100)
        self.right_progress.setMaximum(100)
        h.addWidget(self.game_btn)
        h.addWidget(QLabel("Esquerda:"))
        h.addWidget(self.left_progress)
        h.addWidget(QLabel("Direita:"))
        h.addWidget(self.right_progress)
        game_group.setLayout(h)
        layout.addWidget(game_group)

        # Timer de sessão
        time_group = QGroupBox("Tempo de Sessão")
        h = QHBoxLayout()
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        h.addWidget(self.time_label)
        time_group.setLayout(h)
        layout.addWidget(time_group)

        # Visualização EEG
        self.plot = EEGPlotWidget()
        layout.addWidget(self.plot)

        self.setLayout(layout)
        self._refresh_patients()

    def _toggle(self):
        if not self.thread or not self.thread.isRunning():
            self.thread = StreamingThread(self.host.text(), self.port.value())
            self.thread.data_received.connect(self._handle_data)
            self.thread.connection_status.connect(self._set_status)
            self.thread.prediction_ready.connect(self._update_prediction)
            self.thread.start()
            self.btn.setText("Desconectar")
            self.record_btn.setEnabled(True)
            # Habilitar modo jogo se modelo carregado
            if self.thread.model is not None:
                self.game_btn.setEnabled(True)
        else:
            if self.is_recording:
                self._toggle_recording()
            self.thread.stop()
            self.btn.setText("Conectar")
            self.record_btn.setEnabled(False)
            self.game_btn.setEnabled(False)

    def _set_status(self, ok):
        self.status.setText("Conectado" if ok else "Falha")
        self.status.setStyleSheet("color:green" if ok else "color:red")

    def _handle_data(self, data):
        """Processa dados recebidos"""
        self.last_sample = data
        self.plot.add_data(data)
        
        if self.is_recording:
            # Gravar dados
            if self.logger:
                self.logger.log_data(data)
            
            # Verificar se é hora de adicionar T0
            if self.samples_since_marker is not None:
                self.samples_since_marker += 1
                if self.samples_since_marker >= 400:  # 400 amostras após T1/T2
                    if self.logger:
                        self.logger.log_data(data, "T0")
                    self.samples_since_marker = None

    def _refresh_patients(self):
        """Atualiza a lista de pacientes"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecionar paciente...")
        
        try:
            patients = self.db.get_all_patients()
            for patient in patients:
                self.patient_combo.addItem(
                    f"{patient['name']} (ID: {patient['id']})",
                    patient['id']
                )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")

    def _toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Verificar paciente selecionado
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Aviso", "Selecione um paciente primeiro!")
                return
            
            self.current_patient_id = self.patient_combo.currentData()
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            try:
                # Criar arquivo de gravação
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"P{self.current_patient_id:03d}_{patient_name}_{timestamp}.csv"
                filepath = get_recording_path(filename)
                
                # Iniciar logger
                self.logger = CSVDataLogger(filepath, f"P{self.current_patient_id:03d}")
                self.logger.start_logging()
                
                # Atualizar UI
                self.is_recording = True
                self.record_btn.setText("Parar Gravação")
                self.recording_label.setText(f"Gravando: {filename}")
                self.recording_label.setStyleSheet("color: red;")
                
                # Habilitar marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                
                # Iniciar timer de sessão
                self.session_start_time = datetime.now()
                self.session_timer.start(1000)  # Atualizar a cada segundo
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
                return
            
        else:
            # Parar gravação
            if self.logger:
                self.logger.stop_logging()
                self.logger = None
            
            # Atualizar UI
            self.is_recording = False
            self.record_btn.setText("Iniciar Gravação")
            self.recording_label.setText("")
            self.recording_label.setStyleSheet("")
            
            # Desabilitar marcadores
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_btn.setEnabled(False)
            
            # Parar timer de sessão
            self.session_timer.stop()
            self.time_label.setText("00:00:00")

    def _add_marker(self, marker_type: str):
        """Adiciona um marcador T1/T2 e programa T0"""
        if self.logger and not self.baseline_active:
            self.logger.log_data(self.last_sample, marker_type)
            self.samples_since_marker = 0

    def _start_baseline(self):
        """Inicia período de baseline"""
        if not self.baseline_active:
            self.baseline_active = True
            self.baseline_remaining = 300  # 5 minutos
            self.baseline_timer.start(1000)  # Atualizar a cada segundo
            self.baseline_btn.setEnabled(False)
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_label.setText("Baseline: 05:00")
            self.baseline_label.setStyleSheet("color: purple; font-weight: bold;")
            if self.logger:
                self.logger.log_data(self.last_sample, "BASELINE")

    def _update_baseline_timer(self):
        """Atualiza timer de baseline"""
        self.baseline_remaining -= 1
        minutes = self.baseline_remaining // 60
        seconds = self.baseline_remaining % 60
        self.baseline_label.setText(f"Baseline: {minutes:02d}:{seconds:02d}")
        
        if self.baseline_remaining <= 0:
            self.baseline_timer.stop()
            self.baseline_active = False
            self.baseline_label.setText("")
            if self.is_recording:
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)

    def _update_session_timer(self):
        """Atualiza timer de sessão"""
        if self.session_start_time:
            elapsed = datetime.now() - self.session_start_time
            hours = elapsed.seconds // 3600
            minutes = (elapsed.seconds % 3600) // 60
            seconds = elapsed.seconds % 60
            self.time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def _toggle_game(self):
        """Alterna modo jogo"""
        if not self.game_mode:
            self.game_mode = True
            self.game_btn.setText("Parar Jogo")
            if self.thread:
                self.thread.game_mode = True
        else:
            self.game_mode = False
            self.game_btn.setText("Iniciar Jogo")
            if self.thread:
                self.thread.game_mode = False
            self.left_progress.setValue(0)
            self.right_progress.setValue(0)

    def _update_prediction(self, prob_left, prob_right):
        """Atualiza barras de probabilidade do modo jogo"""
        self.left_progress.setValue(int(prob_left * 100))
        self.right_progress.setValue(int(prob_right * 100))

class StreamingThread(QThread):
    data_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool)
    prediction_ready = pyqtSignal(float, float)  # prob_left, prob_right

    def __init__(self, host='localhost', port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.game_mode = False
        self.window_size = 400  # 3.2s @ 125Hz
        self.eeg_buffer = deque(maxlen=1000)
        
        # Inicialização do modelo
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = EEGNet().to(self.device)
            self.model.load_state_dict(torch.load('models/best_model.pth', map_location=self.device))
            self.model.eval()
        except:
            print("Erro ao carregar modelo. Modo jogo não estará disponível.")

    def run(self):
        self.running = True
        try:
            udp = UDPReceiver(self.host, self.port)
            udp.set_callback(lambda d:self._emit_data(d))
            udp.start()
            self.connection_status.emit(True)
            while self.running:
                time.sleep(0.1)
        except Exception:
            self.connection_status.emit(False)

    def stop(self):
        self.running = False

    def _emit_data(self, data):
        try:
            if isinstance(data, str): data = json.loads(data)
            if isinstance(data, dict) and 'data' in data:
                series = data['data']
                for i in range(len(series[0])):
                    sample = np.array([ch[i] for ch in series])
                    self.data_received.emit(sample)
            elif isinstance(data, list) and len(data)>=16:
                self.data_received.emit(np.array(data[:16]))
        except:
            pass

class StreamingWidget(QWidget):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.thread = None
        self.logger = None
        self.is_recording = False
        self.current_patient_id = None
        self.samples_since_marker = 0
        self.baseline_active = False
        
        # Timer de sessão
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_session_timer)
        self.session_start_time = None
        self.session_elapsed_seconds = 0
        
        # Timer de baseline
        self.baseline_timer = QTimer()
        self.baseline_timer.timeout.connect(self._update_baseline_timer)
        self.baseline_remaining = 300  # 5 minutos
        
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        
        # Grupo de conexão UDP
        conn_group = QGroupBox("Conexão UDP")
        h = QHBoxLayout()
        self.host = QLineEdit("localhost")
        self.port = QSpinBox()
        self.port.setRange(1,65535)
        self.port.setValue(12345)
        self.btn = QPushButton("Conectar")
        self.btn.clicked.connect(self._toggle)
        self.status = QLabel("Desconectado")
        self.status.setStyleSheet("color:red")
        h.addWidget(QLabel("Host:"))
        h.addWidget(self.host)
        h.addWidget(QLabel("Porta:"))
        h.addWidget(self.port)
        h.addWidget(self.btn)
        h.addWidget(self.status)
        conn_group.setLayout(h)
        layout.addWidget(conn_group)

        # Grupo de gravação
        record_group = QGroupBox("Gravação")
        h = QHBoxLayout()
        self.patient_combo = QComboBox()
        self.patient_combo.addItem("Selecionar paciente...")
        self.refresh_btn = QPushButton("Atualizar")
        self.refresh_btn.clicked.connect(self._refresh_patients)
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        self.recording_label = QLabel("")
        h.addWidget(QLabel("Paciente:"))
        h.addWidget(self.patient_combo)
        h.addWidget(self.refresh_btn)
        h.addWidget(self.record_btn)
        h.addWidget(self.recording_label)
        record_group.setLayout(h)
        layout.addWidget(record_group)

        # Grupo de marcadores
        marker_group = QGroupBox("Marcadores")
        h = QHBoxLayout()
        self.t1_btn = QPushButton("T1 - Movimento Real")
        self.t1_btn.clicked.connect(lambda: self._add_marker("T1"))
        self.t1_btn.setEnabled(False)
        self.t2_btn = QPushButton("T2 - Movimento Imaginado")
        self.t2_btn.clicked.connect(lambda: self._add_marker("T2"))
        self.t2_btn.setEnabled(False)
        self.baseline_btn = QPushButton("Baseline")
        self.baseline_btn.clicked.connect(self._start_baseline)
        self.baseline_btn.setEnabled(False)
        self.baseline_label = QLabel("")
        h.addWidget(self.t1_btn)
        h.addWidget(self.t2_btn)
        h.addWidget(self.baseline_btn)
        h.addWidget(self.baseline_label)
        marker_group.setLayout(h)
        layout.addWidget(marker_group)

        # Timer de sessão
        time_group = QGroupBox("Tempo de Sessão")
        h = QHBoxLayout()
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        h.addWidget(self.time_label)
        time_group.setLayout(h)
        layout.addWidget(time_group)

        # Visualização EEG
        self.plot = EEGPlotWidget()
        layout.addWidget(self.plot)

        self.setLayout(layout)
        self._refresh_patients()

    def _toggle(self):
        if not self.thread or not self.thread.isRunning():
            self.thread = StreamingThread(self.host.text(), self.port.value())
            self.thread.data_received.connect(self._handle_data)
            self.thread.connection_status.connect(self._set_status)
            self.thread.start()
            self.btn.setText("Desconectar")
            self.record_btn.setEnabled(True)
        else:
            if self.is_recording:
                self._toggle_recording()
            self.thread.stop()
            self.btn.setText("Conectar")
            self.record_btn.setEnabled(False)

    def _handle_data(self, data):
        """Processa dados recebidos"""
        self.last_sample = data
        self.plot.add_data(data)
        
        if self.is_recording:
            # Gravar dados
            if self.logger:
                self.logger.log_data(data)
            
            # Verificar se é hora de adicionar T0
            if self.samples_since_marker is not None:
                self.samples_since_marker += 1
                if self.samples_since_marker >= 400:  # 400 amostras após T1/T2
                    if self.logger:
                        self.logger.log_data(data, "T0")
                    self.samples_since_marker = None

    def _set_status(self, ok):
        self.status.setText("Conectado" if ok else "Falha")
        self.status.setStyleSheet("color:green" if ok else "color:red")

    def _refresh_patients(self):
        """Atualiza a lista de pacientes"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecionar paciente...")
        
        try:
            patients = self.db.get_all_patients()
            for patient in patients:
                self.patient_combo.addItem(
                    f"{patient['name']} (ID: {patient['id']})",
                    patient['id']
                )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")

    def _toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Verificar paciente selecionado
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Aviso", "Selecione um paciente primeiro!")
                return
            
            self.current_patient_id = self.patient_combo.currentData()
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            try:
                # Criar arquivo de gravação
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"P{self.current_patient_id:03d}_{patient_name}_{timestamp}.csv"
                filepath = get_recording_path(filename)
                
                # Iniciar logger
                self.logger = CSVDataLogger(filepath, f"P{self.current_patient_id:03d}")
                self.logger.start_logging()
                
                # Atualizar UI
                self.is_recording = True
                self.record_btn.setText("Parar Gravação")
                self.recording_label.setText(f"Gravando: {filename}")
                self.recording_label.setStyleSheet("color: red;")
                
                # Habilitar marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                
                # Iniciar timer de sessão
                self.session_start_time = datetime.now()
                self.session_timer.start(1000)  # Atualizar a cada segundo
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
                return
            
        else:
            # Parar gravação
            if self.logger:
                self.logger.stop_logging()
                self.logger = None
            
            # Atualizar UI
            self.is_recording = False
            self.record_btn.setText("Iniciar Gravação")
            self.recording_label.setText("")
            self.recording_label.setStyleSheet("")
            
            # Desabilitar marcadores
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_btn.setEnabled(False)
            
            # Parar timer de sessão
            self.session_timer.stop()
            self.time_label.setText("00:00:00")

    def _add_marker(self, marker_type: str):
        """Adiciona um marcador T1/T2 e programa T0"""
        if self.logger and not self.baseline_active:
            self.logger.log_data(self.last_sample, marker_type)
            self.samples_since_marker = 0

    def _start_baseline(self):
        """Inicia período de baseline"""
        if not self.baseline_active:
            self.baseline_active = True
            self.baseline_remaining = 300  # 5 minutos
            self.baseline_timer.start(1000)  # Atualizar a cada segundo
            self.baseline_btn.setEnabled(False)
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            self.baseline_label.setText("Baseline: 05:00")
            self.baseline_label.setStyleSheet("color: purple; font-weight: bold;")
            if self.logger:
                self.logger.log_data(self.last_sample, "BASELINE")

    def _update_baseline_timer(self):
        """Atualiza timer de baseline"""
        self.baseline_remaining -= 1
        minutes = self.baseline_remaining // 60
        seconds = self.baseline_remaining % 60
        self.baseline_label.setText(f"Baseline: {minutes:02d}:{seconds:02d}")
        
        if self.baseline_remaining <= 0:
            self.baseline_timer.stop()
            self.baseline_active = False
            self.baseline_label.setText("")
            if self.is_recording:
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)

    def _update_session_timer(self):
        """Atualiza timer de sessão"""
        if self.session_start_time:
            elapsed = datetime.now() - self.session_start_time
            hours = elapsed.seconds // 3600
            minutes = (elapsed.seconds % 3600) // 60
            seconds = elapsed.seconds % 60
            self.time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

