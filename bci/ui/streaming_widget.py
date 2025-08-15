import os
import threading
import time
import zmq
import socket
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np
import torch
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QGroupBox, QComboBox, QGridLayout,
                           QProgressBar, QTextEdit, QMessageBox, QCheckBox,
                           QLineEdit, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from ..database.database_manager import DatabaseManager
from ..streaming_logic.streaming_thread import StreamingThread
from ..configs.config import get_recording_path
from .EEG_plot_widget import EEGPlotWidget
from ..AI.EEGNet import EEGNet
from ..network.unity_communication import UDP_sender, UDP_receiver, UnityCommunicator

# Importar loggers
try:
    from ..network.openbci_csv_logger import OpenBCICSVLogger
    USE_OPENBCI_LOGGER = True
except ImportError:
    USE_OPENBCI_LOGGER = False

try:
    from ..network.simple_csv_logger import SimpleCSVLogger
except ImportError:
    SimpleCSVLogger = None

class StreamingWidget(QWidget):
    """Widget para streaming e gravação de dados"""
    
    # Signal para processar mensagens de acurácia de forma thread-safe
    accuracy_message_signal = pyqtSignal(str)
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.streaming_thread = None
        self.csv_logger = None
        self.is_recording = False
        self.current_patient_id = None
        self.pending_marker = None  # Para marcadores pendentes no logger OpenBCI
        self.baseline_timer = QTimer()  # Timer para baseline
        
        # Timer de sessão
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_start_time = None
        self.session_elapsed_seconds = 0
        
        self.setup_ui()
        
        # Inicialização do modelo
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = 400  # 3.2s @ 125Hz
        self.samples_since_last_prediction = 0
        self.predictions = deque(maxlen=50)  # Últimas predições
        self.eeg_buffer = deque(maxlen=1000)  # Buffer para dados EEG
        
        # Estados do servidor UDP
        self.udp_server_active = False
        self.game_mode = False
        self.game_mode = False  # Flag para modo jogo
        
        # Inicializar comunicador Unity
        self.unity_communicator = UnityCommunicator()
        self.unity_communicator.set_message_callback(self._on_unity_message)
        self.unity_communicator.set_connection_callback(self._on_unity_connection)
        
        # Contadores para marcadores
        self.t1_counter = 0
        self.t2_counter = 0
        
        # Timer para ações automáticas no jogo
        self.game_action_timer = QTimer()
        self.game_action_timer.timeout.connect(self.game_random_action)
        
        # Controle para aguardar resposta antes do próximo sinal
        self.waiting_for_response = False
        self.response_received = False
        
        # Controle de janela de tempo para IA (5 segundos de previsão permitida)
        self.ai_prediction_enabled = False
        self.task_start_time = None
        self.ai_window_duration = 5000  # 5 segundos em ms
        
        # Variáveis para cálculo de acurácia
        self.accuracy_data = []  # Lista de tuplas (cor_esperada, trigger_real)
        self.accuracy_correct = 0
        self.accuracy_total = 0
        
        # UDP receiver para acurácia (recebe mensagens do sistema externo)
        self.accuracy_udp_receiver = None
        self.accuracy_thread = None
        
        # Conectar signal para processar mensagens de acurácia
        self.accuracy_message_signal.connect(self.process_accuracy_message)
        
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
        
        # Servidor UDP
        udp_group = QGroupBox("Servidor UDP")
        udp_layout = QVBoxLayout()
        
        # Primeira linha - status e controle do servidor
        udp_row1 = QHBoxLayout()
        
        self.udp_status_label = QLabel("Servidor UDP: Desligado")
        self.udp_status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.udp_toggle_btn = QPushButton("Iniciar Servidor UDP")
        self.udp_toggle_btn.clicked.connect(self.toggle_udp_server)
        self.udp_toggle_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        # Checkbox para habilitar/desabilitar envio automático
        self.udp_auto_send_checkbox = QCheckBox("Envio Automático")
        self.udp_auto_send_checkbox.setChecked(True)
        self.udp_auto_send_checkbox.setToolTip("Quando marcado, envia sinais UDP automaticamente durante as predições")
        
        udp_row1.addWidget(self.udp_status_label)
        udp_row1.addWidget(self.udp_toggle_btn)
        udp_row1.addWidget(self.udp_auto_send_checkbox)
        udp_row1.addStretch()
        
        # Segunda linha - testes manuais
        udp_row2 = QHBoxLayout()
        
        udp_test_label = QLabel("Teste Manual:")
        self.udp_test_left_btn = QPushButton("🤚 Mão Esquerda")
        self.udp_test_left_btn.clicked.connect(lambda: self.manual_udp_test('esquerda'))
        self.udp_test_left_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.udp_test_left_btn.setEnabled(False)
        
        self.udp_test_right_btn = QPushButton("✋ Mão Direita")
        self.udp_test_right_btn.clicked.connect(lambda: self.manual_udp_test('direita'))
        self.udp_test_right_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.udp_test_right_btn.setEnabled(False)
        
        udp_row2.addWidget(udp_test_label)
        udp_row2.addWidget(self.udp_test_left_btn)
        udp_row2.addWidget(self.udp_test_right_btn)
        udp_row2.addStretch()
        
        udp_layout.addLayout(udp_row1)
        udp_layout.addLayout(udp_row2)
        
        udp_group.setLayout(udp_layout)
        controls_layout.addWidget(udp_group)
        
        # Gravação
        recording_group = QGroupBox("Gravação")
        recording_layout = QVBoxLayout()
        
        # Primeira linha - seleção de paciente e tarefa
        recording_row1 = QHBoxLayout()
        
        self.patient_combo = QComboBox()
        self.refresh_patients_btn = QPushButton("Atualizar")
        self.refresh_patients_btn.clicked.connect(self.refresh_patients)
        
        # Dropdown para seleção de tarefa
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Baseline", "Treino", "Teste", "Jogo"])
        self.task_combo.setCurrentIndex(0)  # Baseline como padrão
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        
        recording_row1.addWidget(QLabel("Paciente:"))
        recording_row1.addWidget(self.patient_combo)
        recording_row1.addWidget(self.refresh_patients_btn)
        recording_row1.addWidget(QLabel("Tarefa:"))
        recording_row1.addWidget(self.task_combo)
        
        # Segunda linha - controle de gravação
        recording_row2 = QHBoxLayout()
        
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        
        self.recording_label = QLabel("Não gravando")
        self.recording_label.setStyleSheet("color: gray;")
        
        # Label do timer de sessão
        self.session_timer_label = QLabel("Tempo: 00:00:00")
        self.session_timer_label.setStyleSheet("color: gray; font-weight: bold;")
        
        recording_row2.addWidget(self.record_btn)
        recording_row2.addWidget(self.recording_label)
        recording_row2.addWidget(self.session_timer_label)
        
        recording_layout.addLayout(recording_row1)
        recording_layout.addLayout(recording_row2)
        
        # Segunda linha - marcadores
        markers_group = QGroupBox("Marcadores")
        markers_layout = QVBoxLayout()
        
        # Primeira linha - botões de marcadores
        buttons_row = QHBoxLayout()
        
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
        
        buttons_row.addWidget(self.t1_btn)
        buttons_row.addWidget(self.t2_btn)
        buttons_row.addWidget(self.baseline_btn)
        buttons_row.addWidget(self.baseline_label)
        buttons_row.addStretch()
        
        # Segunda linha - contadores
        counters_row = QHBoxLayout()
        
        self.t1_counter_label = QLabel("T1: 0")
        self.t1_counter_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 14px;")
        
        self.t2_counter_label = QLabel("T2: 0")
        self.t2_counter_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 14px;")
        
        counters_row.addWidget(self.t1_counter_label)
        counters_row.addWidget(self.t2_counter_label)
        counters_row.addStretch()
        
        markers_layout.addLayout(buttons_row)
        markers_layout.addLayout(counters_row)
        
        markers_group.setLayout(markers_layout)
        
        recording_layout.addLayout(recording_row1)
        recording_layout.addWidget(markers_group)
        
        recording_group.setLayout(recording_layout)
        controls_layout.addWidget(recording_group)
        
        layout.addLayout(controls_layout)
        
        # Widget de plot
        self.plot_widget = EEGPlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Timer de sessão
        self.session_label = QLabel("Sessão: 00:00")
        self.session_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.session_label)
        
        # Game feedback group
        game_group = QGroupBox("Predições do Jogo")
        game_group.setVisible(False)  # Inicialmente oculto
        self.game_group = game_group
        game_layout = QVBoxLayout()

        # Label para mostrar predição atual
        self.prediction_label = QLabel("Aguardando predição...")
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        game_layout.addWidget(self.prediction_label)

        # Labels para probabilidades
        self.prob_left_label = QLabel("Mão Esquerda: 0%")
        self.prob_right_label = QLabel("Mão Direita: 0%")
        self.prob_left_label.setStyleSheet("color: #2196F3;")
        self.prob_right_label.setStyleSheet("color: #FF9800;")
        game_layout.addWidget(self.prob_left_label)
        game_layout.addWidget(self.prob_right_label)
        
        # Label para status da janela de IA
        self.ai_status_label = QLabel("🤖 IA: Aguardando tarefa")
        self.ai_status_label.setStyleSheet("color: gray; font-weight: bold; font-size: 12px;")
        self.ai_status_label.setAlignment(Qt.AlignCenter)
        game_layout.addWidget(self.ai_status_label)

        game_group.setLayout(game_layout)
        layout.addWidget(game_group)
        
        # Accuracy group (só aparece no modo jogo)
        accuracy_group = QGroupBox("Acurácia do Modelo")
        accuracy_group.setVisible(False)  # Inicialmente oculto
        self.accuracy_group = accuracy_group
        accuracy_layout = QHBoxLayout()
        
        # Label principal de acurácia
        self.accuracy_label = QLabel("Acurácia: 0% (0/0)")
        self.accuracy_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")
        
        # Labels para detalhes
        self.accuracy_details_label = QLabel("Esperado vs Real")
        self.accuracy_details_label.setStyleSheet("color: gray; font-size: 12px;")
        
        accuracy_layout.addWidget(self.accuracy_label)
        accuracy_layout.addWidget(self.accuracy_details_label)
        accuracy_layout.addStretch()
        
        accuracy_group.setLayout(accuracy_layout)
        layout.addWidget(accuracy_group)
        
        # Stats group
        stats_group = QGroupBox("Estatísticas do Jogo")
        stats_group.setVisible(False)  # Inicialmente oculto
        self.stats_group = stats_group
        stats_layout = QGridLayout()

        self.total_predictions_label = QLabel("Total de predições: 0")
        self.left_predictions_label = QLabel("Mão esquerda: 0")
        self.right_predictions_label = QLabel("Mão direita: 0")
        self.transitions_label = QLabel("Transições: 0")
        self.confidence_label = QLabel("Confiança média: 0%")

        stats_layout.addWidget(self.total_predictions_label, 0, 0)
        stats_layout.addWidget(self.left_predictions_label, 1, 0)
        stats_layout.addWidget(self.right_predictions_label, 1, 1)
        stats_layout.addWidget(self.transitions_label, 2, 0)
        stats_layout.addWidget(self.confidence_label, 2, 1)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Timer para atualizar estatísticas
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_game_stats)
        self.stats_timer.start(1000)  # Atualizar a cada segundo
        
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
    
    def toggle_udp_server(self):
        """Liga/desliga o servidor UDP"""
        if not self.udp_server_active:
            # Iniciar servidor UDP
            try:
                if self.unity_communicator.start_server():
                    self.udp_server_active = True
                    self.udp_status_label.setText("Servidor UDP: Ligado")
                    self.udp_status_label.setStyleSheet("color: green; font-weight: bold;")
                    self.udp_toggle_btn.setText("Parar Servidor UDP")
                    self.udp_toggle_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
                    
                    # Habilitar botões de teste
                    self.udp_test_left_btn.setEnabled(True)
                    self.udp_test_right_btn.setEnabled(True)
                    
                    QMessageBox.information(self, "Sucesso", "Servidor UDP iniciado com sucesso!\nBroadcast do IP enviado automaticamente.")
                else:
                    QMessageBox.critical(self, "Erro", "Falha ao iniciar servidor UDP")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar servidor UDP: {e}")
        else:
            # Parar servidor UDP
            try:
                self.unity_communicator.stop_server()
                self.udp_server_active = False
                self.udp_status_label.setText("Servidor UDP: Desligado")
                self.udp_status_label.setStyleSheet("color: red; font-weight: bold;")
                self.udp_toggle_btn.setText("Iniciar Servidor UDP")
                self.udp_toggle_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                
                # Desabilitar botões de teste
                self.udp_test_left_btn.setEnabled(False)
                self.udp_test_right_btn.setEnabled(False)
                
                QMessageBox.information(self, "Sucesso", "Servidor UDP parado com sucesso!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao parar servidor UDP: {e}")
    
    def manual_udp_test(self, direction):
        """Teste manual do envio UDP"""
        if self.udp_server_active:
            success = UDP_sender.enviar_sinal(direction)
            if success:
                side_text = "esquerda" if direction == 'esquerda' else "direita"
                QMessageBox.information(self, "Teste UDP", f"Sinal enviado: Mão {side_text}")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao enviar sinal UDP!")
        else:
            QMessageBox.warning(self, "Aviso", "Servidor UDP não está ativo!")
    
    def send_udp_signal(self, direction):
        """Envia sinal UDP se o servidor estiver ativo e o envio automático estiver habilitado"""
        if self.udp_server_active and self.udp_auto_send_checkbox.isChecked():
            success = UDP_sender.enviar_sinal(direction)
            if not success:
                print(f"Falha ao enviar sinal UDP para {direction}")
            return success
        return False
    
    def toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Iniciar gravação
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Erro", "Selecione um paciente!")
                return
            
            self.current_patient_id = self.patient_combo.currentData()
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            # Obter tarefa do dropdown
            task = self.task_combo.currentText().lower().replace(" ", "_")  # ex: "Baseline" -> "baseline"
            
            # Verificar se é modo jogo
            self.game_mode = (task == "jogo")
            if self.game_mode:
                if self.model is None:
                    if not self.load_model():
                        return
                # Limpar variáveis do jogo
                self.predictions.clear()
                self.eeg_buffer.clear()
                self.samples_since_last_prediction = 0
                self.prediction_label.setText("Aguardando predição...")
                self.prob_left_label.setText("Mão Esquerda: 0%")
                self.prob_right_label.setText("Mão Direita: 0%")
                
                # Resetar dados de acurácia
                self.reset_accuracy_data()
                
                # Resetar controle de resposta
                self.waiting_for_response = False
                self.response_received = False
                
                # Resetar controle de IA
                self.ai_prediction_enabled = False
                self.task_start_time = None
                
                # Resetar status visual da IA
                self.ai_status_label.setText("🤖 IA: Aguardando tarefa")
                self.ai_status_label.setStyleSheet("color: gray; font-weight: bold; font-size: 12px;")
                
                # Resetar contadores de ações no início da gravação
                self.reset_action_counters()
                
                # Iniciar UDP receiver para acurácia - agora sempre disponível
                try:
                    self.start_accuracy_udp_receiver()
                except Exception as e:
                    print(f"Erro ao iniciar UDP receiver de acurácia: {e}")
                
                # Iniciar primeiro sinal aleatório imediatamente (não usar timer automático)
                # O próximo sinal será enviado apenas após receber CORRECT/WRONG
                QTimer.singleShot(1000, self.send_next_random_signal)  # Aguardar 1 segundo para inicializar
                
                # Manter timer como fallback caso não receba resposta (a cada 30 segundos)
                self.game_action_timer.start(30000)
            
            try:
                # Usar logger OpenBCI se disponível
                if USE_OPENBCI_LOGGER:
                    self.csv_logger = OpenBCICSVLogger(
                        patient_id=f"P{self.current_patient_id:03d}",
                        task=task,
                        patient_name=patient_name,  # Adicionar nome do paciente
                        base_path=os.path.dirname(get_recording_path(""))
                    )
                    filename = self.csv_logger.filename
                    # Mostrar caminho relativo para feedback visual
                    display_path = f"{self.csv_logger.patient_folder}/{filename}"
                else:
                    # Fallback para logger simples
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"patient_{self.current_patient_id}_{patient_name}_{timestamp}.csv"
                    full_path = get_recording_path(filename)
                    self.csv_logger = SimpleCSVLogger(str(full_path))
                    self.csv_logger.start_logging()
                    display_path = filename
                
                self.is_recording = True
                self.update_record_button_text()  # Usar método que considera a tarefa
                self.recording_label.setText(f"Gravando: {display_path}")
                self.recording_label.setStyleSheet("color: red; font-weight: bold;")
                
                # Habilitar botões de marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                
                # Resetar contadores
                self.reset_action_counters()
                
                # Registrar gravação no banco
                recording_path = display_path if USE_OPENBCI_LOGGER else filename
                self.db_manager.add_recording(self.current_patient_id, recording_path, task)
                
                # Iniciar timer de sessão
                self.session_start_time = time.time()
                self.session_timer.start(1000)  # Atualizar a cada segundo
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
        else:
            # Parar gravação
            if self.csv_logger:
                self.csv_logger.stop_logging()
                self.csv_logger = None
            
            self.is_recording = False
            self.game_mode = False  # Desativar modo jogo
            
            # Parar UDP receiver de acurácia
            self.stop_accuracy_udp_receiver()
            
            # Parar timer de ações automáticas no jogo
            if self.game_action_timer.isActive():
                self.game_action_timer.stop()
            
            # Resetar controle de resposta
            self.waiting_for_response = False
            self.response_received = False
            
            # Resetar controle de IA
            self.ai_prediction_enabled = False
            self.task_start_time = None
            
            # Resetar status visual da IA
            self.ai_status_label.setText("🤖 IA: Parada")
            self.ai_status_label.setStyleSheet("color: gray; font-weight: bold; font-size: 12px;")
            
            # Resetar contadores de ações
            self.reset_action_counters()
                
            self.update_record_button_text()  # Usar método que considera a tarefa
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
            
            # Parar timer de sessão
            self.session_timer.stop()
            self.session_elapsed_seconds = 0
            self.update_session_timer()
            
            QMessageBox.information(self, "Sucesso", "Gravação finalizada!")
    

    def game_random_action(self):
        """Executa uma ação aleatória no jogo (fallback caso não receba resposta)"""
        if self.is_recording and self.csv_logger:
            # Verificar se não está aguardando resposta
            if self.waiting_for_response:
                print("⚠️  Timeout: Não recebeu resposta CORRECT/WRONG, enviando sinal de fallback")
                # Resetar estado e enviar novo sinal
                self.waiting_for_response = False
                self.response_received = False
                
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True
            self.response_received = False
            
            # Abrir janela de IA por 5 segundos (fallback)
            self.ai_prediction_enabled = True
            self.task_start_time = time.time() * 1000  # timestamp em ms
            print(f"🤖 Janela de IA aberta por {self.ai_window_duration/1000}s (fallback)")
            
            # Atualizar status visual
            self.ai_status_label.setText("🟡 IA: Ativa (fallback)")
            self.ai_status_label.setStyleSheet("color: orange; font-weight: bold; font-size: 12px;")
            
            # Fechar automaticamente a janela após 5 segundos
            QTimer.singleShot(self.ai_window_duration, self.close_ai_window)
            
            self.add_marker(action)

    def send_next_random_signal(self):
        """Envia o próximo sinal aleatório após receber resposta"""
        if self.is_recording and self.csv_logger:
            print("🎲 Enviando próximo sinal aleatório")
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True
            self.response_received = False
            
            # Abrir janela de IA por 5 segundos
            self.ai_prediction_enabled = True
            self.task_start_time = time.time() * 1000  # timestamp em ms
            print(f"🤖 Janela de IA aberta por {self.ai_window_duration/1000}s")
            
            # Atualizar status visual
            self.ai_status_label.setText("🟢 IA: Ativa (5s)")
            self.ai_status_label.setStyleSheet("color: green; font-weight: bold; font-size: 12px;")
            
            # Fechar automaticamente a janela após 5 segundos
            QTimer.singleShot(self.ai_window_duration, self.close_ai_window)
            
            self.add_marker(action)

    def close_ai_window(self):
        """Fecha a janela de IA após 5 segundos"""
        self.ai_prediction_enabled = False
        print("🚫 Janela de IA fechada automaticamente")
        
        # Atualizar status visual
        self.ai_status_label.setText("🔴 IA: Inativa")
        self.ai_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")

    def add_marker(self, marker_type):
        """Adiciona um marcador durante a gravação"""
        if self.is_recording and self.csv_logger:
            # Incrementar contador
            if marker_type == "T1":
                self.t1_counter += 1
                self.t1_counter_label.setText(f"T1: {self.t1_counter}")
                
                # Enviar trigger apenas nos modos Teste e Treino
                current_task = self.task_combo.currentText()
                if current_task in ["Teste", "Treino", "Jogo"] and self.udp_server_active:
                    UDP_sender.enviar_sinal('trigger_left')  # Enviar sinal para ativar trigger esquerdo
                    
            elif marker_type == "T2":
                self.t2_counter += 1
                self.t2_counter_label.setText(f"T2: {self.t2_counter}")
                
                # Enviar trigger apenas nos modos Teste e Treino
                current_task = self.task_combo.currentText()
                if current_task in ["Teste", "Treino", "Jogo"] and self.udp_server_active:
                    UDP_sender.enviar_sinal('trigger_right')  # Enviar sinal para ativar trigger direito

            if USE_OPENBCI_LOGGER:
                # Para o logger OpenBCI, verificar se baseline está ativo
                if hasattr(self.csv_logger, 'is_baseline_active'):
                    if self.csv_logger.is_baseline_active():
                        QMessageBox.warning(self, "Baseline Ativo", 
                                          "Não é possível adicionar marcadores durante o baseline")
                        return
                # Marcar para adicionar na próxima amostra
                self.pending_marker = marker_type
            else:
                # Logger simples
                marker = self.csv_logger.add_marker(marker_type)
            
            # Feedback visual
            task_name = "jogo" if self.task_combo.currentText() == "Jogo" else "gravação"
            if marker_type == "T1":
                self.recording_label.setText(f"{'Jogando' if task_name == 'jogo' else 'Gravando'} - Marcador T1 adicionado (T0 em 400 amostras)")
            elif marker_type == "T2":
                self.recording_label.setText(f"{'Jogando' if task_name == 'jogo' else 'Gravando'} - Marcador T2 adicionado (T0 em 400 amostras)")
            
            # Resetar texto após 3 segundos
            QTimer.singleShot(3000, self.reset_recording_label)
    
    def start_baseline(self):
        """Inicia o período de baseline"""
        if self.is_recording and self.csv_logger:
            if USE_OPENBCI_LOGGER:
                # Logger OpenBCI
                if hasattr(self.csv_logger, 'start_baseline'):
                    self.csv_logger.start_baseline()
                    
                    # Iniciar timer visual
                    self.baseline_timer.timeout.connect(self.update_baseline_timer)
                    self.baseline_timer.start(1000)
                else:
                    # Fallback
                    self.csv_logger.add_marker("BASELINE")
            else:
                # Logger simples
                self.csv_logger.add_marker("BASELINE")
            
            # Desabilitar outros botões por 5 minutos
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False) 
            self.baseline_btn.setEnabled(False)
            
            # Feedback visual
            task_name = "jogo" if self.task_combo.currentText() == "Jogo" else "gravação"
            status_text = "Jogando" if task_name == "jogo" else "Gravando"
            self.recording_label.setText(f"{status_text} - Baseline iniciado")
    
    def update_baseline_timer(self):
        """Atualiza o timer de baseline"""
        if USE_OPENBCI_LOGGER and hasattr(self.csv_logger, 'get_baseline_remaining'):
            remaining = self.csv_logger.get_baseline_remaining()
            
            if remaining <= 0:
                # Baseline finalizado
                self.baseline_timer.stop()
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                self.baseline_btn.setEnabled(True)
                
                task_name = "jogo" if self.task_combo.currentText() == "Jogo" else "gravação"
                status_text = "Jogando" if task_name == "jogo" else "Gravando"
                self.recording_label.setText(f"{status_text} - Baseline finalizado")
                QMessageBox.information(self, "Baseline", "Período de baseline finalizado!")
            else:
                # Atualizar display
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                task_name = "jogo" if self.task_combo.currentText() == "Jogo" else "gravação"
                status_text = "Jogando" if task_name == "jogo" else "Gravando"
                self.recording_label.setText(f"{status_text} - Baseline: {minutes:02d}:{seconds:02d}")
        else:
            # Fallback para timer simples
            if hasattr(self, 'baseline_time_remaining'):
                self.baseline_time_remaining -= 1
                if self.baseline_time_remaining <= 0:
                    self.baseline_timer.stop()
                    self.t1_btn.setEnabled(True)
                    self.t2_btn.setEnabled(True)
                    self.baseline_btn.setEnabled(True)
                    self.recording_label.setText("Gravando - Baseline finalizado")
                else:
                    minutes = self.baseline_time_remaining // 60
                    seconds = self.baseline_time_remaining % 60
                    self.recording_label.setText(f"Gravando - Baseline: {minutes:02d}:{seconds:02d}")
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
            task_name = "jogo" if self.task_combo.currentText() == "Jogo" else "gravação"
            status_text = "Jogando" if task_name == "jogo" else "Gravando"
            
            if self.csv_logger:
                if USE_OPENBCI_LOGGER and hasattr(self.csv_logger, 'patient_folder'):
                    display_path = f"{self.csv_logger.patient_folder}/{self.csv_logger.filename}"
                else:
                    display_path = self.csv_logger.filename
            else:
                display_path = "arquivo.csv"
            self.recording_label.setText(f"{status_text}: {display_path}")
            self.recording_label.setStyleSheet("color: red; font-weight: bold;")
    
    def load_model(self):
        """Carrega modelo CNN para inferência"""
        try:
            model_path = "bci/models/best_model.pth"
            possible_paths = [
            model_path,
            f"../{model_path}",  
            os.path.join(os.getcwd(), model_path)
            ]
            
            model_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    model_found = True
                    break
            
            if not model_found:
                QMessageBox.warning(self, "Erro", "Modelo não encontrado!")
                return False
            
            self.model = EEGNet(n_channels=16, n_classes=2, n_samples=self.window_size)
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Carregar pesos
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
                
            self.model.to(self.device).eval()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar modelo: {e}")
            return False
            
    def update_game_stats(self):
        """Atualiza estatísticas do jogo"""
        if not self.game_mode or not self.predictions:
            return
            
        # Calcular estatísticas
        total_preds = len(self.predictions)
        left_count = sum(1 for _, pred, _ in self.predictions if pred == 0)
        right_count = sum(1 for _, pred, _ in self.predictions if pred == 1)
        
        # Contar transições
        transitions = 0
        for i in range(1, len(self.predictions)):
            if self.predictions[i][1] != self.predictions[i-1][1]:
                transitions += 1
                
        # Calcular confiança média
        avg_conf = np.mean([conf for _, _, conf in self.predictions]) * 100
        
        # Atualizar labels
        self.total_predictions_label.setText(f"Total de predições: {total_preds}")
        self.left_predictions_label.setText(f"Mão esquerda: {left_count}")
        self.right_predictions_label.setText(f"Mão direita: {right_count}")
        self.transitions_label.setText(f"Transições: {transitions}")
        self.confidence_label.setText(f"Confiança média: {avg_conf:.1f}%")
        
    def process_accuracy_message(self, message):
        """Processa mensagem UDP recebida para cálculo de acurácia"""
        print(f"🔍 DEBUG: Mensagem recebida para acurácia: '{message}'")
        
        if not self.game_mode:
            print("🔍 DEBUG: Ignorando mensagem - não está em modo jogo")
            return
            
        try:
            # Parse da mensagem: "RED_FLOWER,TRIGGER_ACTION_LEFT"
            if "," in message:
                parts = message.strip().split(",")
                if len(parts) == 2:
                    flower_color = parts[0].strip()
                    trigger_action = parts[1].strip()
                    
                    # Mapear cor para ação esperada
                    if flower_color == "RED_FLOWER":
                        expected_action = "LEFT"  # Vermelho = esquerda esperada
                    elif flower_color == "BLUE_FLOWER":
                        expected_action = "RIGHT"  # Azul = direita esperada
                    else:
                        print(f"Cor de flor desconhecida: {flower_color}")
                        return
                    
                    # Mapear trigger para ação real
                    if trigger_action == "TRIGGER_ACTION_LEFT":
                        real_action = "LEFT"
                    elif trigger_action == "TRIGGER_ACTION_RIGHT":
                        real_action = "RIGHT"
                    else:
                        print(f"Trigger desconhecido: {trigger_action}")
                        return
                    
                    # Calcular se foi acerto
                    is_correct = (expected_action == real_action)
                    
                    # Atualizar contadores
                    self.accuracy_total += 1
                    if is_correct:
                        self.accuracy_correct += 1
                    
                    # Armazenar dados
                    self.accuracy_data.append((expected_action, real_action, is_correct))
                    
                    # Atualizar interface
                    self.update_accuracy_display()
                    
                    # Log para debug
                    status = "✓" if is_correct else "✗"
                    print(f"Acurácia: {flower_color} -> {expected_action} vs {trigger_action} -> {real_action} {status}")
                    
        except Exception as e:
            print(f"Erro ao processar mensagem de acurácia: {e}")
            
    def update_accuracy_display(self):
        """Atualiza a interface de acurácia"""
        if self.accuracy_total == 0:
            accuracy_percent = 0
        else:
            accuracy_percent = (self.accuracy_correct / self.accuracy_total) * 100
            
        # Atualizar label principal
        self.accuracy_label.setText(f"Acurácia: {accuracy_percent:.1f}% ({self.accuracy_correct}/{self.accuracy_total})")
        
        # Atualizar detalhes
        if self.accuracy_data:
            last_trial = self.accuracy_data[-1]
            expected, real, correct = last_trial
            status = "✓ Correto" if correct else "✗ Erro"
            self.accuracy_details_label.setText(f"Último: {expected} vs {real} - {status}")
        
        # Atualizar cor baseada na acurácia
        if accuracy_percent >= 80:
            color = "#4CAF50"  # Verde
        elif accuracy_percent >= 60:
            color = "#FF9800"  # Laranja
        else:
            color = "#f44336"  # Vermelho
            
        self.accuracy_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        
    def reset_accuracy_data(self):
        """Reseta os dados de acurácia"""
        self.accuracy_data.clear()
        self.accuracy_correct = 0
        self.accuracy_total = 0
        self.accuracy_label.setText("Acurácia: 0% (0/0)")
        self.accuracy_details_label.setText("Esperado vs Real")
    
    def reset_action_counters(self):
        """Reseta os contadores de ações T1 e T2"""
        self.t1_counter = 0
        self.t2_counter = 0
        self.t1_counter_label.setText("T1: 0")
        self.t2_counter_label.setText("T2: 0")
        print("🔄 Contadores de ações resetados")
        
    def start_accuracy_udp_receiver(self):
        """
        Inicia o receptor de acurácia.
        Agora usa o sistema de callbacks do UnityCommunicator.
        """
        print("✅ Sistema de acurácia ativo - usando callbacks do UnityCommunicator")
        # O receptor de mensagens já está ativo através dos callbacks do unity_communicator
        # As mensagens serão processadas automaticamente via _on_unity_message()
        
    def stop_accuracy_udp_receiver(self):
        """Para o UDP receiver de acurácia"""
        print("Sistema de acurácia parado - callbacks mantidos ativos")
        
    def predict_movement(self, eeg_data):
        """Faz predição do movimento com o modelo CNN"""
        if not self.game_mode or self.model is None:
            return
        
        # Verificar se a IA pode fazer previsões (janela de 5 segundos)
        if not self.ai_prediction_enabled:
            return
            
        # Verificar se ainda está dentro da janela de tempo permitida
        if self.task_start_time is not None:
            elapsed_time = time.time() * 1000 - self.task_start_time  # em ms
            if elapsed_time > self.ai_window_duration:
                self.ai_prediction_enabled = False
                print(f"🚫 Janela de IA fechada após {self.ai_window_duration/1000}s")
                return
            
        try:
            # Normalização por canal
            for ch in range(16):
                channel_data = eeg_data[:, ch]
                q75, q25 = np.percentile(channel_data, [75, 25])
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1.0
                channel_mean = np.mean(channel_data)
                eeg_data[:, ch] = (channel_data - channel_mean) / iqr
            
            # Transpor para (16, 400) e criar tensor
            eeg_array = eeg_data.T
            eeg_tensor = torch.FloatTensor(eeg_array).unsqueeze(0).unsqueeze(0)
            eeg_tensor = eeg_tensor.to(self.device)
            
            # Predição
            with torch.no_grad():
                output = self.model(eeg_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                
                # Probabilidades para cada classe
                left_prob = probs[0][0].item()
                right_prob = probs[0][1].item()
            
            # Atualizar interface
            classes = ['🤚 Mão Esquerda', '✋ Mão Direita']
            timestamp = datetime.now()
            
            self.prediction_label.setText(classes[pred])
            if classes[pred] == '🤚 Mão Esquerda':
                self.prob_left_label.setText(f"Mão Esquerda: {left_prob:.1%}")
                self.send_udp_signal('esquerda')  # Enviar sinal UDP
            else:
                self.prob_right_label.setText(f"Mão Direita: {right_prob:.1%}")
                self.send_udp_signal('direita')  # Enviar sinal UDP 
            
            # Atualizar estilo baseado na predição
            if pred == 0:  # Mão esquerda
                self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3; padding: 10px;")
            else:  # Mão direita
                self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF9800; padding: 10px;")
            
            # Salvar predição
            self.predictions.append((timestamp, pred, conf))
            
        except Exception as e:
            print(f"Erro na predição: {e}")
    
    def on_data_received(self, data):
        """Callback para dados recebidos"""
        # Enviar para plot
        self.plot_widget.add_data(data)
        
        # Adicionar ao buffer de dados e verificar predição
        if self.game_mode:
            # Garantir que temos 16 canais
            eeg_data = data[:16] if len(data) >= 16 else data + [0.0] * (16 - len(data))
            self.eeg_buffer.append(eeg_data)
            self.samples_since_last_prediction += 1
            
            # Fazer predição a cada 400 amostras
            if len(self.eeg_buffer) >= self.window_size and self.samples_since_last_prediction >= self.window_size:
                window_data = list(self.eeg_buffer)[-self.window_size:]
                self.predict_movement(np.array(window_data))
                self.samples_since_last_prediction = 0
                
        # Enviar para logger se estiver gravando
        if self.is_recording and self.csv_logger:
            if USE_OPENBCI_LOGGER and hasattr(self.csv_logger, 'log_sample'):
                # Logger OpenBCI - verificar marcador pendente
                marker = self.pending_marker
                self.pending_marker = None  # Limpar marcador pendente
                
                # Garantir que temos 16 canais
                if len(data) == 16:
                    self.csv_logger.log_sample(data, marker)
                else:
                    # Ajustar dados se necessário
                    eeg_data = data[:16] if len(data) >= 16 else data + [0.0] * (16 - len(data))
                    self.csv_logger.log_sample(eeg_data, marker)
            else:
                # Logger simples (fallback)
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
    
    def update_session_timer(self):
        """Atualiza o display do timer de sessão"""
        if self.session_start_time is not None:
            # Calcular tempo decorrido
            elapsed = int(time.time() - self.session_start_time)
        else:
            elapsed = 0
        
        # Formatar tempo como HH:MM:SS
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        if self.is_recording:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet("color: gray; font-weight: bold;")
    
    def on_task_changed(self):
        """Callback chamado quando a tarefa é alterada"""
        task = self.task_combo.currentText()
        
        if not self.is_recording:
            # Só atualizar o texto se não estiver gravando
            if task == "Jogo":
                self.record_btn.setText("Iniciar Jogo")
                # Carregar modelo se ainda não foi carregado
                if self.model is None:
                    if not self.load_model():
                        self.task_combo.setCurrentText("Baseline")
                        return
                self.game_group.setVisible(True)
                self.stats_group.setVisible(True)
                self.accuracy_group.setVisible(True)  # Mostrar acurácia no jogo
            else:
                self.record_btn.setText("Iniciar Gravação")
                self.game_group.setVisible(False)
                self.stats_group.setVisible(False)
                self.accuracy_group.setVisible(False)  # Esconder acurácia fora do jogo
    
    def update_record_button_text(self):
        """Atualiza o texto do botão de gravação baseado no estado e tarefa"""
        task = self.task_combo.currentText()
        
        if self.is_recording:
            if task == "Jogo":
                self.record_btn.setText("Parar Jogo")
            else:
                self.record_btn.setText("Parar Gravação")
        else:
            if task == "Jogo":
                self.record_btn.setText("Iniciar Jogo")
            else:
                self.record_btn.setText("Iniciar Gravação")
    
    def _on_unity_message(self, message: str):
        """Callback para mensagens recebidas do Unity"""
        print(f"[Unity] Mensagem recebida: {message}")
        
        # Verificar se recebeu resposta CORRECT ou WRONG
        if "CORRECT" in message or "WRONG" in message:
            print(f"✅ Resposta recebida: {message}")
            if self.waiting_for_response:
                self.waiting_for_response = False
                self.response_received = True
                print("🔓 Liberado para enviar próximo sinal aleatório")
                # Aguardar 7 segundos antes do próximo sinal
                QTimer.singleShot(7000, self.send_next_random_signal)
        
        # Processar mensagens específicas do Unity
        if "FLOWER" in message:
            # Usar o signal existente para processar mensagens de acurácia
            self.accuracy_message_signal.emit(message)
        elif "CONNECTED" in message:
            print("[Unity] Confirmação de conexão recebida")
        elif "STATUS" in message:
            print(f"[Unity] Status: {message}")
    
    def _on_unity_connection(self, connected: bool):
        """Callback para mudanças no status de conexão com Unity"""
        if connected:
            print("[Unity] TCP conectado")
            # Aqui você pode atualizar a UI para mostrar que o Unity está conectado
        else:
            print("[Unity] TCP desconectado")
            # Aqui você pode atualizar a UI para mostrar que o Unity foi desconectado
