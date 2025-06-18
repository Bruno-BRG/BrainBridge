"""
Main BCI System GUI Application
Provides patient management, LSL data acquisition, and real-time inference
"""
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTableWidget, QTableWidgetItem, QLineEdit, QTextEdit,
                           QComboBox, QSpinBox, QDateEdit, QGroupBox, QFormLayout,
                           QMessageBox, QFileDialog, QProgressBar, QFrame,
                           QSplitter, QTreeWidget, QTreeWidgetItem, QCheckBox,
                           QDoubleSpinBox, QLCDNumber, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
import logging
from datetime import datetime, date
import json
from typing import Dict, List, Optional

# Import our modules
from database import BCIDatabaseManager
from lsl_streamer import LSLDataStreamer, HandMovementAnnotator
from inference_engine import RealTimeInferenceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientManagementTab(QWidget):
    """Tab for patient management"""
    
    def __init__(self, db_manager: BCIDatabaseManager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
        self.refresh_patient_list()
    
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left side - Patient list
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Patient list
        self.patient_table = QTableWidget()
        self.patient_table.setColumnCount(5)
        self.patient_table.setHorizontalHeaderLabels([
            "ID", "Nome", "Idade", "Mão Afetada", "Última Sessão"
        ])
        self.patient_table.cellClicked.connect(self.on_patient_selected)
        
        # Add patient button
        add_patient_btn = QPushButton("Adicionar Paciente")
        add_patient_btn.clicked.connect(self.add_patient_dialog)
        
        left_layout.addWidget(QLabel("Lista de Pacientes"))
        left_layout.addWidget(self.patient_table)
        left_layout.addWidget(add_patient_btn)
        left_widget.setLayout(left_layout)
        
        # Right side - Patient details
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Patient info form
        patient_info_group = QGroupBox("Informações do Paciente")
        patient_info_layout = QFormLayout()
        
        self.patient_id_edit = QLineEdit()
        self.patient_name_edit = QLineEdit()
        self.patient_age_edit = QSpinBox()
        self.patient_age_edit.setRange(0, 120)
        self.patient_gender_combo = QComboBox()
        self.patient_gender_combo.addItems(["", "Masculino", "Feminino", "Outro"])
        self.patient_hand_combo = QComboBox()
        self.patient_hand_combo.addItems(["", "Esquerda", "Direita", "Ambas"])
        self.patient_stroke_date = QDateEdit()
        self.patient_stroke_date.setCalendarPopup(True)
        self.patient_time_since_stroke = QSpinBox()
        self.patient_time_since_stroke.setRange(0, 9999)
        self.patient_time_since_stroke.setSuffix(" dias")
        self.patient_medical_info = QTextEdit()
        
        patient_info_layout.addRow("ID do Paciente:", self.patient_id_edit)
        patient_info_layout.addRow("Nome:", self.patient_name_edit)
        patient_info_layout.addRow("Idade:", self.patient_age_edit)
        patient_info_layout.addRow("Sexo:", self.patient_gender_combo)
        patient_info_layout.addRow("Mão Afetada:", self.patient_hand_combo)
        patient_info_layout.addRow("Data do AVC:", self.patient_stroke_date)
        patient_info_layout.addRow("Tempo desde AVC:", self.patient_time_since_stroke)
        patient_info_layout.addRow("Informações Médicas:", self.patient_medical_info)
        
        patient_info_group.setLayout(patient_info_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Salvar")
        save_btn.clicked.connect(self.save_patient)
        delete_btn = QPushButton("Excluir")
        delete_btn.clicked.connect(self.delete_patient)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(delete_btn)
        
        # Sessions info
        sessions_group = QGroupBox("Sessões Recentes")
        sessions_layout = QVBoxLayout()
        
        self.sessions_table = QTableWidget()
        self.sessions_table.setColumnCount(4)
        self.sessions_table.setHorizontalHeaderLabels([
            "Data", "Tipo", "Duração", "Notas"
        ])
        
        sessions_layout.addWidget(self.sessions_table)
        sessions_group.setLayout(sessions_layout)
        
        right_layout.addWidget(patient_info_group)
        right_layout.addLayout(button_layout)
        right_layout.addWidget(sessions_group)
        right_widget.setLayout(right_layout)
        
        # Add to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
        self.current_patient_id = None
    
    def refresh_patient_list(self):
        """Refresh the patient list table"""
        patients = self.db_manager.get_all_patients()
        
        self.patient_table.setRowCount(len(patients))
        
        for row, patient in enumerate(patients):
            # Get latest session
            latest_session = self.db_manager.get_latest_session(patient['patient_id'])
            last_session = latest_session['session_date'] if latest_session else "Nunca"
            
            self.patient_table.setItem(row, 0, QTableWidgetItem(patient['patient_id']))
            self.patient_table.setItem(row, 1, QTableWidgetItem(patient['name']))
            self.patient_table.setItem(row, 2, QTableWidgetItem(str(patient['age'] or "")))
            self.patient_table.setItem(row, 3, QTableWidgetItem(patient['affected_hand'] or ""))
            self.patient_table.setItem(row, 4, QTableWidgetItem(last_session))
    
    def on_patient_selected(self, row, column):
        """Handle patient selection"""
        patient_id = self.patient_table.item(row, 0).text()
        self.load_patient_details(patient_id)
    
    def load_patient_details(self, patient_id: str):
        """Load patient details into form"""
        patient = self.db_manager.get_patient(patient_id)
        if not patient:
            return
        
        self.current_patient_id = patient_id
        
        self.patient_id_edit.setText(patient['patient_id'])
        self.patient_name_edit.setText(patient['name'])
        self.patient_age_edit.setValue(patient['age'] or 0)
        
        # Set combo boxes
        if patient['gender']:
            index = self.patient_gender_combo.findText(patient['gender'])
            if index >= 0:
                self.patient_gender_combo.setCurrentIndex(index)
        
        if patient['affected_hand']:
            index = self.patient_hand_combo.findText(patient['affected_hand'])
            if index >= 0:
                self.patient_hand_combo.setCurrentIndex(index)
        
        # Set dates
        if patient['stroke_date']:
            try:
                stroke_date = datetime.fromisoformat(patient['stroke_date']).date()
                self.patient_stroke_date.setDate(stroke_date)
            except:
                pass
        
        self.patient_time_since_stroke.setValue(patient['time_since_stroke'] or 0)
        self.patient_medical_info.setPlainText(patient['medical_info'] or "")
        
        # Load sessions
        self.load_patient_sessions(patient_id)
    
    def load_patient_sessions(self, patient_id: str):
        """Load patient sessions"""
        sessions = self.db_manager.get_patient_sessions(patient_id)
        
        self.sessions_table.setRowCount(len(sessions))
        
        for row, session in enumerate(sessions):
            session_date = datetime.fromisoformat(session['session_date']).strftime("%d/%m/%Y %H:%M")
            
            self.sessions_table.setItem(row, 0, QTableWidgetItem(session_date))
            self.sessions_table.setItem(row, 1, QTableWidgetItem(session['session_type']))
            self.sessions_table.setItem(row, 2, QTableWidgetItem(f"{session['duration_minutes'] or 0} min"))
            self.sessions_table.setItem(row, 3, QTableWidgetItem(session['notes'] or ""))
    
    def add_patient_dialog(self):
        """Show add patient dialog"""
        # Clear form
        self.patient_id_edit.clear()
        self.patient_name_edit.clear()
        self.patient_age_edit.setValue(0)
        self.patient_gender_combo.setCurrentIndex(0)
        self.patient_hand_combo.setCurrentIndex(0)
        self.patient_stroke_date.setDate(date.today())
        self.patient_time_since_stroke.setValue(0)
        self.patient_medical_info.clear()
        
        self.current_patient_id = None
    
    def save_patient(self):
        """Save patient data"""
        if not self.patient_id_edit.text() or not self.patient_name_edit.text():
            QMessageBox.warning(self, "Erro", "ID e Nome são obrigatórios!")
            return
        
        patient_data = {
            'patient_id': self.patient_id_edit.text(),
            'name': self.patient_name_edit.text(),
            'age': self.patient_age_edit.value() if self.patient_age_edit.value() > 0 else None,
            'gender': self.patient_gender_combo.currentText() if self.patient_gender_combo.currentIndex() > 0 else None,
            'affected_hand': self.patient_hand_combo.currentText() if self.patient_hand_combo.currentIndex() > 0 else None,
            'stroke_date': self.patient_stroke_date.date().toString(Qt.ISODate),
            'time_since_stroke': self.patient_time_since_stroke.value() if self.patient_time_since_stroke.value() > 0 else None,
            'medical_info': self.patient_medical_info.toPlainText()
        }
        
        try:
            if self.current_patient_id:
                # Update existing patient
                self.db_manager.update_patient(self.current_patient_id, patient_data)
                QMessageBox.information(self, "Sucesso", "Paciente atualizado com sucesso!")
            else:
                # Add new patient
                self.db_manager.add_patient(patient_data)
                QMessageBox.information(self, "Sucesso", "Paciente adicionado com sucesso!")
                self.current_patient_id = patient_data['patient_id']
            
            self.refresh_patient_list()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar paciente: {str(e)}")
    
    def delete_patient(self):
        """Delete current patient"""
        if not self.current_patient_id:
            QMessageBox.warning(self, "Erro", "Nenhum paciente selecionado!")
            return
        
        reply = QMessageBox.question(self, "Confirmar", 
                                   "Tem certeza que deseja excluir este paciente?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Note: In a real application, you would implement patient deletion
            # For now, we'll just show a message
            QMessageBox.information(self, "Info", "Funcionalidade de exclusão não implementada por segurança.")

class LSLAcquisitionTab(QWidget):
    """Tab for LSL data acquisition"""
    
    def __init__(self, db_manager: BCIDatabaseManager):
        super().__init__()
        self.db_manager = db_manager
        self.lsl_streamer = LSLDataStreamer()
        self.annotator = None
        self.current_session_id = None
        
        self.init_ui()
        
        # Timer for updating UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every second
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection section
        connection_group = QGroupBox("Conexão LSL")
        connection_layout = QVBoxLayout()
        
        # Stream discovery
        discovery_layout = QHBoxLayout()
        self.discover_btn = QPushButton("Buscar Streams")
        self.discover_btn.clicked.connect(self.discover_streams)
        
        self.refresh_btn = QPushButton("Atualizar")
        self.refresh_btn.clicked.connect(self.discover_streams)
        
        self.stream_combo = QComboBox()
        self.stream_combo.setMinimumWidth(300)
        
        discovery_layout.addWidget(QLabel("Streams Disponíveis:"))
        discovery_layout.addWidget(self.stream_combo)
        discovery_layout.addWidget(self.discover_btn)
        discovery_layout.addWidget(self.refresh_btn)
        discovery_layout.addStretch()
        
        # Connection controls
        connect_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Conectar ao Stream")
        self.connect_btn.clicked.connect(self.connect_stream)
        self.connect_btn.setEnabled(False)
        
        self.disconnect_btn = QPushButton("Desconectar")
        self.disconnect_btn.clicked.connect(self.disconnect_stream)
        self.disconnect_btn.setEnabled(False)
        
        self.connection_status = QLabel("Desconectado")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        
        self.stream_info_label = QLabel("Nenhum stream selecionado")
        self.stream_info_label.setWordWrap(True)
        
        connect_layout.addWidget(self.connect_btn)
        connect_layout.addWidget(self.disconnect_btn)
        connect_layout.addWidget(QLabel("Status:"))
        connect_layout.addWidget(self.connection_status)
        connect_layout.addStretch()
        
        connection_layout.addLayout(discovery_layout)
        connection_layout.addLayout(connect_layout)
        connection_layout.addWidget(QLabel("Info do Stream:"))
        connection_layout.addWidget(self.stream_info_label)
        
        connection_group.setLayout(connection_layout)
        
        # Patient selection
        patient_group = QGroupBox("Seleção de Paciente")
        patient_layout = QHBoxLayout()
        
        self.patient_combo = QComboBox()
        self.refresh_patient_combo()
        
        patient_layout.addWidget(QLabel("Paciente:"))
        patient_layout.addWidget(self.patient_combo)
        patient_layout.addStretch()
        
        patient_group.setLayout(patient_layout)
        
        # Recording section
        recording_group = QGroupBox("Gravação")
        recording_layout = QVBoxLayout()
        
        # Recording controls
        controls_layout = QHBoxLayout()
        
        self.start_recording_btn = QPushButton("Iniciar Gravação")
        self.start_recording_btn.clicked.connect(self.start_recording)
        self.start_recording_btn.setEnabled(False)
        
        self.stop_recording_btn = QPushButton("Parar Gravação")
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        self.stop_recording_btn.setEnabled(False)
        
        self.recording_status = QLabel("Parado")
        self.recording_time = QLCDNumber()
        self.recording_time.setDigitCount(8)
        self.recording_time.display("00:00:00")
        
        controls_layout.addWidget(self.start_recording_btn)
        controls_layout.addWidget(self.stop_recording_btn)
        controls_layout.addWidget(QLabel("Status:"))
        controls_layout.addWidget(self.recording_status)
        controls_layout.addWidget(QLabel("Tempo:"))
        controls_layout.addWidget(self.recording_time)
        
        # Annotation buttons
        annotation_layout = QHBoxLayout()
        
        self.left_hand_btn = QPushButton("Mão Esquerda (400 amostras)")
        self.left_hand_btn.clicked.connect(self.mark_left_hand)
        self.left_hand_btn.setEnabled(False)
        self.left_hand_btn.setStyleSheet("background-color: lightblue;")
        
        self.right_hand_btn = QPushButton("Mão Direita (400 amostras)")
        self.right_hand_btn.clicked.connect(self.mark_right_hand)
        self.right_hand_btn.setEnabled(False)
        self.right_hand_btn.setStyleSheet("background-color: lightgreen;")
        
        annotation_layout.addWidget(self.left_hand_btn)
        annotation_layout.addWidget(self.right_hand_btn)
        
        recording_layout.addLayout(controls_layout)
        recording_layout.addLayout(annotation_layout)
        
        recording_group.setLayout(recording_layout)
        
        # Stream info
        info_group = QGroupBox("Informações do Stream")
        info_layout = QFormLayout()
        
        self.stream_name_label = QLabel("N/A")
        self.sample_rate_label = QLabel("N/A")
        self.channels_label = QLabel("N/A")
        self.samples_received_label = QLabel("0")
        
        info_layout.addRow("Nome do Stream:", self.stream_name_label)
        info_layout.addRow("Taxa de Amostragem:", self.sample_rate_label)
        info_layout.addRow("Canais:", self.channels_label)
        info_layout.addRow("Amostras Recebidas:", self.samples_received_label)
        
        info_group.setLayout(info_layout)
        
        # Add all groups to main layout
        layout.addWidget(connection_group)
        layout.addWidget(patient_group)
        layout.addWidget(recording_group)
        layout.addWidget(info_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        self.recording_start_time = None
    
    def refresh_patient_combo(self):
        """Refresh patient combo box"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecione um paciente...", "")
        
        patients = self.db_manager.get_all_patients()
        for patient in patients:
            self.patient_combo.addItem(f"{patient['name']} ({patient['patient_id']})", 
                                     patient['patient_id'])
    
    def connect_stream(self):
        """Connect to LSL stream"""
        if self.lsl_streamer.is_connected():
            self.lsl_streamer.disconnect()
            self.connect_btn.setText("Conectar ao Stream")
            self.connection_status.setText("Desconectado")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            self.start_recording_btn.setEnabled(False)
            return
        
        if self.lsl_streamer.find_stream():
            self.connect_btn.setText("Desconectar")
            self.connection_status.setText("Conectado")
            self.connection_status.setStyleSheet("color: green; font-weight: bold;")
            self.start_recording_btn.setEnabled(True)
            
            # Update stream info
            info = self.lsl_streamer.get_stream_info()
            self.stream_name_label.setText(info.get('name', 'N/A'))
            self.sample_rate_label.setText(f"{info.get('sample_rate', 0)} Hz")
            self.channels_label.setText(str(info.get('channel_count', 0)))
        else:
            QMessageBox.warning(self, "Erro", "Não foi possível conectar ao stream LSL!")
    
    def start_recording(self):
        """Start recording"""
        patient_id = self.patient_combo.currentData()
        if not patient_id:
            QMessageBox.warning(self, "Erro", "Selecione um paciente!")
            return
        
        # Create session
        session_data = {
            'patient_id': patient_id,
            'session_type': 'acquisition',
            'notes': 'Aquisição via LSL'
        }
        
        self.current_session_id = self.db_manager.add_session(session_data)
        
        # Setup file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_session_{timestamp}.csv"
        
        # Start recording
        self.lsl_streamer.start_recording(filename)
        
        # Setup annotator
        self.annotator = HandMovementAnnotator(self.lsl_streamer)
        
        # Update UI
        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self.left_hand_btn.setEnabled(True)
        self.right_hand_btn.setEnabled(True)
        self.recording_status.setText("Gravando")
        self.recording_status.setStyleSheet("color: red; font-weight: bold;")
        
        self.recording_start_time = datetime.now()
    
    def stop_recording(self):
        """Stop recording"""
        if not self.lsl_streamer.is_recording:
            return
        
        # Stop recording and get file path
        file_path = self.lsl_streamer.stop_recording()
        
        if file_path and self.current_session_id:
            # Calculate duration
            duration = (datetime.now() - self.recording_start_time).total_seconds() / 60
            
            # Add recording to database
            recording_data = {
                'session_id': self.current_session_id,
                'patient_id': self.patient_combo.currentData(),
                'file_path': file_path,
                'recording_type': 'eeg',
                'sample_rate': self.lsl_streamer.sample_rate,
                'channels': self.lsl_streamer.n_channels,
                'duration_seconds': duration * 60,
                'annotations': self.lsl_streamer.annotations
            }
            
            self.db_manager.add_recording(recording_data)
            
            # Update session duration
            self.db_manager.update_session(self.current_session_id, 
                                         {'duration_minutes': int(duration)})
        
        # Update UI
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.left_hand_btn.setEnabled(False)
        self.right_hand_btn.setEnabled(False)
        self.recording_status.setText("Parado")
        self.recording_status.setStyleSheet("color: black;")
        
        self.current_session_id = None
        self.recording_start_time = None
        
        if file_path:
            QMessageBox.information(self, "Sucesso", f"Gravação salva em: {file_path}")
    
    def mark_left_hand(self):
        """Mark left hand movement"""
        if self.annotator:
            self.annotator.start_left_hand_annotation()
            self.left_hand_btn.setStyleSheet("background-color: blue; color: white;")
            QTimer.singleShot(3200, lambda: self.left_hand_btn.setStyleSheet("background-color: lightblue;"))
    
    def mark_right_hand(self):
        """Mark right hand movement"""
        if self.annotator:
            self.annotator.start_right_hand_annotation()
            self.right_hand_btn.setStyleSheet("background-color: green; color: white;")
            QTimer.singleShot(3200, lambda: self.right_hand_btn.setStyleSheet("background-color: lightgreen;"))
    
    def update_ui(self):
        """Update UI elements"""
        if self.lsl_streamer.is_connected():
            # Update sample count (you would need to implement this in the streamer)
            pass
        
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            self.recording_time.display(time_str)
    
    def discover_streams(self):
        """Discover available LSL streams"""
        self.discover_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.discover_btn.setText("Buscando...")
        
        # Clear current streams
        self.stream_combo.clear()
        self.stream_combo.addItem("Buscando streams...", None)
        
        try:
            # Discover streams
            streams = self.lsl_streamer.discover_streams(timeout=3.0)
            
            # Update combo box
            self.stream_combo.clear()
            if streams:
                self.stream_combo.addItem("Selecione um stream...", None)
                for i, stream in enumerate(streams):
                    stream_text = f"{stream['name']} ({stream['type']}) - {stream['channel_count']} ch @ {stream['nominal_srate']} Hz"
                    self.stream_combo.addItem(stream_text, i)
                
                self.stream_combo.currentIndexChanged.connect(self.on_stream_selected)
                self.connect_btn.setEnabled(True)
                
                QMessageBox.information(self, "Sucesso", f"Encontrados {len(streams)} stream(s) LSL!")
            else:
                self.stream_combo.addItem("Nenhum stream encontrado", None)
                QMessageBox.warning(self, "Aviso", 
                    "Nenhum stream LSL encontrado!\n\n"
                    "Verifique se:\n"
                    "• Seu dispositivo EEG está conectado\n"
                    "• O software LSL está rodando\n"
                    "• O firewall não está bloqueando as portas LSL")
                    
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao buscar streams: {str(e)}")
            self.stream_combo.clear()
            self.stream_combo.addItem("Erro na busca", None)
        
        finally:
            self.discover_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.discover_btn.setText("Buscar Streams")
    
    def on_stream_selected(self):
        """Handle stream selection"""
        stream_index = self.stream_combo.currentData()
        if stream_index is not None and self.lsl_streamer.available_streams:
            stream = self.lsl_streamer.available_streams[stream_index]
            info_text = (f"Nome: {stream['name']}\n"
                        f"Tipo: {stream['type']}\n"
                        f"Canais: {stream['channel_count']}\n"
                        f"Taxa: {stream['nominal_srate']} Hz\n"
                        f"Host: {stream['hostname']}\n"
                        f"Source ID: {stream['source_id']}")
            self.stream_info_label.setText(info_text)
            self.connect_btn.setEnabled(True)
        else:
            self.stream_info_label.setText("Nenhum stream selecionado")
            self.connect_btn.setEnabled(False)
    
    def disconnect_stream(self):
        """Disconnect from current stream"""
        if self.lsl_streamer.is_recording:
            self.stop_recording()
        
        self.lsl_streamer.disconnect()
        self.connection_status.setText("Desconectado")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_recording_btn.setEnabled(False)
        
        # Clear stream info
        self.stream_name_label.setText("N/A")
        self.sample_rate_label.setText("N/A")
        self.channels_label.setText("N/A")
        self.samples_received_label.setText("0")

class RealTimeInferenceTab(QWidget):
    """Tab for real-time inference"""
    
    def __init__(self, db_manager: BCIDatabaseManager):
        print("🔧 Inicializando RealTimeInferenceTab...")
        super().__init__()
        self.db_manager = db_manager
        print("✅ DB manager definido")
        
        try:
            print("🔧 Criando LSLDataStreamer...")
            self.lsl_streamer = LSLDataStreamer()
            print("✅ LSLDataStreamer criado")
        except Exception as e:
            print(f"❌ Erro ao criar LSLDataStreamer: {e}")
            raise
            
        try:
            print("🔧 Criando RealTimeInferenceManager...")
            self.inference_manager = RealTimeInferenceManager()
            print("✅ RealTimeInferenceManager criado")
        except Exception as e:
            print(f"❌ Erro ao criar RealTimeInferenceManager: {e}")
            raise
        
        try:
            print("🔧 Inicializando UI do RealTimeInferenceTab...")
            self.init_ui()
            print("✅ UI do RealTimeInferenceTab inicializada")
        except Exception as e:
            print(f"❌ Erro ao inicializar UI do RealTimeInferenceTab: {e}")
            raise
            
        try:
            print("🔧 Carregando modelos disponíveis...")
            self.load_available_models()
            print("✅ Modelos carregados")
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            raise
        
        try:
            print("🔧 Configurando timer de atualização...")
            # Timer for updating UI
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_ui)
            self.update_timer.start(100)  # Update every 100ms
            print("✅ Timer configurado")
        except Exception as e:
            print(f"❌ Erro ao configurar timer: {e}")
            raise
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection section
        connection_group = QGroupBox("Conexão e Configuração")
        connection_layout = QVBoxLayout()
        
        # LSL connection
        lsl_layout = QHBoxLayout()
        self.connect_lsl_btn = QPushButton("Conectar LSL")
        self.connect_lsl_btn.clicked.connect(self.connect_lsl)
        self.lsl_status = QLabel("Desconectado")
        self.lsl_status.setStyleSheet("color: red; font-weight: bold;")
        
        lsl_layout.addWidget(self.connect_lsl_btn)
        lsl_layout.addWidget(QLabel("LSL Status:"))
        lsl_layout.addWidget(self.lsl_status)
        lsl_layout.addStretch()
        
        # Model selection
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.load_model_btn = QPushButton("Carregar Modelo")
        self.load_model_btn.clicked.connect(self.load_model)
        self.model_status = QLabel("Nenhum modelo carregado")
        
        model_layout.addWidget(QLabel("Modelo:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_status)
        model_layout.addStretch()
        
        connection_layout.addLayout(lsl_layout)
        connection_layout.addLayout(model_layout)
        connection_group.setLayout(connection_layout)
        
        # Inference controls
        inference_group = QGroupBox("Inferência em Tempo Real")
        inference_layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_inference_btn = QPushButton("Iniciar Inferência")
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.start_inference_btn.setEnabled(False)
        
        self.stop_inference_btn = QPushButton("Parar Inferência")
        self.stop_inference_btn.clicked.connect(self.stop_inference)
        self.stop_inference_btn.setEnabled(False)
        
        self.inference_status = QLabel("Parado")
        
        control_layout.addWidget(self.start_inference_btn)
        control_layout.addWidget(self.stop_inference_btn)
        control_layout.addWidget(QLabel("Status:"))
        control_layout.addWidget(self.inference_status)
        control_layout.addStretch()
        
        inference_layout.addLayout(control_layout)
        inference_group.setLayout(inference_layout)
        
        # Results display
        results_group = QGroupBox("Resultados da Predição")
        results_layout = QVBoxLayout()
        
        # Current prediction
        prediction_layout = QHBoxLayout()
        
        self.predicted_class_label = QLabel("---")
        self.predicted_class_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.predicted_class_label.setAlignment(Qt.AlignCenter)
        self.predicted_class_label.setStyleSheet("border: 2px solid gray; padding: 20px; background-color: lightgray;")
        
        self.confidence_label = QLabel("Confiança: ---%")
        self.confidence_label.setFont(QFont("Arial", 16))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        prediction_layout.addWidget(self.predicted_class_label)
        prediction_layout.addWidget(self.confidence_label)
        
        # Probability bars
        prob_layout = QVBoxLayout()
        
        self.left_prob_label = QLabel("Mão Esquerda: 0%")
        self.left_prob_bar = QProgressBar()
        self.left_prob_bar.setRange(0, 100)
        
        self.right_prob_label = QLabel("Mão Direita: 0%")
        self.right_prob_bar = QProgressBar()
        self.right_prob_bar.setRange(0, 100)
        
        prob_layout.addWidget(self.left_prob_label)
        prob_layout.addWidget(self.left_prob_bar)
        prob_layout.addWidget(self.right_prob_label)
        prob_layout.addWidget(self.right_prob_bar)
        
        results_layout.addLayout(prediction_layout)
        results_layout.addLayout(prob_layout)
        results_group.setLayout(results_layout)
        
        # Buffer status
        buffer_group = QGroupBox("Status do Buffer")
        buffer_layout = QFormLayout()
        
        self.buffer_size_label = QLabel("0")
        self.samples_needed_label = QLabel("0")
        self.buffer_progress = QProgressBar()
        
        buffer_layout.addRow("Amostras no Buffer:", self.buffer_size_label)
        buffer_layout.addRow("Amostras Necessárias:", self.samples_needed_label)
        buffer_layout.addRow("Progresso:", self.buffer_progress)
        
        buffer_group.setLayout(buffer_layout)
        
        # Add all groups to main layout
        layout.addWidget(connection_group)
        layout.addWidget(inference_group)
        layout.addWidget(results_group)
        layout.addWidget(buffer_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def load_available_models(self):
        """Load available models from the models directory"""
        self.model_combo.clear()
        self.model_combo.addItem("Selecione um modelo...", "")
        
        models_dir = Path("models/teste")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                model_name = model_file.stem
                self.model_combo.addItem(model_name, str(model_file))
    
    def connect_lsl(self):
        """Connect to LSL stream"""
        if self.lsl_streamer.is_connected():
            self.lsl_streamer.disconnect()
            self.connect_lsl_btn.setText("Conectar LSL")
            self.lsl_status.setText("Desconectado")
            self.lsl_status.setStyleSheet("color: red; font-weight: bold;")
            self.start_inference_btn.setEnabled(False)
            return
        
        if self.lsl_streamer.find_stream():
            self.connect_lsl_btn.setText("Desconectar LSL")
            self.lsl_status.setText("Conectado")
            self.lsl_status.setStyleSheet("color: green; font-weight: bold;")
            self.update_inference_button_state()
        else:
            QMessageBox.warning(self, "Erro", "Não foi possível conectar ao stream LSL!")
    
    def load_model(self):
        """Load selected model"""
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "Erro", "Selecione um modelo!")
            return
        
        model_name = self.model_combo.currentText()
        
        if self.inference_manager.load_model(model_name, model_path):
            self.inference_manager.set_active_model(model_name)
            self.model_status.setText(f"Modelo carregado: {model_name}")
            self.model_status.setStyleSheet("color: green;")
            self.update_inference_button_state()
            
            # Update buffer info
            if model_name in self.inference_manager.engines:
                engine = self.inference_manager.engines[model_name]
                self.samples_needed_label.setText(str(engine.window_size))
                self.buffer_progress.setRange(0, engine.window_size)
        else:
            QMessageBox.critical(self, "Erro", f"Falha ao carregar modelo: {model_name}")
    
    def update_inference_button_state(self):
        """Update the state of inference button"""
        can_infer = (self.lsl_streamer.is_connected() and 
                    self.inference_manager.active_engine is not None)
        self.start_inference_btn.setEnabled(can_infer)
    
    def start_inference(self):
        """Start real-time inference"""
        # Set up data callback
        self.lsl_streamer.set_data_callback(self.on_new_sample)
        
        # Start inference manager
        self.inference_manager.start_inference(self.on_prediction)
        
        # Update UI
        self.start_inference_btn.setEnabled(False)
        self.stop_inference_btn.setEnabled(True)
        self.inference_status.setText("Executando")
        self.inference_status.setStyleSheet("color: green; font-weight: bold;")
    
    def stop_inference(self):
        """Stop real-time inference"""
        self.inference_manager.stop_inference()
        
        # Update UI
        self.start_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        self.inference_status.setText("Parado")
        self.inference_status.setStyleSheet("color: black;")
        
        # Reset display
        self.predicted_class_label.setText("---")
        self.confidence_label.setText("Confiança: ---%")
        self.left_prob_bar.setValue(0)
        self.right_prob_bar.setValue(0)
        self.left_prob_label.setText("Mão Esquerda: 0%")
        self.right_prob_label.setText("Mão Direita: 0%")    
    def on_new_sample(self, sample, timestamp):
        """Handle new sample from LSL stream"""
        if len(sample) >= 16:  # Ensure we have EEG channels
            eeg_sample = sample[:16]  # Take first 16 channels
            self.inference_manager.add_sample(eeg_sample)
    
    def on_prediction(self, prediction_data):
        """Handle new prediction"""
        predicted_class = prediction_data['predicted_class']
        class_label = prediction_data['class_label']
        confidence = prediction_data['confidence']
        probabilities = prediction_data['probabilities']
        is_confident = prediction_data.get('is_confident', True)
        
        # Update main prediction display
        self.predicted_class_label.setText(class_label)
        self.confidence_label.setText(f"Confiança: {confidence*100:.1f}%")
        
        # Color coding with three states
        if predicted_class == 2 or not is_confident:  # Uncertain/Rest
            self.predicted_class_label.setStyleSheet(
                "border: 2px solid orange; padding: 20px; background-color: lightyellow; color: orange; font-weight: bold;")
        elif predicted_class == 0:  # Left hand
            self.predicted_class_label.setStyleSheet(
                "border: 2px solid blue; padding: 20px; background-color: lightblue; color: blue; font-weight: bold;")
        else:  # Right hand
            self.predicted_class_label.setStyleSheet(
                "border: 2px solid green; padding: 20px; background-color: lightgreen; color: green; font-weight: bold;")
        
        # Update probability bars
        if len(probabilities) >= 2:
            left_prob = probabilities[0] * 100
            right_prob = probabilities[1] * 100
            
            # Adjust bar opacity/style based on confidence
            if not is_confident:
                # Dim the bars for uncertain predictions
                self.left_prob_bar.setStyleSheet("QProgressBar { opacity: 0.5; }")
                self.right_prob_bar.setStyleSheet("QProgressBar { opacity: 0.5; }")
            else:
                # Normal style for confident predictions
                self.left_prob_bar.setStyleSheet("")
                self.right_prob_bar.setStyleSheet("")
            
            self.left_prob_bar.setValue(int(left_prob))
            self.right_prob_bar.setValue(int(right_prob))
            self.left_prob_label.setText(f"Mão Esquerda: {left_prob:.1f}%")
            self.right_prob_label.setText(f"Mão Direita: {right_prob:.1f}%")
    
    def update_ui(self):
        """Update UI elements"""
        if (self.inference_manager.active_engine and 
            self.inference_manager.active_engine in self.inference_manager.engines):
            
            engine = self.inference_manager.engines[self.inference_manager.active_engine]
            buffer_size = engine.get_buffer_size()
            
            self.buffer_size_label.setText(str(buffer_size))
            self.buffer_progress.setValue(buffer_size)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        print("🔧 Inicializando MainWindow...")
        super().__init__()
        print("✅ Super().__init__() concluído")
        
        try:
            print("🔧 Inicializando BCIDatabaseManager...")
            self.db_manager = BCIDatabaseManager()
            print("✅ BCIDatabaseManager inicializado")
        except Exception as e:
            print(f"❌ Erro ao inicializar BCIDatabaseManager: {e}")
            raise
        
        try:
            print("🔧 Inicializando UI...")
            self.init_ui()
            print("✅ UI inicializada com sucesso")
        except Exception as e:
            print(f"❌ Erro ao inicializar UI: {e}")
            raise
    
    def init_ui(self):
        print("🔧 Configurando janela principal...")
        self.setWindowTitle("Sistema BCI - Brain-Computer Interface")
        self.setGeometry(100, 100, 1400, 900)
        print("✅ Janela configurada")
        
        print("🔧 Criando widget central...")
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        print("✅ Widget central criado")
        
        print("🔧 Criando layout...")
        layout = QVBoxLayout()
        print("✅ Layout criado")
        
        print("🔧 Criando tab widget...")
        # Create tab widget
        self.tab_widget = QTabWidget()
        print("✅ Tab widget criado")
        
        try:
            print("🔧 Criando PatientManagementTab...")
            # Add tabs
            self.patient_tab = PatientManagementTab(self.db_manager)
            print("✅ PatientManagementTab criado")
        except Exception as e:
            print(f"❌ Erro ao criar PatientManagementTab: {e}")
            raise
        
        try:
            print("🔧 Criando LSLAcquisitionTab...")
            self.acquisition_tab = LSLAcquisitionTab(self.db_manager)
            print("✅ LSLAcquisitionTab criado")
        except Exception as e:
            print(f"❌ Erro ao criar LSLAcquisitionTab: {e}")
            raise
        
        try:
            print("🔧 Criando RealTimeInferenceTab...")
            self.inference_tab = RealTimeInferenceTab(self.db_manager)
            print("✅ RealTimeInferenceTab criado")
        except Exception as e:
            print(f"❌ Erro ao criar RealTimeInferenceTab: {e}")
            raise
        
        self.tab_widget.addTab(self.patient_tab, "Gestão de Pacientes")
        self.tab_widget.addTab(self.acquisition_tab, "Aquisição LSL")
        self.tab_widget.addTab(self.inference_tab, "Inferência em Tempo Real")
        
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sistema BCI iniciado")
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 10px 0;
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
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

def main():
    print("🔧 Iniciando aplicação BCI...")
    app = QApplication(sys.argv)
    print("✅ QApplication criada")
    
    # Set application properties
    app.setApplicationName("Sistema BCI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("BCI Research")
    print("✅ Propriedades da aplicação definidas")
    
    try:
        print("🔧 Criando janela principal...")
        # Create and show main window
        window = MainWindow()
        print("✅ MainWindow criada com sucesso")
        
        print("🔧 Mostrando janela...")
        window.show()
        print("✅ Janela mostrada")
        
        print("🔧 Iniciando loop da aplicação...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"❌ Erro durante inicialização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
