from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from brainbridge_v2.bootstrap.container import AppContainer
from brainbridge_v2.presentation.gui.widgets.patient_form import PatientRegistrationWidget
from brainbridge_v2.presentation.gui.widgets.streaming import StreamingWidget
from brainbridge_v2.presentation.gui.styles import Theme


class MainWindow(QMainWindow):
    """Janela principal da aplicação BCI (v2)"""

    def __init__(self, container: AppContainer):
        super().__init__()
        self.container = container
        self.setup_ui()

    def setup_ui(self):
        """Configura a interface principal"""
        self.setWindowTitle("BrainBridge - Sistema BCI")
        available = self.screen().availableGeometry()
        self.resize(min(1400, available.width() - 40), min(900, available.height() - 60))
        
        # Aplicar tema
        self.setStyleSheet(Theme.get_stylesheet())

        # Widget central com abas
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        title_label = QLabel("🧠 BrainBridge")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setStyleSheet(Theme.header_bar())
        layout.addWidget(title_label)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Aba de pacientes
        self.patient_widget = PatientRegistrationWidget(
            self.container.patient_controller,
            recording_controller=self.container.recording_controller,
        )
        patient_scroll = QScrollArea()
        patient_scroll.setWidgetResizable(True)
        patient_scroll.setWidget(self.patient_widget)
        self.tabs.addTab(patient_scroll, "👥 Cadastro de Pacientes")

        # Aba de streaming
        self.streaming_widget = StreamingWidget(
            eeg_stream_controller=self.container.eeg_stream_controller,
            inference_controller=self.container.inference_controller,
            training_controller=self.container.training_controller,
            patient_controller=self.container.patient_controller,
            recording_controller=self.container.recording_controller,
            session_controller=self.container.session_controller,
            marker_controller=self.container.marker_controller,
            unity_controller=self.container.unity_controller,
            esp32_controller=self.container.esp32_controller,
        )
        self.tabs.addTab(self.streaming_widget, "📊 Streaming e Gravação")

        layout.addWidget(self.tabs, 1)
        central_widget.setLayout(layout)

        self.statusBar().showMessage("✓ Sistema BCI inicializado")

    def closeEvent(self, event):
        """Evento de fechamento da aplicação"""
        # Parar streaming se estiver rodando
        try:
            self.streaming_widget.stop_streaming()
        except Exception:
            pass

        event.accept()
