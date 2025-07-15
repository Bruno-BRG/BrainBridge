from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from bci.ui.patient_registration import PatientRegistrationWidget
from bci.ui.streaming_widget import StreamingWidget

class MainWindow(QMainWindow):
    def __init__(self, db_manager):
        super().__init__()
        self.setWindowTitle("Sistema BCI")
        self.setGeometry(100,100,1200,800)
        central = QWidget()
        v = QVBoxLayout()
        title = QLabel("Sistema BCI")
        title.setFont(QFont("Arial",16,QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)
        tabs = QTabWidget()
        tabs.addTab(PatientRegistrationWidget(db_manager), "Pacientes")
        tabs.addTab(StreamingWidget(db_manager), "Streaming")
        v.addWidget(tabs)
        central.setLayout(v)
        self.setCentralWidget(central)
