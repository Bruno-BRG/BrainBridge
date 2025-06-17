"""
Class:   BCIMainWindow
Purpose: Main application window for the BCI project, hosting various tabs for data management, model training, and real-time interaction.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Integrates different UI tabs and backend functionalities.
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTabWidget
)

# Add project root to Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.train_model import main as train_main_script # Import the training script

# Matplotlib imports for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# PyLSL imports for real-time streaming
try:
    from pylsl import StreamInlet, resolve_streams
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    

# Import tab classes

from src.UI.pylsl_tab import PylslTab # Import PylslTab
from src.UI.patient_management_tab import PatientManagementTab  # Updated import

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Application")
        self.setGeometry(100, 100, 1000, 800) # Adjusted size
        self.current_model_name = "unnamed_model" # Initialize current_model_name

        # Initialize data cache
        self.data_cache = {
            "windows": None,
            "labels": None,
            "subject_ids": None,
            "data_summary": "No data loaded yet.",
            "data_path": os.path.join(project_root, "eeg_data"), # Default data path relative to project root
            "subjects_list": None, 
            "current_plot_index": 0,
            "label_map": {0: "Right Hand", 1: "Left Hand"} # Assuming 0 for Right, 1 for Left
        }
        # Initialize training parameters cache / config
        self.training_params_config = {
            "model_name": "eeg_inception_openbci_cv10",  # Nome padrão igual ao notebook
            # NOTA: Todos os outros parâmetros estão FIXADOS no TrainingThread
            # para garantir reprodutibilidade exata do notebook
        }

        # Initialize patient data for compatibility
        self.current_patient_id = None
        self.current_patient_data = None

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tab widget for different sections
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.patient_tab = PatientManagementTab(self)  # Instantiate PatientManagementTab
        self.patient_tab.patient_selected.connect(self.on_patient_selected)  # Connect signal
        self.pylsl_tab = PylslTab(self)

        # Add tabs to the tab widget
        self.tabs.insertTab(0, self.patient_tab, "Patient Management")  # Add as first tab
        self.tabs.addTab(self.pylsl_tab, "OpenBCI Live (PyLSL)") # Add PylslTab instance

        # Initialize custom parameter inputs for backward compatibility
        self.custom_param_inputs = {}  # CRITICAL: Add missing attribute

        # Exit button
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.close)
        self.main_layout.addWidget(self.exit_button)

    def on_patient_selected(self, patient_id, patient_data):
        """Handle patient selection from patient management tab."""
        self.current_patient_id = patient_id
        self.current_patient_data = patient_data
        
        # Update window title to show current patient
        self.setWindowTitle(f"BCI Application - Patient: {patient_id}")
        
        # Update data paths to be patient-specific
        patient_data_dir = os.path.join(project_root, "patient_data", patient_id)
        os.makedirs(patient_data_dir, exist_ok=True)
        
        # Configure PylslTab to use patient folder automatically
        if hasattr(self.pylsl_tab, 'set_patient_folder'):
            self.pylsl_tab.set_patient_folder(patient_id, patient_data)
        
        print(f"Patient selected: {patient_id} - {patient_data.get('name', 'Unknown')}")

    def clear_patient_selection(self):
        """Clear the current patient selection and reset related configurations."""
        self.current_patient_id = None
        self.current_patient_data = None
        
        # Reset window title
        self.setWindowTitle("BCI Application")
        
        # Clear PylslTab patient folder configuration
        if hasattr(self.pylsl_tab, 'clear_patient_folder'):
            self.pylsl_tab.clear_patient_folder()
        
        print("Patient selection cleared")

    def closeEvent(self, event):
        # Ensure resources in tabs are cleaned up if they have specific cleanup methods
        if hasattr(self.pylsl_tab, 'clear_resources'):
            self.pylsl_tab.clear_resources()
        # Add similar calls for other tabs if they implement resource cleanup
        super().closeEvent(event)

def start_gui():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
        # Add similar calls for other tabs if they implement resource cleanup
    super().closeEvent(event)

def start_gui():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
