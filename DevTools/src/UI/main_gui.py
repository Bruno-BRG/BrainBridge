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
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTabWidget, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QGroupBox, QTextEdit, QHBoxLayout,
    QRadioButton, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
# Add project root to Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import BCIDataLoader
from src.model.train_model import main as train_main_script # Import the training script

# Matplotlib imports for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyLSL imports for real-time streaming
try:
    from pylsl import StreamInlet, resolve_streams
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    
# Additional imports for real-time functionality
from PyQt5.QtCore import QTimer
from collections import deque
import time
from datetime import datetime

# Import tab classes
from src.UI.data_management_tab import DataManagementTab
from src.UI.training_tab import TrainingTab
from src.UI.pylsl_tab import PylslTab
from src.UI.fine_tuning_tab import FineTuningTab
from src.UI.plot_canvas import PlotCanvas


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
            "use_default_params": True,
            "model_name": "unnamed_model", # Add model_name here
            # Placeholder for custom params
            "epochs": 50, # Default from CLI
            "k_folds": 5, # Default from CLI
            "learning_rate": 0.001, # Default from CLI
            "early_stopping_patience": 5, # Default from CLI
            "batch_size": 32, # Default from CLI
            "test_split_size": 0.2, # Default from CLI
            "train_subject_ids": "all" # Default to all loaded subjects
        }

        # References to custom param input fields
        self.custom_param_inputs = {}

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tab widget for different sections
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)        # Create tabs
        self.data_tab = DataManagementTab(self) # Use the new class
        self.training_tab = TrainingTab(self) # Use the new TrainingTab class
        self.fine_tuning_tab = FineTuningTab() # Add Fine-Tuning tab
        self.pylsl_tab = PylslTab(self) # Instantiate PylslTab

        self.tabs.addTab(self.data_tab, "Data Management")
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.fine_tuning_tab, "Fine-Tuning") # Add Fine-Tuning tab
        self.tabs.addTab(self.pylsl_tab, "OpenBCI Live (PyLSL)") # Add PylslTab instance

        # Exit button
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.close)
        self.main_layout.addWidget(self.exit_button)

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
