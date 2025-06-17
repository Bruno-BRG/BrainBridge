"""
Class:   FineTuningTab
Purpose: GUI tab for fine-tuning pre-trained EEG models with patient data.
Author:  Bruno Rocha
Created: 2025-06-01
License: BSD (3-clause)
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Task 1.3: Fine-Tuning GUI Tab
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                            QDoubleSpinBox, QTextEdit, QProgressBar, QGroupBox,
                            QTableWidget, QTableWidgetItem, QFileDialog,
                            QMessageBox, QTabWidget, QCheckBox, QSlider,
                            QSplitter, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.fine_tuning import ModelFineTuner
from src.data.patient_data_manager import PatientDataManager


class FineTuningWorker(QThread):
    """Worker thread for fine-tuning operations to prevent GUI freezing."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_metrics_updated = pyqtSignal(dict)
    validation_completed = pyqtSignal(dict)
    fine_tuning_completed = pyqtSignal(bool, str)
    
    def __init__(self, fine_tuner: ModelFineTuner, patient_manager: PatientDataManager,
                 fine_tuning_params: Dict):
        super().__init__()
        self.fine_tuner = fine_tuner
        self.patient_manager = patient_manager
        self.fine_tuning_params = fine_tuning_params
        self.should_stop = False
    
    def run(self):
        """Execute fine-tuning in separate thread."""
        try:
            # Emit status updates during the process
            self.status_updated.emit("Loading pre-trained model...")
            self.progress_updated.emit(10)
            
            # Load pre-trained model
            model_loaded = self.fine_tuner.load_pretrained_model(
                model_path=self.fine_tuning_params['model_path']
            )
            
            if not model_loaded:
                self.fine_tuning_completed.emit(False, "Failed to load pre-trained model")
                return
            
            self.status_updated.emit("Configuring model for fine-tuning...")
            self.progress_updated.emit(20)
            
            # Configure for fine-tuning - now handled inside fine_tune_model
            
            self.status_updated.emit("Preparing patient data...")
            self.progress_updated.emit(30)
            
            try:
                # Prepare patient data using PatientDataManager
                train_loader, val_loader, test_loader = self.prepare_data_loaders()
                
                self.status_updated.emit("Starting fine-tuning...")
                self.progress_updated.emit(40)
                
                # Define callbacks for updating UI
                def status_callback(message):
                    if self.should_stop:
                        raise InterruptedError("Fine-tuning cancelled by user")
                    self.status_updated.emit(message)
                
                def progress_callback(value):
                    self.progress_updated.emit(40 + int(value * 0.5))  # Map 0-100 to 40-90
                
                def metrics_callback(metrics):
                    if self.should_stop:
                        raise InterruptedError("Fine-tuning cancelled by user")
                    self.training_metrics_updated.emit(metrics)
                
                # CRITICAL: Use *fixed* parameters matching the notebook exactly
                results = self.fine_tuner.fine_tune_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    epochs=30,                     # Fixed at 30 to match notebook
                    learning_rate=0.001,           # Fixed at 0.001 to match notebook
                    learning_rate_ratio=0.3,       # CHANGED: 0.3 instead of 0.1 (more aggressive)
                    early_stopping_patience=5,     # Fixed at 5 to match notebook
                    freeze_strategy="early",       # But don't actually freeze anything
                    status_callback=status_callback,
                    progress_callback=progress_callback,
                    metrics_callback=metrics_callback
                )
                
                self.status_updated.emit("Validating fine-tuned model...")
                self.progress_updated.emit(90)
                
                # Create validation results dict
                validation_results = {
                    'accuracy': results.get('final_test_accuracy', 0),
                    'loss': results.get('test_loss', 0),
                    'predictions': results.get('predictions', []),
                    'true_labels': results.get('true_labels', []),
                }
                
                # Create confusion matrix if we have predictions
                if results.get('predictions') and results.get('true_labels'):
                    from sklearn.metrics import confusion_matrix
                    import numpy as np
                    
                    cm = confusion_matrix(
                        results['true_labels'], 
                        results['predictions'],
                        labels=[0, 1]  # Binary classification: left(0) vs right(1) hand
                    )
                    validation_results['confusion_matrix'] = cm
                
                self.validation_completed.emit(validation_results)
                
                self.progress_updated.emit(100)
                self.status_updated.emit("Fine-tuning completed successfully!")
                self.fine_tuning_completed.emit(True, "Fine-tuning completed successfully")
                
            except InterruptedError:
                self.fine_tuning_completed.emit(False, "Fine-tuning cancelled by user")
                return
                
        except Exception as e:
            self.fine_tuning_completed.emit(False, f"Error during fine-tuning: {str(e)}")
    
    def prepare_data_loaders(self):
        """Prepare data loaders for fine-tuning using patient manager."""
        from sklearn.model_selection import train_test_split
        
        # Load patient recordings
        X, y = self.patient_manager.load_patient_recordings(
            self.fine_tuning_params['session_ids']
        )
        
        # CRITICAL: Use the exact same normalizer as the notebook
        from src.data.data_normalizer import ImprovedEEGNormalizer
        
        # Initialize normalizer with EXACT same parameters as notebook
        normalizer = ImprovedEEGNormalizer(
            method='robust_zscore',  # CRITICAL: Same method as notebook
            scope='channel',         # CRITICAL: Same scope as notebook
            outlier_threshold=3.0    # CRITICAL: Same threshold as notebook
        )
        
        # Apply normalization
        X_normalized = normalizer.fit_transform(X)
        
        # CRITICAL: Use EXACT same split ratio as notebook - 70/15/15
        # First split between train and temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_normalized, y,
            test_size=0.30,  # 30% for temp (val+test combined)
            random_state=42, # CRITICAL: Same random_state as notebook
            stratify=y
        )
        
        # Split temp into val and test (50/50 split of the 30%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,  # 50% of 30% = 15% of original data
            random_state=42, # CRITICAL: Same random_state as notebook
            stratify=y_temp
        )
        
        # Create datasets
        from src.data.data_loader import BCIDataset
        
        # CRITICAL: Use augmentation only for training dataset, exactly as notebook
        train_dataset = BCIDataset(X_train, y_train, augment=True)  # Only train gets augmentation
        val_dataset = BCIDataset(X_val, y_val, augment=False)
        test_dataset = BCIDataset(X_test, y_test, augment=False)
        
        # Create data loaders
        # CRITICAL: Use same batch size as notebook for fine-tuning
        batch_size = 8  # CRITICAL: Use fixed value of 8 for fine-tuning (not 10 like training)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,  # CRITICAL: Drop incomplete batches
            num_workers=0  # For Windows compatibility
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def stop(self):
        """Stop the fine-tuning process."""
        self.should_stop = True


class FineTuningTab(QWidget):
    """
    GUI tab for fine-tuning pre-trained EEG models with patient-specific data.
    
    Features:
    - Patient selection and data management
    - Recording session browser and selection
    - Fine-tuning parameter configuration
    - Real-time training progress monitoring
    - Validation results display
    - Model comparison tools
    """
    
    def __init__(self):
        super().__init__()
        self.fine_tuner = None
        self.patient_manager = None
        self.current_patient_id = None
        self.fine_tuning_worker = None
        self.training_history = []
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Configuration
        left_panel = self.create_configuration_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Monitoring and Results
        right_panel = self.create_monitoring_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def create_configuration_panel(self) -> QWidget:
        """Create the left configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Patient Selection Group
        patient_group = QGroupBox("Patient Selection")
        patient_layout = QGridLayout()
        
        # Patient ID input
        patient_layout.addWidget(QLabel("Patient ID:"), 0, 0)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter patient ID (e.g., P001)")
        patient_layout.addWidget(self.patient_id_input, 0, 1)
        
        # Load patient button
        self.load_patient_btn = QPushButton("Load Patient Data")
        patient_layout.addWidget(self.load_patient_btn, 0, 2)
        
        # Patient status label
        self.patient_status_label = QLabel("No patient loaded")
        self.patient_status_label.setStyleSheet("color: gray; font-style: italic;")
        patient_layout.addWidget(self.patient_status_label, 1, 0, 1, 3)
        
        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # Recording Sessions Group
        sessions_group = QGroupBox("Recording Sessions")
        sessions_layout = QVBoxLayout()
        
        # Sessions list
        self.sessions_list = QListWidget()
        self.sessions_list.setMaximumHeight(150)
        sessions_layout.addWidget(QLabel("Available Sessions:"))
        sessions_layout.addWidget(self.sessions_list)
        
        # Session actions
        session_actions_layout = QHBoxLayout()
        self.add_session_btn = QPushButton("Add Session")
        self.remove_session_btn = QPushButton("Remove Session")
        self.refresh_sessions_btn = QPushButton("Refresh")
        
        session_actions_layout.addWidget(self.add_session_btn)
        session_actions_layout.addWidget(self.remove_session_btn)
        session_actions_layout.addWidget(self.refresh_sessions_btn)
        sessions_layout.addLayout(session_actions_layout)
        
        sessions_group.setLayout(sessions_layout)
        layout.addWidget(sessions_group)
        
        # Model Configuration Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        # Pre-trained model selection
        model_layout.addWidget(QLabel("Pre-trained Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.populate_model_options()
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # Information label about fixed parameters
        info_label = QLabel("All fine-tuning parameters are optimized and fixed\nto match the notebook implementation.")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        model_layout.addWidget(info_label, 1, 0, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Control Buttons
        control_layout = QVBoxLayout()
        
        self.start_fine_tuning_btn = QPushButton("Start Fine-Tuning")
        self.start_fine_tuning_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.stop_fine_tuning_btn = QPushButton("Stop Fine-Tuning")
        self.stop_fine_tuning_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_fine_tuning_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_fine_tuning_btn)
        control_layout.addWidget(self.stop_fine_tuning_btn)
        
        layout.addLayout(control_layout)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_monitoring_panel(self) -> QWidget:
        """Create the right monitoring and results panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Progress and Status Group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready to start fine-tuning")
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Training Metrics Tabs
        self.metrics_tabs = QTabWidget()
        
        # Training Curves Tab
        self.training_curves_tab = self.create_training_curves_tab()
        self.metrics_tabs.addTab(self.training_curves_tab, "Training Curves")
        
        # Validation Results Tab
        self.validation_results_tab = self.create_validation_results_tab()
        self.metrics_tabs.addTab(self.validation_results_tab, "Validation Results")
        
        # Model Comparison Tab
        self.model_comparison_tab = self.create_model_comparison_tab()
        self.metrics_tabs.addTab(self.model_comparison_tab, "Model Comparison")
        
        layout.addWidget(self.metrics_tabs)
        
        panel.setLayout(layout)
        return panel
    
    def create_training_curves_tab(self) -> QWidget:
        """Create the training curves visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.training_figure = Figure(figsize=(10, 6))
        self.training_canvas = FigureCanvas(self.training_figure)
        
        # Initialize subplots
        self.loss_ax = self.training_figure.add_subplot(221)
        self.acc_ax = self.training_figure.add_subplot(222)
        self.lr_ax = self.training_figure.add_subplot(223)
        self.val_metrics_ax = self.training_figure.add_subplot(224)
        
        self.training_figure.tight_layout()
        
        layout.addWidget(self.training_canvas)
        widget.setLayout(layout)
        
        return widget
    
    def create_validation_results_tab(self) -> QWidget:
        """Create the validation results display tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Validation metrics table
        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(2)
        self.validation_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.validation_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Validation Metrics:"))
        layout.addWidget(self.validation_table)
        
        # Confusion matrix display
        self.confusion_figure = Figure(figsize=(6, 5))
        self.confusion_canvas = FigureCanvas(self.confusion_figure)
        
        layout.addWidget(QLabel("Confusion Matrix:"))
        layout.addWidget(self.confusion_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def create_model_comparison_tab(self) -> QWidget:
        """Create the model comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Model comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(5)
        self.comparison_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Precision", "Recall", "F1-Score"
        ])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Model Performance Comparison:"))
        layout.addWidget(self.comparison_table)
        
        # Comparison controls
        comparison_controls_layout = QHBoxLayout()
        self.save_model_btn = QPushButton("Save Fine-Tuned Model")
        self.load_comparison_btn = QPushButton("Load Comparison Models")
        
        comparison_controls_layout.addWidget(self.save_model_btn)
        comparison_controls_layout.addWidget(self.load_comparison_btn)
        comparison_controls_layout.addStretch()
        
        layout.addLayout(comparison_controls_layout)
        
        widget.setLayout(layout)
        return widget
    
    def setup_connections(self):
        """Set up signal-slot connections."""
        # Patient management
        self.load_patient_btn.clicked.connect(self.load_patient_data)
        
        # Session management
        self.add_session_btn.clicked.connect(self.add_recording_session)
        self.remove_session_btn.clicked.connect(self.remove_recording_session)
        self.refresh_sessions_btn.clicked.connect(self.refresh_sessions)
        
        # Fine-tuning control
        self.start_fine_tuning_btn.clicked.connect(self.start_fine_tuning)
        self.stop_fine_tuning_btn.clicked.connect(self.stop_fine_tuning)
        
        # Model comparison
        self.save_model_btn.clicked.connect(self.save_fine_tuned_model)
        self.load_comparison_btn.clicked.connect(self.load_comparison_models)
    
    def populate_model_options(self):
        """Populate the pre-trained model selection combo box."""
        # Look for available pre-trained models
        models_dir = "models"
        model_options = []
        
        if os.path.exists(models_dir):
            for subdir in os.listdir(models_dir):
                subdir_path = os.path.join(models_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Look for .pth files in subdirectory
                    for file in os.listdir(subdir_path):
                        if file.endswith("_final.pt"):
                            model_options.append(f"{subdir}/{file}")
        
        if not model_options:
            model_options = ["models\eeg_inception_openbci_cv10\eeginceptionerp_fold_final.pt"]

        self.model_combo.addItems(model_options)
    
    def load_patient_data(self):
        """Load patient data and initialize PatientDataManager."""
        patient_id = self.patient_id_input.text().strip()
        
        if not patient_id:
            QMessageBox.warning(self, "Input Error", "Please enter a patient ID.")
            return
        
        try:
            # Initialize patient data manager
            self.patient_manager = PatientDataManager(
                patient_id=patient_id,
                data_root="patient_data",
                verbose=True
            )
            
            self.current_patient_id = patient_id
            self.patient_status_label.setText(f"Patient {patient_id} loaded successfully")
            self.patient_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Initialize fine tuner
            self.fine_tuner = ModelFineTuner(verbose=True)
            
            # Refresh sessions list
            self.refresh_sessions()
            
            # Enable controls
            self.start_fine_tuning_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load patient data: {str(e)}")
            self.patient_status_label.setText(f"Failed to load patient {patient_id}")
            self.patient_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def refresh_sessions(self):
        """Refresh the recording sessions list."""
        if not self.patient_manager:
            return
        
        self.sessions_list.clear()
        
        try:
            summary = self.patient_manager.get_session_summary()
            
            for session in summary['sessions']:
                item_text = f"{session['session_id']} ({session['recording_files']} files)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, session['session_id'])
                self.sessions_list.addItem(item)
            
            if summary['total_sessions'] == 0:
                self.sessions_list.addItem("No recording sessions available")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh sessions: {str(e)}")
    
    def add_recording_session(self):
        """Add a new recording session."""
        if not self.patient_manager:
            QMessageBox.warning(self, "Error", "Please load patient data first.")
            return
        
        # Open file dialog to select recording files
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Recording Files",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not files:
            return
        
        # Get session ID from user
        session_id, ok = self.get_session_id_dialog()
        if not ok or not session_id:
            return
        
        try:
            # Add session to patient manager
            success = self.patient_manager.add_recording_session(
                session_id=session_id,
                recording_files=files,
                session_metadata={
                    "added_date": pd.Timestamp.now().isoformat(),
                    "file_count": len(files)
                }
            )
            
            if success:
                QMessageBox.information(self, "Success", f"Session '{session_id}' added successfully.")
                self.refresh_sessions()
            else:
                QMessageBox.warning(self, "Error", f"Failed to add session '{session_id}'.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error adding session: {str(e)}")
    
    def get_session_id_dialog(self):
        """Get session ID from user input dialog."""
        from PyQt5.QtWidgets import QInputDialog
        
        session_id, ok = QInputDialog.getText(
            self,
            "Session ID",
            "Enter session ID:",
            QLineEdit.Normal,
            f"session_{len(self.patient_manager.sessions) + 1:03d}"
        )
        
        return session_id.strip(), ok
    
    def remove_recording_session(self):
        """Remove the selected recording session."""
        current_item = self.sessions_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Selection Error", "Please select a session to remove.")
            return
        
        session_id = current_item.data(Qt.UserRole)
        if not session_id:
            return
        
        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove session '{session_id}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                success = self.patient_manager.remove_session(session_id)
                if success:
                    QMessageBox.information(self, "Success", f"Session '{session_id}' removed.")
                    self.refresh_sessions()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to remove session '{session_id}'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing session: {str(e)}")
    
    def start_fine_tuning(self):
        """Start the fine-tuning process with fixed parameters to match notebook."""
        if not self.patient_manager or not self.fine_tuner:
            QMessageBox.warning(self, "Error", "Please load patient data first.")
            return
        
        # Get selected sessions
        selected_sessions = []
        for i in range(self.sessions_list.count()):
            item = self.sessions_list.item(i)
            session_id = item.data(Qt.UserRole)
            if session_id:
                selected_sessions.append(session_id)
        
        if not selected_sessions:
            QMessageBox.warning(self, "Error", "No recording sessions available.")
            return
        
        # Use FIXED parameters matching the notebook exactly
        self.fine_tuning_params = {
            'model_path': os.path.join("models", self.model_combo.currentText()),
            'session_ids': selected_sessions,
            # Don't bother with any other parameters, they'll be overridden in run()
        }
        
        # Disable start button, enable stop button
        self.start_fine_tuning_btn.setEnabled(False)
        self.stop_fine_tuning_btn.setEnabled(True)
        
        # Clear previous results
        self.training_history.clear()
        self.progress_bar.setValue(0)
        
        # Start fine-tuning worker thread
        self.fine_tuning_worker = FineTuningWorker(
            self.fine_tuner, self.patient_manager, self.fine_tuning_params
        )
        
        # Connect worker signals
        self.fine_tuning_worker.progress_updated.connect(self.update_progress)
        self.fine_tuning_worker.status_updated.connect(self.update_status)
        self.fine_tuning_worker.training_metrics_updated.connect(self.update_training_metrics)
        self.fine_tuning_worker.validation_completed.connect(self.display_validation_results)
        self.fine_tuning_worker.fine_tuning_completed.connect(self.fine_tuning_finished)
        
        # Start the worker
        self.fine_tuning_worker.start()
    
    def stop_fine_tuning(self):
        """Stop the fine-tuning process."""
        if self.fine_tuning_worker and self.fine_tuning_worker.isRunning():
            self.fine_tuning_worker.stop()
            self.fine_tuning_worker.wait()
        
        self.start_fine_tuning_btn.setEnabled(True)
        self.stop_fine_tuning_btn.setEnabled(False)
        self.status_label.setText("Fine-tuning stopped by user")
    def update_progress(self, value: int):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
    
    def update_status(self, status: str):
        """Update the status label."""
        self.status_label.setText(status)
    
    def update_training_metrics(self, metrics: Dict):
        """Update the training curves with new metrics."""
        self.training_history.append(metrics)
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot the training curves."""
        if not self.training_history:
            return
        
        # Clear previous plots
        self.loss_ax.clear()
        self.acc_ax.clear()
        self.lr_ax.clear()
        self.val_metrics_ax.clear()
        
        epochs = [m['epoch'] for m in self.training_history]
        train_loss = [m['train_loss'] for m in self.training_history]
        train_acc = [m['train_accuracy'] for m in self.training_history]
        val_loss = [m['val_loss'] for m in self.training_history]
        val_acc = [m['val_accuracy'] for m in self.training_history]
        
        # Loss plot
        self.loss_ax.plot(epochs, train_loss, 'b-', label='Train Loss')
        self.loss_ax.plot(epochs, val_loss, 'r-', label='Val Loss')
        self.loss_ax.set_title('Training Loss')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.legend()
        self.loss_ax.grid(True)
        
        # Accuracy plot
        self.acc_ax.plot(epochs, train_acc, 'b-', label='Train Acc')
        self.acc_ax.plot(epochs, val_acc, 'r-', label='Val Acc')
        self.acc_ax.set_title('Training Accuracy')
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_ylabel('Accuracy')
        self.acc_ax.legend()
        self.acc_ax.grid(True)
        
        # Validation metrics over time
        self.val_metrics_ax.plot(epochs, val_acc, 'g-', label='Validation Accuracy')
        self.val_metrics_ax.set_title('Validation Metrics')
        self.val_metrics_ax.set_xlabel('Epoch')
        self.val_metrics_ax.set_ylabel('Score')
        self.val_metrics_ax.legend()
        self.val_metrics_ax.grid(True)
        
        self.training_figure.tight_layout()
        self.training_canvas.draw()
    
    def display_validation_results(self, results: Dict):
        """Display validation results in the table and confusion matrix."""
        # Update validation table
        self.validation_table.setRowCount(len(results))
        
        row = 0
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                self.validation_table.setItem(row, 0, QTableWidgetItem(metric.title()))
                if isinstance(value, float):
                    self.validation_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
                else:
                    self.validation_table.setItem(row, 1, QTableWidgetItem(str(value)))
                row += 1
        
        # Plot confusion matrix if available
        if 'confusion_matrix' in results:
            self.plot_confusion_matrix(results['confusion_matrix'])
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Left Hand', 'Right Hand'],
               yticklabels=['Left Hand', 'Right Hand'],
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        self.confusion_figure.tight_layout()
        self.confusion_canvas.draw()
    
    def fine_tuning_finished(self, success: bool, message: str):
        """Handle fine-tuning completion."""
        self.start_fine_tuning_btn.setEnabled(True)
        self.stop_fine_tuning_btn.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.update_model_comparison()
        else:
            QMessageBox.warning(self, "Error", message)
    
    def update_model_comparison(self):
        """Update the model comparison table with current results."""
        if not self.training_history:
            return
        
        # Get the final metrics
        final_metrics = self.training_history[-1]
        
        # Add to comparison table
        row_count = self.comparison_table.rowCount()
        self.comparison_table.insertRow(row_count)
        
        model_name = f"Fine-tuned_{self.current_patient_id}_{len(self.training_history)}"
        self.comparison_table.setItem(row_count, 0, QTableWidgetItem(model_name))
        self.comparison_table.setItem(row_count, 1, QTableWidgetItem(f"{final_metrics['val_accuracy']:.4f}"))
        self.comparison_table.setItem(row_count, 2, QTableWidgetItem("N/A"))  # Precision - would need actual calculation
        self.comparison_table.setItem(row_count, 3, QTableWidgetItem("N/A"))  # Recall - would need actual calculation
        self.comparison_table.setItem(row_count, 4, QTableWidgetItem("N/A"))  # F1 - would need actual calculation
    
    def save_fine_tuned_model(self):
        """Save the fine-tuned model."""
        if not self.fine_tuner or not self.fine_tuner.model:
            QMessageBox.warning(self, "Error", "No fine-tuned model available to save.")
            return
        
        # Get save location
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Fine-Tuned Model",
            f"models/fine_tuned_{self.current_patient_id}.",
            "PyTorch Models (*.pth);;All Files (*)"
        )
        
        if save_path:
            try:
                # Create model state dictionary with additional info for PyTorch 2.6 compatibility
                model_state = {
                    'model_state_dict': self.fine_tuner.model.state_dict(),
                    'model_name': f"fine_tuned_{self.current_patient_id}",
                    'model_version': '2.0',
                    'constructor_args': {
                        'n_chans': 16,
                        'n_outputs': 2,
                        'n_times': 400,
                        'sfreq': 125.0,
                        'drop_prob': 0.5,
                        'n_filters': 8
                    },
                    'created_with': f"PyTorch {torch.__version__}",
                    'creation_date': datetime.now().isoformat()
                }
                
                # Save with PyTorch 2.6 compatibility
                try:
                    torch.save(model_state, save_path, _use_new_zipfile_serialization=True)
                except TypeError:
                    # Fallback for newer PyTorch versions where _use_new_zipfile_serialization is deprecated
                    torch.save(model_state, save_path)
                    
                QMessageBox.information(self, "Success", f"Model saved to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def load_comparison_models(self):
        """Load models for comparison."""
        # This would implement loading multiple models for comparison
        QMessageBox.information(self, "Info", "Model comparison loading not yet implemented.")


# Test the FineTuningTab if run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show the fine-tuning tab
    fine_tuning_tab = FineTuningTab()
    fine_tuning_tab.show()
    
    sys.exit(app.exec_())
        # This would implement loading multiple models for comparison
    QMessageBox.information(self, "Info", "Model comparison loading not yet implemented.")


# Test the FineTuningTab if run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show the fine-tuning tab
    fine_tuning_tab = FineTuningTab()
    fine_tuning_tab.show()
    
    sys.exit(app.exec_())
    sys.exit(app.exec_())
