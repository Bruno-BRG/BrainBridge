"""
File:    pylsl_tab.py
Class:   PylslTab
Purpose: Provides the user interface tab for discovering, connecting to,
         and visualizing EEG data streams using the Lab Streaming
         Layer (LSL) library. Includes CSV recording functionality.
Author:  Bruno Rocha
Created: 2025-05-28
Modified: 2025-05-31
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Requires PyLSL to be installed. If PyLSL is not available,
         the tab will display an error message.
"""

import os
import sys
import numpy as np
import traceback
import csv
import datetime
import torch
import glob
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QHBoxLayout,
    QMessageBox, QFileDialog, QComboBox, QSlider, QCheckBox, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from collections import deque
from typing import Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .plot_canvas import PlotCanvas

# PyLSL imports
try:
    from pylsl import StreamInlet, resolve_streams
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    StreamInlet = None
    resolve_streams = None

# Model imports for inference
try:
    from src.model.eeg_inception_erp import EEGInceptionERPModel
    from src.model.realtime_inference import RealTimeInferenceProcessor
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model imports not available for inference: {e}")
    MODEL_IMPORTS_AVAILABLE = False
    EEGInceptionERPModel = None
    RealTimeInferenceProcessor = None

class PylslTab(QWidget):
    """
    Manages the Pylsl Tab in the main GUI application.

    This class is responsible for setting up the UI elements related to LSL stream
    interaction, handling user actions (e.g., start/stop stream),
    and managing the LSL data inlet and plotting timer for real-time visualization.
    Supports both Recording mode (saves CSV data) and Inference mode (real-time prediction).
    """
    def __init__(self, parent_main_window):
        """
        Initializes the PylslTab.

        Args:
            parent_main_window (QMainWindow): Reference to the main application window.
        """
        super().__init__()
        self.main_window = parent_main_window

        layout = QVBoxLayout(self)

        if not PYLSL_AVAILABLE:
            error_label = QLabel("PyLSL is not installed. Please install it to use this feature:")
            error_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            install_label = QLabel("pip install pylsl")
            install_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; font-family: monospace;")
            layout.addWidget(error_label)
            layout.addWidget(install_label)
            layout.addStretch()
            return        # Initialize mode variables
        self.current_mode = "recording"  # "recording" or "inference"
        self.inference_processor = None
        self.confidence_threshold = 0.7  # Default threshold
        self.inference_window_size = 400  # Changed from 600 to 400 samples
        self.selected_model_path = None
        self.loaded_model = None

        # Simplified UI setup directly in __init__
        self._setup_pylsl_ui(layout)
        
        self.setLayout(layout)

    def _setup_pylsl_ui(self, layout):
        """
        Sets up the UI elements for PyLSL interaction.

        Args:
            layout (QVBoxLayout): The main QVBoxLayout of the PylslTab.
        """
        # Mode selection group
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout()
        
        self.mode_recording_btn = QPushButton("Recording Mode")
        self.mode_inference_btn = QPushButton("Inference Mode")
        self.mode_recording_btn.setCheckable(True)
        self.mode_inference_btn.setCheckable(True)
        self.mode_recording_btn.setChecked(True)  # Default to recording mode
        
        self.mode_status_label = QLabel("Mode: Recording")
        self.mode_status_label.setStyleSheet("padding: 5px; background-color: #e6f3ff; border: 1px solid #0066cc; font-weight: bold;")
        
        mode_layout.addWidget(self.mode_recording_btn)
        mode_layout.addWidget(self.mode_inference_btn)
        mode_layout.addWidget(self.mode_status_label)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Recording setup group (only visible in recording mode)
        self.recording_setup_group = QGroupBox("Recording Setup")
        recording_setup_layout = QVBoxLayout()
        
        folder_selection_layout = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Recording Folder")
        self.selected_folder_label = QLabel("No folder selected")
        self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        folder_selection_layout.addWidget(self.select_folder_btn)
        folder_selection_layout.addWidget(self.selected_folder_label)
        recording_setup_layout.addLayout(folder_selection_layout)
        
        self.recording_setup_group.setLayout(recording_setup_layout)
        layout.addWidget(self.recording_setup_group)
        
        # Inference setup group (only visible in inference mode)
        self.inference_setup_group = QGroupBox("Inference Setup")
        inference_setup_layout = QVBoxLayout()
          # Model selection
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model:"))
        self.browse_model_btn = QPushButton("Browse Model...")
        self.clear_model_btn = QPushButton("Clear")
        model_selection_layout.addWidget(self.browse_model_btn)
        model_selection_layout.addWidget(self.clear_model_btn)
        inference_setup_layout.addLayout(model_selection_layout)
        
        # Selected model display
        self.selected_model_label = QLabel("No model selected")
        self.selected_model_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; font-family: monospace;")
        self.selected_model_label.setWordWrap(True)
        inference_setup_layout.addWidget(self.selected_model_label)
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 95)  # 0.5 to 0.95
        self.threshold_slider.setValue(70)  # Default 0.7
        self.threshold_label = QLabel("0.70")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        inference_setup_layout.addLayout(threshold_layout)
        
        # Inference results display
        self.inference_result_label = QLabel("Prediction: Waiting for data...")
        self.inference_result_label.setStyleSheet("padding: 10px; background-color: #f9f9f9; border: 2px solid #ddd; font-size: 14px; font-weight: bold;")
        inference_setup_layout.addWidget(self.inference_result_label)
        
        self.inference_setup_group.setLayout(inference_setup_layout)
        self.inference_setup_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.inference_setup_group)
        
        # Stream connection group
        connection_group = QGroupBox("Stream Connection")
        connection_layout = QVBoxLayout()
        
        stream_controls_layout = QHBoxLayout()
        self.pylsl_start_btn = QPushButton("Start LSL Stream & Recording")
        self.pylsl_stop_btn = QPushButton("Stop Stream & Recording")
        self.pylsl_refresh_btn = QPushButton("Refresh Streams")
        self.pylsl_stop_btn.setEnabled(False)
        self.pylsl_start_btn.setEnabled(False)  # Disabled until folder is selected
        
        stream_controls_layout.addWidget(self.pylsl_start_btn)
        stream_controls_layout.addWidget(self.pylsl_stop_btn)
        stream_controls_layout.addWidget(self.pylsl_refresh_btn)
        connection_layout.addLayout(stream_controls_layout)
        
        self.pylsl_status_label = QLabel("Status: Disconnected")
        self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        connection_layout.addWidget(self.pylsl_status_label)
        
        self.pylsl_info_text = QTextEdit()
        self.pylsl_info_text.setMaximumHeight(100)
        self.pylsl_info_text.setPlainText("No stream information available")
        connection_layout.addWidget(self.pylsl_info_text)
        
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # Real-time visualization group
        visualization_group = QGroupBox("Real-time EEG Visualization")
        visualization_layout = QVBoxLayout()
        self.pylsl_plot_canvas = PlotCanvas(self, width=10, height=12, dpi=80)
        visualization_layout.addWidget(self.pylsl_plot_canvas)
        
        viz_controls_layout = QHBoxLayout()
        self.pylsl_channel_display = QLabel("Channels: 0")
        self.pylsl_sample_rate = QLabel("Sample Rate: 0 Hz")
        self.pylsl_buffer_size = QLabel("Buffer: 0 samples")
        viz_controls_layout.addWidget(self.pylsl_channel_display)
        viz_controls_layout.addWidget(self.pylsl_sample_rate)
        viz_controls_layout.addWidget(self.pylsl_buffer_size)
        visualization_layout.addLayout(viz_controls_layout)        # Annotation controls (only visible in recording mode)
        self.annotation_controls_layout = QHBoxLayout()
        self.record_left_btn = QPushButton("Record: Left (T1) - 400 samples")
        self.record_right_btn = QPushButton("Record: Right (T2) - 400 samples")
        self.record_left_btn.setEnabled(False)
        self.record_right_btn.setEnabled(False)
        
        self.annotation_controls_layout.addWidget(self.record_left_btn)
        self.annotation_controls_layout.addWidget(self.record_right_btn)
        visualization_layout.addLayout(self.annotation_controls_layout)
        
        # Annotation status
        self.annotation_status_label = QLabel("Annotation Status: None")
        self.annotation_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        visualization_layout.addWidget(self.annotation_status_label)
        
        # Patient status indicator
        self.patient_status_label = QLabel("Patient: None selected")
        self.patient_status_label.setStyleSheet("padding: 5px; background-color: #fff3cd; border: 1px solid #ffeaa7; font-weight: bold;")
        visualization_layout.addWidget(self.patient_status_label)
        
        visualization_group.setLayout(visualization_layout)
        layout.addWidget(visualization_group)
        
        layout.addStretch()        # Initialize PyLSL variables
        self.pylsl_inlet = None
        self.pylsl_buffer = None
        self.pylsl_time_buffer = None
        self.pylsl_timer = QTimer(self)
        self.pylsl_timer.timeout.connect(self.update_pylsl_plot)
        self.current_sample_rate = 125  # Default sample rate
        
        # Stream monitoring variables
        self.last_sample_time = None
        self.stream_timeout_threshold = 3.0  # seconds
        self.consecutive_empty_pulls = 0
        self.max_consecutive_empty_pulls = 10  # ~0.4 seconds at 25fps
          # CSV recording variables
        self.csv_file = None
        self.csv_writer = None
        self.recording_folder = None
        self.sample_index = 0
        self.recording_start_time = None          # Annotation tracking variables
        self.current_annotation = ""
        self.annotation_samples_remaining = 0
        self.annotation_duration_samples = 400  # Changed from 600 to 400 samples
        self.add_t0_next_sample = False  # Flag to add T0 after annotation window ends
          # Connect signals
        # Mode switching
        self.mode_recording_btn.clicked.connect(self.switch_to_recording_mode)
        self.mode_inference_btn.clicked.connect(self.switch_to_inference_mode)
        
        # Recording mode signals
        self.select_folder_btn.clicked.connect(self.select_recording_folder)
        self.record_left_btn.clicked.connect(lambda: self.start_annotation("T1"))
        self.record_right_btn.clicked.connect(lambda: self.start_annotation("T2"))
          # Inference mode signals
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        self.clear_model_btn.clicked.connect(self.clear_selected_model)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        # Common signals
        self.pylsl_start_btn.clicked.connect(self.start_pylsl_stream)
        self.pylsl_stop_btn.clicked.connect(self.stop_pylsl_stream)
        self.pylsl_refresh_btn.clicked.connect(self.refresh_pylsl_streams)
        
        # Initialize UI
        self.refresh_pylsl_streams()  # Auto-refresh streams on load
        # Initialize with no model selected
        print("Model selection interface initialized")

    # ===== MODE MANAGEMENT FUNCTIONS =====
    
    def switch_to_recording_mode(self):
        """Switch to recording mode."""
        self.current_mode = "recording"
        self.mode_recording_btn.setChecked(True)
        self.mode_inference_btn.setChecked(False)
        self.mode_status_label.setText("Mode: Recording")
        self.mode_status_label.setStyleSheet("padding: 5px; background-color: #e6f3ff; border: 1px solid #0066cc; font-weight: bold;")
        
        # Show recording controls, hide inference controls
        self.recording_setup_group.setVisible(True)
        self.inference_setup_group.setVisible(False)
        
        # Update button texts
        self.pylsl_start_btn.setText("Start LSL Stream & Recording")
        self.pylsl_stop_btn.setText("Stop Stream & Recording")
        
        # Show annotation controls
        for i in range(self.annotation_controls_layout.count()):
            widget = self.annotation_controls_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(True)
        
        # Update start button availability
        if hasattr(self, 'pylsl_info_text'):
            self.refresh_pylsl_streams()
        
        print("Switched to Recording Mode")
    
    def switch_to_inference_mode(self):
        """Switch to inference mode."""
        self.current_mode = "inference"
        self.mode_recording_btn.setChecked(False)
        self.mode_inference_btn.setChecked(True)
        self.mode_status_label.setText("Mode: Inference")
        self.mode_status_label.setStyleSheet("padding: 5px; background-color: #e6ffe6; border: 1px solid #00cc00; font-weight: bold;")
        
        # Show inference controls, hide recording controls
        self.recording_setup_group.setVisible(False)
        self.inference_setup_group.setVisible(True)
        
        # Update button texts
        self.pylsl_start_btn.setText("Start LSL Stream & Inference")
        self.pylsl_stop_btn.setText("Stop Stream & Inference")
        
        # Hide annotation controls
        for i in range(self.annotation_controls_layout.count()):
            widget = self.annotation_controls_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(False)
        
        # Update start button availability
        if hasattr(self, 'pylsl_info_text'):
            self.refresh_pylsl_streams()
        
        print("Switched to Inference Mode")
      # ===== MODEL MANAGEMENT FUNCTIONS =====
    
    def browse_model_file(self):
        """Open file dialog to browse and select a model file."""
        if not MODEL_IMPORTS_AVAILABLE:
            QMessageBox.warning(self, "Model Imports Error", "Model imports are not available. Cannot load models for inference.")
            return
        
        # Start from models directory if it exists, otherwise home directory
        models_base_dir = os.path.join(project_root, "models")
        start_dir = models_base_dir if os.path.exists(models_base_dir) else os.path.expanduser("~")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            start_dir,
            "PyTorch Model Files (*.pt *.pth);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_model_path = file_path
            
            # Update the display label
            model_name = os.path.basename(file_path)
            model_dir = os.path.dirname(file_path)
            display_text = f"{model_name}\nLocation: {model_dir}"
            
            self.selected_model_label.setText(display_text)
            self.selected_model_label.setStyleSheet("padding: 5px; background-color: #e6f3ff; border: 1px solid #0066cc; font-family: monospace;")
            
            print(f"Selected model: {self.selected_model_path}")
            
            # Update start button availability
            if hasattr(self, 'pylsl_info_text'):
                self.refresh_pylsl_streams()
        else:
            print("Model selection cancelled")
    
    def clear_selected_model(self):
        """Clear the currently selected model."""
        self.selected_model_path = None
        self.selected_model_label.setText("No model selected")
        self.selected_model_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; font-family: monospace;")
        
        print("Model selection cleared")
        
        # Update start button availability
        if hasattr(self, 'pylsl_info_text'):
            self.refresh_pylsl_streams()
    
    def load_model_for_inference(self, n_channels, sample_rate):
        """Load the selected model for inference."""
        if not self.selected_model_path or not MODEL_IMPORTS_AVAILABLE:
            return False
        
        try:
            # Create model instance with appropriate parameters
            # Using default parameters - adjust as needed based on your training
            n_times = self.inference_window_size  # 400 samples
            
            self.loaded_model = EEGInceptionERPModel(
                n_chans=n_channels,
                n_outputs=2,  # Left vs Right hand
                n_times=n_times,
                sfreq=sample_rate
            )
            
            # Load model weights
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.selected_model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.loaded_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.loaded_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.loaded_model.load_state_dict(checkpoint)
            else:
                self.loaded_model.load_state_dict(checkpoint)
            
            self.loaded_model.eval()
            
            # Create inference processor
            self.inference_processor = RealTimeInferenceProcessor(
                model=self.loaded_model,
                n_channels=n_channels,
                sample_rate=sample_rate,
                window_size=self.inference_window_size,
                filter_enabled=True
            )
            
            print(f"Successfully loaded model: {os.path.basename(self.selected_model_path)}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load model: {str(e)}")
            self.loaded_model = None
            self.inference_processor = None
            return False
    
    def update_threshold(self, value):
        """Update confidence threshold from slider."""
        self.confidence_threshold = value / 100.0  # Convert to 0.0-1.0 range
        self.threshold_label.setText(f"{self.confidence_threshold:.2f}")
    
    def process_inference(self, samples):
        """Process samples for inference and update UI."""
        if not self.inference_processor:
            return
        
        try:
            # Run inference
            result = self.inference_processor.predict(samples)
            
            if result['status'] == 'success':
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    display_prediction = "UNCERTAIN"
                    color = "#ffcc99"  # Orange for uncertain
                    border_color = "#ff9900"
                else:
                    display_prediction = prediction.upper().replace('_', ' ')
                    if prediction == "left_hand":
                        color = "#ccffcc"  # Light green for left
                        border_color = "#00aa00"
                    else:  # right_hand
                        color = "#ccccff"  # Light blue for right
                        border_color = "#0000aa"
                
                # Update inference result display
                self.inference_result_label.setText(
                    f"Prediction: {display_prediction} (Confidence: {confidence:.2f})"
                )
                self.inference_result_label.setStyleSheet(
                    f"padding: 10px; background-color: {color}; border: 2px solid {border_color}; "
                    f"font-size: 14px; font-weight: bold;"
                )
                
                # Print to console for debugging
                left_prob = result['class_probabilities']['left_hand']
                right_prob = result['class_probabilities']['right_hand']
                print(f"Inference: {display_prediction} | Left: {left_prob:.3f}, Right: {right_prob:.3f} | Confidence: {confidence:.3f}")
                
            elif result['status'] == 'insufficient_data':
                self.inference_result_label.setText("Prediction: Collecting data...")
                self.inference_result_label.setStyleSheet(
                    "padding: 10px; background-color: #f0f0f0; border: 2px solid #ccc; "
                    "font-size: 14px; font-weight: bold;"
                )
            else:
                self.inference_result_label.setText(f"Prediction: Error - {result.get('message', 'Unknown error')}")
                self.inference_result_label.setStyleSheet(
                    "padding: 10px; background-color: #ffcccc; border: 2px solid #cc0000; "
                    "font-size: 14px; font-weight: bold;"
                )
                
        except Exception as e:
            print(f"Error during inference processing: {e}")
            self.inference_result_label.setText("Prediction: Processing Error")
            self.inference_result_label.setStyleSheet(
                "padding: 10px; background-color: #ffcccc; border: 2px solid #cc0000; "
                "font-size: 14px; font-weight: bold;"
            )

    def select_recording_folder(self):
        """Select folder where CSV recordings will be saved."""
        folder = QFileDialog.getExistingDirectory(self, "Select Recording Folder")
        if folder:
            self.recording_folder = folder
            self.selected_folder_label.setText(f"Custom folder: {folder}")
            self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #ccffcc; border: 1px solid #00aa00;")
            # Reset button text when custom folder is selected
            self.select_folder_btn.setText("Select Recording Folder")
            # Enable start button if streams are available
            self.refresh_pylsl_streams()
        else:
            # Only reset if no patient folder is set
            if not hasattr(self.main_window, 'current_patient_id') or not self.main_window.current_patient_id:
                self.recording_folder = None
                self.selected_folder_label.setText("No folder selected")
                self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
                self.pylsl_start_btn.setEnabled(False)

    def refresh_pylsl_streams(self):
        """
        Refreshes the list of available LSL streams.
        Queries for LSL streams (specifically EEG type) and updates the
        information text area. Enables the start button if EEG streams are found.
        """
        if not PYLSL_AVAILABLE:
            return
        try:
            streams = resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            info_text = f"Found {len(streams)} total streams, {len(eeg_streams)} EEG streams:\\n"
            if eeg_streams:
                for i, stream in enumerate(eeg_streams):
                    info_text += f"  {i+1}. {stream.name()} - {stream.channel_count()} channels @ {stream.nominal_srate()} Hz\\n"
            else:
                info_text += "No EEG streams found. Make sure your EEG device is streaming to LSL."
            self.pylsl_info_text.setPlainText(info_text)            # Enable start button based on current mode
            if self.current_mode == "recording":
                self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0 and self.recording_folder is not None)
            else:  # inference mode
                self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0 and self.selected_model_path is not None)
        except Exception as e:
            self.pylsl_info_text.setPlainText(f"Error refreshing streams: {str(e)}")
            self.pylsl_start_btn.setEnabled(False)

    def set_patient_folder(self, patient_id, patient_data):
        """
        Set the recording folder automatically based on selected patient.
        
        Args:
            patient_id (str): The ID of the selected patient
            patient_data (dict): Patient information dictionary
        """
        if not patient_id:
            return
            
        # Create patient-specific folder structure
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        patient_folder = os.path.join(project_root, "patient_data", patient_id)
        
        # Create organized subfolder structure
        self._create_patient_folder_structure(patient_folder)
        
        # Set EEG recordings as the primary recording folder
        eeg_recordings_folder = os.path.join(patient_folder, "eeg_recordings")
        self.recording_folder = eeg_recordings_folder
        
        # Update UI elements
        patient_name = patient_data.get('name', 'Unknown')
        self.selected_folder_label.setText(f"Patient: {patient_name} ({patient_id}) - EEG Recordings")
        self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #e6f3ff; border: 1px solid #0066cc;")
        
        # Update button text to show it's automatically set
        self.select_folder_btn.setText("Change Recording Folder")
        
        # Update patient status indicator
        self.patient_status_label.setText(f"Patient: {patient_name} ({patient_id}) - Auto-configured")
        self.patient_status_label.setStyleSheet("padding: 5px; background-color: #d4edda; border: 1px solid #28a745; font-weight: bold;")
        
        # Enable start button if streams are available
        self.refresh_pylsl_streams()
        
        print(f"PylslTab: Auto-configured recording folder for patient {patient_id}: {eeg_recordings_folder}")

    def _create_patient_folder_structure(self, patient_folder):
        """
        Create organized folder structure for patient data.
        
        Args:
            patient_folder (str): Base patient folder path
        """
        subfolders = [
            "eeg_recordings",     # Raw EEG CSV files
            "processed_data",     # Preprocessed data
            "models",            # Trained models for this patient
            "reports",           # Analysis reports and visualizations
            "sessions"           # Session-specific data
        ]
        
        for subfolder in subfolders:
            folder_path = os.path.join(patient_folder, subfolder)
            os.makedirs(folder_path, exist_ok=True)
            
        print(f"Created patient folder structure: {patient_folder}")

    def clear_patient_folder(self):
        """
        Clear the patient-specific folder configuration.
        This is called when no patient is selected.
        """
        self.recording_folder = None
        self.selected_folder_label.setText("No folder selected")
        self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.select_folder_btn.setText("Select Recording Folder")
        self.patient_status_label.setText("Patient: None selected")
        self.patient_status_label.setStyleSheet("padding: 5px; background-color: #fff3cd; border: 1px solid #ffeaa7; font-weight: bold;")
        self.pylsl_start_btn.setEnabled(False)
        
        print("PylslTab: Cleared patient folder configuration")

    def get_patient_status_info(self):
        """
        Get information about current patient folder configuration.
        
        Returns:
            dict: Status information about patient configuration
        """
        has_patient = (hasattr(self.main_window, 'current_patient_id') and 
                      self.main_window.current_patient_id)
        
        return {
            'has_patient': has_patient,
            'patient_id': getattr(self.main_window, 'current_patient_id', None),
            'recording_folder': self.recording_folder,
            'folder_type': 'patient' if has_patient and self.recording_folder and 'patient_data' in str(self.recording_folder) else 'custom'
        }

    def start_csv_recording(self, num_channels):
        """Initialize CSV recording with OpenBCI format."""
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(self.recording_folder, f"eeg_recording_{timestamp_str}.csv")
            
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
              # Write OpenBCI-style headers
            self.csv_file.write("%OpenBCI Raw EXG Data\n")
            self.csv_file.write(f"%Number of channels = {num_channels}\n")
            self.csv_file.write(f"%Sample Rate = {int(self.current_sample_rate)} Hz\n")
            self.csv_file.write("%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n")
            
            # Write column headers
            headers = ["Sample Index"]
            for i in range(num_channels):
                headers.append(f"EXG Channel {i}")
            
            # Add placeholder columns to match OpenBCI format
            headers.extend(["Accel Channel 0", "Accel Channel 1", "Accel Channel 2"])
            for i in range(7):  # Other columns
                headers.append("Other")
            headers.extend(["Analog Channel 0", "Analog Channel 1", "Analog Channel 2"])
            headers.append("Timestamp")
            headers.append("Other")
            headers.append("Timestamp (Formatted)")
            headers.append("Annotations")
            self.csv_writer.writerow(headers)
            
            self.sample_index = 1
            self.recording_start_time = datetime.datetime.now()
            
            # Reset annotation tracking variables
            self.current_annotation = ""
            self.annotation_samples_remaining = 0
            self.add_t0_next_sample = False
            
            # Add T0 annotation for recording start
            self.add_annotation("T0")
            
            QMessageBox.information(self, "Recording Started", f"CSV recording started: {csv_filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start CSV recording: {str(e)}")
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None    
    
    def write_samples_to_csv(self, samples):
        """Write samples to CSV file in OpenBCI format."""
        if not self.csv_writer or not samples:
            return
            
        try:
            current_time = datetime.datetime.now()
            for sample in samples:
                row = [self.sample_index]
                
                # Add EXG channel data with proper formatting
                for value in sample:
                    # Format to match OpenBCI precision (max 4 decimal places, remove trailing zeros)
                    if isinstance(value, (int, float)):
                        # Round to 4 decimal places and remove trailing zeros
                        formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
                        # Convert back to float if it's just a number, otherwise keep as string
                        try:
                            formatted_value = float(formatted_value) if '.' in formatted_value else int(formatted_value)
                        except ValueError:
                            formatted_value = 0
                    else:
                        formatted_value = value
                    row.append(formatted_value)
                
                # Add placeholder data for other columns to match OpenBCI format
                num_eeg_channels = len(sample)
                
                # Accel channels (3)
                row.extend([0, 0, 0])
                
                # Other columns (7)
                row.extend([0, 0, 0, 0, 0, 0, 0])
                
                # Analog channels (3)
                row.extend([0, 0, 0])
                  # Timestamp (relative to recording start in seconds)
                elapsed_seconds = (current_time - self.recording_start_time).total_seconds()
                # Format timestamp to match OpenBCI precision
                formatted_timestamp = f"{elapsed_seconds:.4f}".rstrip('0').rstrip('.')
                try:
                    formatted_timestamp = float(formatted_timestamp) if '.' in formatted_timestamp else int(formatted_timestamp)
                except ValueError:
                    formatted_timestamp = 0
                row.append(formatted_timestamp)
                
                # Other
                row.append(0)
                
                # Formatted timestamp
                row.append(current_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
                  # Annotations - handle annotation logic
                annotation_text = ""
                
                # Check if we need to add T1/T2 annotation (only on first sample of window)
                if self.annotation_samples_remaining == self.annotation_duration_samples:
                    # This is the first sample of the annotation window
                    annotation_text = self.current_annotation
                    self.annotation_samples_remaining -= 1
                elif self.annotation_samples_remaining > 0:
                    # We're in the middle of annotation window - no annotation text
                    self.annotation_samples_remaining -= 1
                    if self.annotation_samples_remaining == 0:
                        # Window just ended - schedule T0 for next sample
                        self.add_t0_next_sample = True
                        self.current_annotation = ""
                        self.annotation_status_label.setText("Annotation Status: None")
                        self.annotation_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
                    else:
                        # Update status with remaining samples
                        self.annotation_status_label.setText(f"Annotation Status: {self.current_annotation} - {self.annotation_samples_remaining} samples remaining")
                elif hasattr(self, 'add_t0_next_sample') and self.add_t0_next_sample:
                    # Add T0 annotation to the sample after the window ended
                    annotation_text = "T0"
                    self.add_t0_next_sample = False
                
                row.append(annotation_text)
                
                self.csv_writer.writerow(row)
                self.sample_index += 1
                
            self.csv_file.flush()  # Ensure data is written to disk
            
        except Exception as e:
            print(f"Error writing samples to CSV: {str(e)}")

    def start_annotation(self, annotation):
        """Start a new annotation (T1 or T2) that will continue for 600 samples."""
        if annotation in ["T1", "T2"]:
            self.current_annotation = annotation
            self.annotation_samples_remaining = self.annotation_duration_samples
            self.annotation_status_label.setText(f"Annotation Status: {annotation} - {self.annotation_samples_remaining} samples remaining")
            self.annotation_status_label.setStyleSheet("padding: 5px; background-color: #ffffcc; border: 1px solid #ffaa00;")
            print(f"Started annotation: {annotation} for {self.annotation_duration_samples} samples")

    def add_annotation(self, annotation):
        """Add annotation (T0, T1, T2) to CSV file."""
        if not self.csv_writer:
            return
            
        try:
            current_time = datetime.datetime.now()
            
            # Create a row with the annotation
            row = [self.sample_index]
            
            # Get current number of channels from the last buffer sample
            if self.pylsl_buffer and len(self.pylsl_buffer) > 0:
                last_sample = list(self.pylsl_buffer)[-1]
                num_channels = len(last_sample)
                # Use zeros for EXG channels in annotation row
                row.extend([0] * num_channels)
            else:
                # Default to 16 channels if no buffer data
                row.extend([0] * 16)
            
            # Add placeholder data for other columns
            row.extend([0, 0, 0])  # Accel channels
            row.extend([0, 0, 0, 0, 0, 0, 0])  # Other columns
            row.extend([0, 0, 0])  # Analog channels
            
            # Timestamp
            elapsed_seconds = (current_time - self.recording_start_time).total_seconds()
            row.append(elapsed_seconds)
            
            # Other
            row.append(0)
            
            # Formatted timestamp
            row.append(current_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            
            # Annotation
            row.append(annotation)
            
            self.csv_writer.writerow(row)
            self.sample_index += 1
            self.csv_file.flush()
            
            print(f"Added annotation: {annotation} at sample {self.sample_index - 1}")
            
        except Exception as e:
            print(f"Error adding annotation: {str(e)}")

    def stop_csv_recording(self):
        """Stop CSV recording and close file."""
        if self.csv_file:
            try:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
                QMessageBox.information(self, "Recording Stopped", "CSV recording has been saved and stopped.")
            except Exception as e:                QMessageBox.critical(self, "Recording Error", f"Error stopping CSV recording: {str(e)}")

    def start_pylsl_stream(self):
        """
        Starts an LSL EEG stream for either recording or inference mode.
        """
        if not PYLSL_AVAILABLE:
            QMessageBox.warning(self, "PyLSL Not Available", "PyLSL is not installed. Please install it with: pip install pylsl")
            return
            
        # Check mode-specific requirements
        if self.current_mode == "recording":
            if not self.recording_folder:
                QMessageBox.warning(self, "No Folder Selected", "Please select a recording folder before starting the stream.")
                return
        else:  # inference mode
            if not self.selected_model_path:
                QMessageBox.warning(self, "No Model Selected", "Please select a model before starting inference.")
                return
            
        try:
            streams = resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            if not eeg_streams:
                QMessageBox.warning(self, "No EEG Streams", "No EEG streams found. Make sure your device is streaming.")
                return
              # Connect to the first EEG stream found with MINIMAL buffer for instant response
            self.pylsl_inlet = StreamInlet(eeg_streams[0], max_buflen=8)  # Extremely aggressive: only 8 samples buffer
            info = self.pylsl_inlet.info()
            n_channels = info.channel_count()
            
            # Use nominal_srate, ensure it's a positive value
            sample_rate = info.nominal_srate()
            if sample_rate <= 0:
                print(f"Warning: Stream {info.name()} reported nominal_srate of {sample_rate}. Using default 125 Hz.")
                self.current_sample_rate = 125.0
            else:
                self.current_sample_rate = float(sample_rate)

            # Reduced buffer size for faster response
            buffer_size = 250  # Reduced from 400
            
            self.pylsl_buffer = deque(maxlen=buffer_size)
            self.pylsl_time_buffer = deque(maxlen=buffer_size)
              # Reset stream monitoring variables
            self.last_sample_time = None
            self.consecutive_empty_pulls = 0
            
            # Mode-specific initialization
            if self.current_mode == "recording":
                # Start CSV recording
                self.start_csv_recording(n_channels)
                message_suffix = "Recording"
                folder_text = f"\\nRecording to: {self.recording_folder}"
            else:  # inference mode
                # Load model for inference
                if not self.load_model_for_inference(n_channels, self.current_sample_rate):
                    return  # Error message already shown in load_model_for_inference
                message_suffix = "Inference"
                folder_text = f"\\nUsing model: {os.path.basename(self.selected_model_path)}"
                self.inference_result_label.setText("Prediction: Collecting data...")
                self.inference_result_label.setStyleSheet(
                    "padding: 10px; background-color: #f0f0f0; border: 2px solid #ccc; "
                    "font-size: 14px; font-weight: bold;"
                )
            
            self.pylsl_channel_display.setText(f"Channels: {n_channels}")
            self.pylsl_sample_rate.setText(f"Sample Rate: {self.current_sample_rate:.2f} Hz")
            self.pylsl_buffer_size.setText(f"Buffer: {buffer_size} samples")
            self.pylsl_status_label.setText(f"Status: Connected & {message_suffix}")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            self.pylsl_start_btn.setEnabled(False)
            self.pylsl_stop_btn.setEnabled(True)
            
            # Only enable annotation buttons in recording mode
            if self.current_mode == "recording":
                self.record_left_btn.setEnabled(True)
                self.record_right_btn.setEnabled(True)
            
            # Faster timer for more responsive updates
            self.pylsl_timer.start(int(1000 / 30))  # Increased to 30 FPS from 25 FPS
            
            QMessageBox.information(self, "Stream Started", 
                                  f"Successfully connected to EEG stream: {info.name()}\\n"
                                  f"Channels: {n_channels}\\n"
                                  f"Sample Rate: {self.current_sample_rate:.2f} Hz\\n"
                                  f"Mode: {message_suffix}{folder_text}")
        except Exception as e:            QMessageBox.critical(self, "Stream Error", f"Failed to start stream: {str(e)}\\n{traceback.format_exc()}")

    def stop_pylsl_stream(self):
        """
        Stops the currently active LSL stream and recording/inference.
        """
        try:
            self.pylsl_timer.stop()
            
            # Aggressively clear any remaining buffered data
            if self.pylsl_inlet:
                try:
                    # Flush any remaining data in the inlet with short timeout
                    remaining_samples, _ = self.pylsl_inlet.pull_chunk(timeout=0.1, max_samples=1000)
                    if remaining_samples:
                        print(f"Flushed {len(remaining_samples)} remaining samples from inlet buffer")
                except:
                    pass  # Ignore errors during flush
            
            # Mode-specific cleanup
            if self.current_mode == "recording":
                # Stop CSV recording
                self.stop_csv_recording()
                self.annotation_status_label.setText("Annotation Status: None")
                self.annotation_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
            else:  # inference mode
                # Clean up inference processor
                self.inference_processor = None
                self.loaded_model = None
                self.inference_result_label.setText("Prediction: Stopped")
                self.inference_result_label.setStyleSheet(
                    "padding: 10px; background-color: #f0f0f0; border: 2px solid #ccc; "
                    "font-size: 14px; font-weight: bold;"
                )
            
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream()
            self.pylsl_inlet = None
            
            # Clear buffers immediately
            if self.pylsl_buffer:
                self.pylsl_buffer.clear()
            if self.pylsl_time_buffer:
                self.pylsl_time_buffer.clear()
            self.pylsl_buffer = None
            self.pylsl_time_buffer = None
            
            # Reset stream monitoring variables
            self.last_sample_time = None
            self.consecutive_empty_pulls = 0
            
            # Reset annotation tracking
            self.current_annotation = ""
            self.annotation_samples_remaining = 0
            self.add_t0_next_sample = False            
            self.pylsl_status_label.setText("Status: Disconnected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
            
            self.pylsl_plot_canvas.clear_plot()
            
            # Update button availability based on mode
            if self.current_mode == "recording":
                self.pylsl_start_btn.setEnabled(self.recording_folder is not None)
            else:  # inference mode
                self.pylsl_start_btn.setEnabled(self.selected_model_path is not None)
            
            self.pylsl_stop_btn.setEnabled(False)
            self.record_left_btn.setEnabled(False)
            self.record_right_btn.setEnabled(False)
            
            mode_text = "Recording" if self.current_mode == "recording" else "Inference"
            QMessageBox.information(self, "Stream Stopped", f"LSL stream disconnected and {mode_text.lower()} stopped.")
        except Exception as e:            QMessageBox.critical(self, "Stop Error", f"Error stopping stream: {str(e)}\\n{traceback.format_exc()}")

    def update_pylsl_plot(self):
        """
        Pulls new data from the LSL stream, updates the plot, and processes data based on current mode.
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None:
            return
        try:
            # Pull samples with short timeout for faster response
            samples_to_pull = int(self.current_sample_rate * (self.pylsl_timer.interval() / 1000.0))
            # Use very short timeout to prevent blocking
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.01, max_samples=max(1, samples_to_pull))
            
            current_time = time.time()
            
            if samples:
                # Reset empty pull counter when we get data
                self.consecutive_empty_pulls = 0
                self.last_sample_time = current_time
                
                # Add to buffer for visualization
                for sample_group in samples:
                    self.pylsl_buffer.append(sample_group)
                
                # Mode-specific processing
                if self.current_mode == "recording":
                    # Write samples to CSV
                    self.write_samples_to_csv(samples)
                    status_text = "Recording"
                else:  # inference mode
                    # Process samples for inference
                    self.process_inference(np.array(samples))
                    status_text = "Inference"
                
                # Update visualization
                self._update_visualization_plot()
                
                # Update status to show active streaming
                self.pylsl_status_label.setText(f"Status: Connected & {status_text}")
                self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
                
            else:
                # No samples received - check for stream timeout
                self.consecutive_empty_pulls += 1
                
                # Check if stream has timed out
                stream_timed_out = False
                if self.last_sample_time is not None:
                    time_since_last_sample = current_time - self.last_sample_time
                    if time_since_last_sample > self.stream_timeout_threshold:
                        stream_timed_out = True
                
                # Update status based on timeout conditions
                if stream_timed_out or self.consecutive_empty_pulls > self.max_consecutive_empty_pulls:
                    if "No data" not in self.pylsl_status_label.text():
                        self.pylsl_status_label.setText("Status: Connected but no data received")
                        self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffffcc; border: 1px solid #ffaa00;")
                        print(f"Stream timeout detected - no data for {time_since_last_sample:.1f}s")
                
                # If no data for very long time, offer to stop automatically
                if self.last_sample_time is not None:
                    time_since_last_sample = current_time - self.last_sample_time
                    if time_since_last_sample > 10.0:  # 10 seconds without data
                        self.pylsl_status_label.setText("Status: Stream appears disconnected - consider stopping")
                        self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffdddd; border: 1px solid #ff6666;")
                
        except Exception as e:
            print(f"DEBUG: Plot update error: {e}")
            # Check if it's a stream-related error that suggests disconnection
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['timeout', 'connection', 'stream', 'inlet']):
                self.pylsl_status_label.setText("Status: Stream error detected")
                self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc; border: 1px solid #ff0000;")
            traceback.print_exc()

    def _update_visualization_plot(self):
        """
        Updates the EEG data plot with the current buffer content.
        """
        if self.pylsl_buffer and len(self.pylsl_buffer) > 0:
            try:
                current_buffer_list = list(self.pylsl_buffer)
                if not current_buffer_list:
                    self.pylsl_plot_canvas.clear_plot()
                    return

                # Convert to numpy array
                plot_data_samples_channels = np.array(current_buffer_list)

                if plot_data_samples_channels.size == 0:
                    self.pylsl_plot_canvas.clear_plot()
                    return                # Transpose to (num_channels, num_samples)
                plot_data_channels_samples = plot_data_samples_channels.T
                
                max_channels_to_display = 16
                
                if plot_data_channels_samples.shape[0] > max_channels_to_display:
                    data_to_plot = plot_data_channels_samples[:max_channels_to_display, :]
                else:
                    data_to_plot = plot_data_channels_samples
                
                if data_to_plot.size > 0:
                    mode_text = "Recording Active" if self.current_mode == "recording" else "Inference Active"
                    self.pylsl_plot_canvas.plot(data_to_plot, title=f"Real-time EEG Data ({mode_text})")
                else:
                    self.pylsl_plot_canvas.clear_plot()
            except Exception as e:
                print(f"Error during plot data preparation: {str(e)}")
                traceback.print_exc()
                self.pylsl_plot_canvas.clear_plot()
        else:
            self.pylsl_plot_canvas.clear_plot()

    def closeEvent(self, event):
        """
        Handles the tab being closed or the application quitting.
        Ensures the LSL stream is stopped and CSV recording is saved.
        """
        self.stop_pylsl_stream()
        super().closeEvent(event)
