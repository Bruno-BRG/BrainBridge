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
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QHBoxLayout,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import QTimer
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

class PylslTab(QWidget):
    """
    Manages the Pylsl Tab in the main GUI application.

    This class is responsible for setting up the UI elements related to LSL stream
    interaction, handling user actions (e.g., start/stop stream),
    and managing the LSL data inlet and plotting timer for real-time visualization.
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
            return

        # Simplified UI setup directly in __init__
        self._setup_pylsl_ui(layout)
        
        self.setLayout(layout)

    def _setup_pylsl_ui(self, layout):
        """
        Sets up the UI elements for PyLSL interaction.

        Args:
            layout (QVBoxLayout): The main QVBoxLayout of the PylslTab.
        """
        # Recording setup group
        recording_setup_group = QGroupBox("Recording Setup")
        recording_setup_layout = QVBoxLayout()
        
        folder_selection_layout = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Recording Folder")
        self.selected_folder_label = QLabel("No folder selected")
        self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        folder_selection_layout.addWidget(self.select_folder_btn)
        folder_selection_layout.addWidget(self.selected_folder_label)
        recording_setup_layout.addLayout(folder_selection_layout)
        
        recording_setup_group.setLayout(recording_setup_layout)
        layout.addWidget(recording_setup_group)
        
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
        visualization_layout.addLayout(viz_controls_layout)

        # Annotation controls
        annotation_controls_layout = QHBoxLayout()
        self.record_left_btn = QPushButton("Record: Left (T1) - 600 samples")
        self.record_right_btn = QPushButton("Record: Right (T2) - 600 samples")
        self.record_left_btn.setEnabled(False)
        self.record_right_btn.setEnabled(False)
        
        annotation_controls_layout.addWidget(self.record_left_btn)
        annotation_controls_layout.addWidget(self.record_right_btn)
        visualization_layout.addLayout(annotation_controls_layout)
        
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
        self.recording_start_time = None
          # Annotation tracking variables
        self.current_annotation = ""
        self.annotation_samples_remaining = 0
        self.annotation_duration_samples = 600  # Duration for T1 and T2 annotations
        self.add_t0_next_sample = False  # Flag to add T0 after annotation window ends
          # Connect signals
        self.select_folder_btn.clicked.connect(self.select_recording_folder)
        self.pylsl_start_btn.clicked.connect(self.start_pylsl_stream)
        self.pylsl_stop_btn.clicked.connect(self.stop_pylsl_stream)
        self.pylsl_refresh_btn.clicked.connect(self.refresh_pylsl_streams)
        self.record_left_btn.clicked.connect(lambda: self.start_annotation("T1"))
        self.record_right_btn.clicked.connect(lambda: self.start_annotation("T2"))
        
        self.refresh_pylsl_streams()  # Auto-refresh streams on load

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
            self.pylsl_info_text.setPlainText(info_text)
            # Enable start button only if both folder is selected and EEG streams are available
            self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0 and self.recording_folder is not None)
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
            except Exception as e:
                QMessageBox.critical(self, "Recording Error", f"Error stopping CSV recording: {str(e)}")    
    
    def start_pylsl_stream(self):
        """
        Starts an LSL EEG stream and begins CSV recording.
        """
        if not PYLSL_AVAILABLE:
            QMessageBox.warning(self, "PyLSL Not Available", "PyLSL is not installed. Please install it with: pip install pylsl")
            return
            
        if not self.recording_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select a recording folder before starting the stream.")
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
            
            # Start CSV recording
            self.start_csv_recording(n_channels)
            
            self.pylsl_channel_display.setText(f"Channels: {n_channels}")
            self.pylsl_sample_rate.setText(f"Sample Rate: {self.current_sample_rate:.2f} Hz")
            self.pylsl_buffer_size.setText(f"Buffer: {buffer_size} samples")
            self.pylsl_status_label.setText("Status: Connected & Recording")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            self.pylsl_start_btn.setEnabled(False)
            self.pylsl_stop_btn.setEnabled(True)
            self.record_left_btn.setEnabled(True)
            self.record_right_btn.setEnabled(True)
            
            # Faster timer for more responsive updates
            self.pylsl_timer.start(int(1000 / 30))  # Increased to 30 FPS from 25 FPS
            
            QMessageBox.information(self, "Stream Started", 
                                  f"Successfully connected to EEG stream: {info.name()}\\n"
                                  f"Channels: {n_channels}\\n"
                                  f"Sample Rate: {self.current_sample_rate:.2f} Hz\\n"
                                  f"Recording to: {self.recording_folder}")
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", f"Failed to start stream: {str(e)}\\n{traceback.format_exc()}")    
    
    def stop_pylsl_stream(self):
        """
        Stops the currently active LSL stream and CSV recording.
        Includes aggressive buffer clearing for faster response.
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
            
            # Stop CSV recording first
            self.stop_csv_recording()
            
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
            
            # Reset annotation status label
            self.annotation_status_label.setText("Annotation Status: None")
            self.annotation_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
            
            self.pylsl_plot_canvas.clear_plot()
            
            self.pylsl_start_btn.setEnabled(self.recording_folder is not None)
            self.pylsl_stop_btn.setEnabled(False)
            self.record_left_btn.setEnabled(False)
            self.record_right_btn.setEnabled(False)
            
            QMessageBox.information(self, "Stream Stopped", "LSL stream disconnected and recording saved.")
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Error stopping stream: {str(e)}\\n{traceback.format_exc()}")

    def update_pylsl_plot(self):
        """
        Pulls new data from the LSL stream, updates the plot, and writes to CSV.
        Includes automatic stream timeout detection for faster response when stream stops.
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None:
            return
        try:
            # Pull samples with short timeout for faster response
            samples_to_pull = int(self.current_sample_rate * (self.pylsl_timer.interval() / 1000.0))
            # Use very short timeout to prevent blocking
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.01, max_samples=max(1, samples_to_pull))
            
            import time
            current_time = time.time()
            
            if samples:
                # Reset empty pull counter when we get data
                self.consecutive_empty_pulls = 0
                self.last_sample_time = current_time
                
                # Add to buffer for visualization
                for sample_group in samples:
                    self.pylsl_buffer.append(sample_group)
                
                # Write samples to CSV
                self.write_samples_to_csv(samples)
                
                # Update visualization
                self._update_visualization_plot()
                
                # Update status to show active streaming
                if "Recording" in self.pylsl_status_label.text():
                    self.pylsl_status_label.setText("Status: Connected & Recording")
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
                    return

                # Transpose to (num_channels, num_samples)
                plot_data_channels_samples = plot_data_samples_channels.T
                
                max_channels_to_display = 16
                
                if plot_data_channels_samples.shape[0] > max_channels_to_display:
                    data_to_plot = plot_data_channels_samples[:max_channels_to_display, :]
                else:
                    data_to_plot = plot_data_channels_samples
                
                if data_to_plot.size > 0:
                    self.pylsl_plot_canvas.plot(data_to_plot, title="Real-time EEG Data (Recording Active)")
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
