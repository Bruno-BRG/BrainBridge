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
        
        visualization_group.setLayout(visualization_layout)
        layout.addWidget(visualization_group)
        
        layout.addStretch()

        # Initialize PyLSL variables
        self.pylsl_inlet = None
        self.pylsl_buffer = None
        self.pylsl_time_buffer = None
        self.pylsl_timer = QTimer(self)
        self.pylsl_timer.timeout.connect(self.update_pylsl_plot)
        self.current_sample_rate = 125  # Default sample rate
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
            self.selected_folder_label.setText(f"Selected: {folder}")
            self.selected_folder_label.setStyleSheet("padding: 5px; background-color: #ccffcc; border: 1px solid #00aa00;")
            # Enable start button if streams are available
            self.refresh_pylsl_streams()
        else:
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

    def start_csv_recording(self, num_channels):
        """Initialize CSV recording with OpenBCI format."""
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(self.recording_folder, f"eeg_recording_{timestamp_str}.csv")
            
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write OpenBCI-style headers
            self.csv_file.write("%OpenBCI Raw EXG Data\\n")
            self.csv_file.write(f"%Number of channels = {num_channels}\\n")
            self.csv_file.write(f"%Sample Rate = {self.current_sample_rate} Hz\\n")
            self.csv_file.write("%Board = LSL_Stream\\n")
            
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
                
                # Add EXG channel data
                for value in sample:
                    row.append(value)
                
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
                row.append(elapsed_seconds)
                
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
            
            # Connect to the first EEG stream found
            self.pylsl_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            info = self.pylsl_inlet.info()
            n_channels = info.channel_count()
            
            # Use nominal_srate, ensure it's a positive value
            sample_rate = info.nominal_srate()
            if sample_rate <= 0:
                print(f"Warning: Stream {info.name()} reported nominal_srate of {sample_rate}. Using default 125 Hz.")
                self.current_sample_rate = 125.0
            else:
                self.current_sample_rate = float(sample_rate)

            # Buffer size set to display 400 samples
            buffer_size = 400
            
            self.pylsl_buffer = deque(maxlen=buffer_size)
            self.pylsl_time_buffer = deque(maxlen=buffer_size)
            
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
            
            self.pylsl_timer.start(int(1000 / 25))  # Target ~25 FPS plot updates
            
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
        """
        try:
            self.pylsl_timer.stop()
              # Stop CSV recording first
            self.stop_csv_recording()
            
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream()
            self.pylsl_inlet = None
            self.pylsl_buffer = None
            self.pylsl_time_buffer = None
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
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None:
            return
        try:
            # Pull samples
            samples_to_pull = int(self.current_sample_rate * (self.pylsl_timer.interval() / 1000.0))
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.0, max_samples=max(1, samples_to_pull))
            
            if samples:
                # Add to buffer for visualization
                for sample_group in samples:
                    self.pylsl_buffer.append(sample_group)
                
                # Write samples to CSV
                self.write_samples_to_csv(samples)
                
                # Update visualization
                self._update_visualization_plot()
                
        except Exception as e:
            print(f"DEBUG: Plot update error: {e}")
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
