"""
File:    pylsl_tab.py
Class:   PylslTab
Purpose: Provides the user interface tab for discovering, connecting to,
         visualizing, and recording EEG data streams using the Lab Streaming
         Layer (LSL) library. It allows users to refresh available LSL streams,
         start/stop a selected stream, view real-time data plots, and
         record/save the EEG data. Includes real-time inference capabilities
         for trained EEG models.
Author:  Bruno Rocha
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Requires PyLSL to be installed. If PyLSL is not available,
         the tab will display an error message.
         Optionally integrates with a StreamingWidget if available.
"""

import os
import sys
import numpy as np
import traceback
import torch
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QHBoxLayout, 
    QMessageBox, QFileDialog, QComboBox, QProgressBar, QSpinBox, QCheckBox, QFrame
)
from PyQt5.QtCore import QTimer, pyqtSignal
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .plot_canvas import PlotCanvas # Assuming PlotCanvas is in plot_canvas.py

# PyLSL imports
try:
    from pylsl import StreamInlet, resolve_streams
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    StreamInlet = None # Define for type hinting if not available
    resolve_streams = None

# Attempt to import StreamingWidget (optional)
try:
    from .StreamingWidget import StreamingWidget
except ImportError:
    try:
        from src.UI.StreamingWidget import StreamingWidget
    except ImportError:
        StreamingWidget = None

# Import model classes and filtering utilities
try:
    from src.model.eeg_inception_erp import EEGInceptionERPModel
    from src.model.eeg_it_net import EEGITNetModel
    from src.model.EEGFilter import EEGFilter
    from src.model.realtime_inference import RealTimeInferenceProcessor
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model classes not available: {e}")
    EEGInceptionERPModel = None
    EEGITNetModel = None
    EEGFilter = None
    RealTimeInferenceProcessor = None
    MODELS_AVAILABLE = False

class PylslTab(QWidget):
    """
    Manages the Pylsl Tab in the main GUI application.

    This class is responsible for setting up the UI elements related to LSL stream
    interaction, handling user actions (e.g., start/stop stream, record data),
    and managing the LSL data inlet and plotting timer.
    """
    def __init__(self, parent_main_window):
        """
        Initializes the PylslTab.

        Args:
            parent_main_window (QMainWindow): Reference to the main application window.
                                              Used for accessing shared resources or configurations if necessary.
        """
        super().__init__()
        self.main_window = parent_main_window # Reference to MainWindow if needed, though try to minimize direct access

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

        # Try to use StreamingWidget if available
        if StreamingWidget:
            try:
                self.streaming_widget_instance = StreamingWidget(self) # Or parent_main_window if it needs that context
                layout.addWidget(self.streaming_widget_instance)
                # If StreamingWidget handles everything, we might not need the manual setup below.
                # For now, assume StreamingWidget might fail or not be fully implemented, so proceed with manual setup as fallback.
                print("INFO: StreamingWidget loaded.")
                # If StreamingWidget is self-contained and replaces manual setup, you might return here.
            except Exception as e:
                print(f"Warning: StreamingWidget failed to load: {e}. Falling back to manual PyLSL setup.")
                self._setup_manual_pylsl_ui(layout)
        else:
            print("INFO: StreamingWidget not found. Using manual PyLSL setup.")
            self._setup_manual_pylsl_ui(layout)
        
        self.setLayout(layout)

    def _setup_manual_pylsl_ui(self, layout):
        """
        Sets up the UI elements for manual PyLSL interaction.

        This method is called if the StreamingWidget is not available or fails to load.
        It creates QGroupBoxes for stream connection, visualization, and recording,
        populating them with buttons, labels, and a plot canvas.

        Args:
            layout (QVBoxLayout): The main QVBoxLayout of the PylslTab to which
                                  UI elements will be added.
        """
        # Stream connection group
        connection_group = QGroupBox("Stream Connection")
        connection_layout = QVBoxLayout()
        
        stream_controls_layout = QHBoxLayout()
        self.pylsl_start_btn = QPushButton("Start LSL Stream")
        self.pylsl_stop_btn = QPushButton("Stop Stream")
        self.pylsl_refresh_btn = QPushButton("Refresh Streams")
        self.pylsl_stop_btn.setEnabled(False)
        
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
        
        visualization_group.setLayout(visualization_layout)
        layout.addWidget(visualization_group)
        
        # Data recording group
        recording_group = QGroupBox("Data Recording")
        recording_layout = QVBoxLayout()
        
        recording_controls_layout = QHBoxLayout()
        self.pylsl_record_btn = QPushButton("Start Recording")
        self.pylsl_stop_record_btn = QPushButton("Stop Recording")
        self.pylsl_save_btn = QPushButton("Save Data")
        self.pylsl_record_btn.setEnabled(False) # Initially disabled until stream starts
        self.pylsl_stop_record_btn.setEnabled(False)
        self.pylsl_save_btn.setEnabled(False)
        
        recording_controls_layout.addWidget(self.pylsl_record_btn)
        recording_controls_layout.addWidget(self.pylsl_stop_record_btn)
        recording_controls_layout.addWidget(self.pylsl_save_btn)
        recording_layout.addLayout(recording_controls_layout)
        
        self.pylsl_recording_status = QLabel("Recording: Not started")
        self.pylsl_recording_status.setStyleSheet("padding: 5px;")
        recording_layout.addWidget(self.pylsl_recording_status)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)

        # Inference group
        inference_group = QGroupBox("Real-time Inference")
        inference_layout = QVBoxLayout()
        
        self.model_selector = QComboBox()
        self.model_selector.addItem("Select Model", userData=None)
        self.model_selector.setEnabled(False)
        inference_layout.addWidget(self.model_selector)
        
        self.start_inference_btn = QPushButton("Start Inference")
        self.stop_inference_btn = QPushButton("Stop Inference")
        self.start_inference_btn.setEnabled(False)
        self.stop_inference_btn.setEnabled(False)
        
        inference_layout.addWidget(self.start_inference_btn)
        inference_layout.addWidget(self.stop_inference_btn)
        
        self.inference_status = QLabel("Inference: Not started")
        self.inference_status.setStyleSheet("padding: 5px;")
        inference_layout.addWidget(self.inference_status)
        
        self.inference_progress = QProgressBar()
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(0)
        inference_layout.addWidget(self.inference_progress)
        
        # Inference results display
        self.inference_results_display = QTextEdit()
        self.inference_results_display.setReadOnly(True)
        self.inference_results_display.setMaximumHeight(150)
        inference_layout.addWidget(self.inference_results_display)
        
        # Optional: Frame to visually separate inference section
        self.inference_frame = QFrame()
        self.inference_frame.setFrameShape(QFrame.StyledPanel)
        self.inference_frame.setLayout(inference_layout)
        layout.addWidget(self.inference_frame)

        layout.addStretch()        # Initialize PyLSL variables
        self.pylsl_inlet = None
        self.pylsl_buffer = None
        self.pylsl_time_buffer = None # Added for consistency
        self.pylsl_timer = QTimer(self)
        self.pylsl_timer.timeout.connect(self.update_pylsl_plot)
        self.pylsl_recording_data = []
        self.pylsl_is_recording = False
        self.current_sample_rate = 125  # Default sample rate
        
        # Initialize inference variables
        self.selected_model_info = None
        self.inference_processor = None
        self.inference_timer = None
        
        # Connect signals
        self.pylsl_start_btn.clicked.connect(self.start_pylsl_stream)
        self.pylsl_stop_btn.clicked.connect(self.stop_pylsl_stream)
        self.pylsl_refresh_btn.clicked.connect(self.refresh_pylsl_streams)
        self.pylsl_record_btn.clicked.connect(self.start_pylsl_recording)
        self.pylsl_stop_record_btn.clicked.connect(self.stop_pylsl_recording)
        self.pylsl_save_btn.clicked.connect(self.save_pylsl_data)
        
        # Inference connections
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.stop_inference_btn.clicked.connect(self.stop_inference)
        
        self.refresh_pylsl_streams() # Auto-refresh streams on load        # Load available models for inference
        self._load_available_models()

    def _load_available_models(self):
        """
        Loads and populates the model selector with available EEG models.

        Scans the model directory for PyTorch model files (.pth, .pt) and
        creates entries for known model types. For each model, adds an entry
        to the model selector combobox.
        """
        if not MODELS_AVAILABLE:
            return
        try:
            # Look for saved model files in common locations
            model_dirs = [
                os.path.join(project_root, "src", "model"),
                os.path.join(project_root, "models"),
                os.path.join(project_root, "saved_models"),
                os.path.join(project_root, "checkpoints")
            ]
            
            model_files = []
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for ext in ['.pth', '.pt', '.pkl']:
                        pattern = os.path.join(model_dir, f"*{ext}")
                        import glob
                        model_files.extend(glob.glob(pattern))
            
            if not model_files:
                # Add demo models for testing if no real models found
                self.model_selector.addItem("EEG Inception ERP (Demo)", userData={
                    "name": "EEG Inception ERP (Demo)", 
                    "type": "EEGInceptionERP",
                    "path": None,
                    "n_channels": 16,
                    "demo": True
                })
                self.model_selector.addItem("EEG IT-Net (Demo)", userData={
                    "name": "EEG IT-Net (Demo)", 
                    "type": "EEGITNet",
                    "path": None,
                    "n_channels": 16,
                    "demo": True
                })
                print("No trained model files found. Added demo models for testing.")
                
            else:
                for model_file in model_files:
                    filename = os.path.basename(model_file)
                    name = os.path.splitext(filename)[0]
                    
                    # Try to infer model type from filename
                    model_type = "EEGInceptionERP"  # Default
                    if "itnet" in name.lower() or "it_net" in name.lower():
                        model_type = "EEGITNet"
                    elif "inception" in name.lower():
                        model_type = "EEGInceptionERP"
                    
                    model_info = {
                        "name": name,
                        "type": model_type,
                        "path": model_file,
                        "n_channels": 16,  # Default, could be parsed from filename
                        "demo": False
                    }
                    
                    self.model_selector.addItem(name, userData=model_info)
            
            self.model_selector.setEnabled(True)
            self.start_inference_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.start_inference_btn.setEnabled(False)
            self.start_inference_btn.setEnabled(False)

    def on_model_selected(self):
        """
        Handles the event when a model is selected from the combobox.

        Updates the inference UI elements based on the selected model's properties.
        """
        current_index = self.model_selector.currentIndex()
        model_info = self.model_selector.itemData(current_index)
        
        if model_info is not None:
            # Update UI based on model info
            self.selected_model_info = model_info
            self.inference_status.setText(f"Selected Model: {model_info['name']}")
            self.inference_status.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            # Enable inference controls
            self.start_inference_btn.setEnabled(True)
            self.stop_inference_btn.setEnabled(False)
        else:
            # No valid model selected
            self.selected_model_info = None
            self.inference_status.setText("Inference: Not started")
            self.inference_status.setStyleSheet("padding: 5px;")
            self.start_inference_btn.setEnabled(False)
            self.stop_inference_btn.setEnabled(False)

    def refresh_pylsl_streams(self):
        """
        Refreshes the list of available LSL streams.

        Queries for LSL streams (specifically EEG type) and updates the
        information text area with the names, channel counts, and sample rates
        of found streams. Enables the start button if EEG streams are found.
        """
        if not PYLSL_AVAILABLE:
            return
        try:
            streams = resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            info_text = f"Found {len(streams)} total streams, {len(eeg_streams)} EEG streams:\n"
            for i, stream in enumerate(eeg_streams):
                info_text += f"  {i+1}. {stream.name()} - {stream.channel_count()} channels @ {stream.nominal_srate()} Hz\n"
            if not eeg_streams:
                info_text += "No EEG streams found. Make sure your EEG device is streaming to LSL."
            self.pylsl_info_text.setPlainText(info_text)
            self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0)
        except Exception as e:
            self.pylsl_info_text.setPlainText(f"Error refreshing streams: {str(e)}")
            self.pylsl_start_btn.setEnabled(False)

    def start_pylsl_stream(self):
        """
        Starts an LSL EEG stream.

        Resolves available EEG streams, connects to the first one found,
        initializes the data buffer and plotting timer, and updates UI elements
        to reflect the connected state. Displays messages for success or failure.
        """
        if not PYLSL_AVAILABLE:
            QMessageBox.warning(self, "PyLSL Not Available", "PyLSL is not installed. Please install it with: pip install pylsl")
            return
        try:
            streams = resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            if not eeg_streams:
                QMessageBox.warning(self, "No EEG Streams", "No EEG streams found. Make sure your device is streaming.")
                return
            
            self.pylsl_inlet = StreamInlet(eeg_streams[0], max_buflen=10) # Consider larger max_buflen if data loss occurs
            info = self.pylsl_inlet.info()
            n_channels = info.channel_count()
            sample_rate = int(info.nominal_srate())
            self.current_sample_rate = sample_rate if sample_rate > 0 else 125
            
            buffer_size = 400 # Samples
            self.pylsl_buffer = deque(maxlen=buffer_size)
            self.pylsl_time_buffer = deque(maxlen=buffer_size)
            
            self.pylsl_channel_display.setText(f"Channels: {n_channels}")
            self.pylsl_sample_rate.setText(f"Sample Rate: {self.current_sample_rate} Hz")
            self.pylsl_buffer_size.setText(f"Buffer: {buffer_size} samples")
            self.pylsl_status_label.setText("Status: Connected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            self.pylsl_start_btn.setEnabled(False)
            self.pylsl_stop_btn.setEnabled(True)
            self.pylsl_record_btn.setEnabled(True)
            
            self.pylsl_timer.start(40)  # Approx 25 FPS updates
            QMessageBox.information(self, "Stream Started", f"Successfully connected to EEG stream: {info.name()}\nChannels: {n_channels}\nSample Rate: {self.current_sample_rate} Hz")
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", f"Failed to start stream: {str(e)}\n{traceback.format_exc()}")

    def stop_pylsl_stream(self):
        """
        Stops the currently active LSL stream.

        Closes the LSL inlet, stops the plotting timer, clears the plot and
        data buffers, and updates UI elements to reflect the disconnected state.
        Also stops any active recording.
        """
        try:
            self.pylsl_timer.stop()
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream() # Properly close the inlet
            self.pylsl_inlet = None
            self.pylsl_buffer = None
            self.pylsl_time_buffer = None
            
            self.pylsl_status_label.setText("Status: Disconnected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
            self.pylsl_plot_canvas.clear_plot() # Use the method from PlotCanvas
            
            self.pylsl_start_btn.setEnabled(True)
            self.pylsl_stop_btn.setEnabled(False)
            self.pylsl_record_btn.setEnabled(False)
            
            if self.pylsl_is_recording:
                self.stop_pylsl_recording(show_message=False) # Avoid double message
            QMessageBox.information(self, "Stream Stopped", "LSL stream disconnected")
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Error stopping stream: {str(e)}\n{traceback.format_exc()}")

    def update_pylsl_plot(self):
        """
        Pulls new data from the LSL stream and updates the plot.

        This method is connected to the `pylsl_timer`. It pulls a chunk of
        samples from the LSL inlet, appends them to the data buffer, adds them
        to the recording data if recording is active, and calls
        `_update_simple_plot` to refresh the visualization.
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None:
            return
        try:
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.0, max_samples=int(self.current_sample_rate / 25) + 1) # Pull enough for ~40ms
            if samples:
                for i, sample in enumerate(samples):
                    self.pylsl_buffer.append(sample)
                    self.pylsl_time_buffer.append(timestamps[i]) # Store corresponding timestamp
                if self.pylsl_is_recording:
                    self.pylsl_recording_data.extend(zip(samples, timestamps))
                self._update_simple_plot()
        except Exception as e:
            print(f"DEBUG: Plot update error: {e}")
            # traceback.print_exc() # Can be noisy, enable if needed

    def _update_simple_plot(self):
        """
        Updates the EEG data plot with the current buffer content.

        Clears the previous plot and redraws the EEG channels. Data is scaled
        and offset for display. Limits the number of displayed channels for clarity.
        """
        if not self.pylsl_buffer or len(self.pylsl_buffer) == 0:
            self.pylsl_plot_canvas.clear_plot()
            return
        try:
            data_array = np.array(list(self.pylsl_buffer))
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1) # Ensure 2D for consistency
            
            n_channels = data_array.shape[1]
            display_samples = len(data_array)
            time_axis = np.arange(display_samples)

            self.pylsl_plot_canvas.axes.clear()
            
            channels_to_show = min(n_channels, 16) # Limit displayed channels
            for ch in range(channels_to_show):
                channel_data = data_array[:, ch]
                # Basic scaling and offset
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val < 1e-6: std_val = 1 # Avoid division by zero for flat signals
                scaled_data = (channel_data - mean_val) / (std_val * 3) # Normalize and scale to +/- 0.33 approx
                self.pylsl_plot_canvas.axes.plot(time_axis, scaled_data + ch, linewidth=0.8)
            
            self.pylsl_plot_canvas.axes.set_title("Real-time EEG Data")
            self.pylsl_plot_canvas.axes.set_xlabel("Sample Points")
            self.pylsl_plot_canvas.axes.set_ylabel("Channels")
            self.pylsl_plot_canvas.axes.grid(True, alpha=0.3)
            
            if n_channels > 0:
                self.pylsl_plot_canvas.axes.set_ylim(-0.7, channels_to_show - 0.3)
                self.pylsl_plot_canvas.axes.set_yticks(range(channels_to_show))
                self.pylsl_plot_canvas.axes.set_yticklabels([f'Ch {i+1}' for i in range(channels_to_show)])
            
            self.pylsl_plot_canvas.axes.set_xlim(0, display_samples)
            self.pylsl_plot_canvas.draw()
        except Exception as e:
            print(f"DEBUG: Simple plot error: {e}")
            # traceback.print_exc()

    def start_pylsl_recording(self):
        """
        Starts recording data from the LSL stream.

        Initializes the recording data list, sets the recording flag to True,
        and updates UI elements to reflect the recording state.
        """
        self.pylsl_recording_data = []
        self.pylsl_is_recording = True
        self.pylsl_recording_status.setText("Recording: Active")
        self.pylsl_recording_status.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        self.pylsl_record_btn.setEnabled(False)
        self.pylsl_stop_record_btn.setEnabled(True)
        self.pylsl_save_btn.setEnabled(False) # Disable save until recording stops
        QMessageBox.information(self, "Recording Started", "EEG data recording started")

    def stop_pylsl_recording(self, show_message=True):
        """
        Stops the current LSL data recording.

        Sets the recording flag to False and updates UI elements. Displays a
        message with the number of samples recorded.

        Args:
            show_message (bool, optional): Whether to display a confirmation
                                           QMessageBox. Defaults to True.
        """
        self.pylsl_is_recording = False
        samples_recorded = len(self.pylsl_recording_data)
        self.pylsl_recording_status.setText(f"Recording: Stopped ({samples_recorded} samples)")
        self.pylsl_recording_status.setStyleSheet("padding: 5px; background-color: #ccffcc;")
        self.pylsl_record_btn.setEnabled(True if self.pylsl_inlet else False) # Only enable if stream is active
        self.pylsl_stop_record_btn.setEnabled(False)
        self.pylsl_save_btn.setEnabled(samples_recorded > 0)
        if show_message:
            QMessageBox.information(self, "Recording Stopped", f"Recording stopped. {samples_recorded} samples recorded.")

    def save_pylsl_data(self):
        """
        Saves the recorded LSL data to a file.

        Prompts the user for a file location and name. Saves data in CSV or NPY
        format. For CSV, it attempts to use pandas for a more structured output;
        if pandas is not available, it falls back to a simpler numpy save or
        warns the user.
        """
        if not self.pylsl_recording_data:
            QMessageBox.warning(self, "No Data", "No recorded data to save")
            return
        try:
            default_filename = f"eeg_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # Ensure project_root/eeg_data exists for default save location
            save_dir = os.path.join(project_root, "eeg_data", "LSL_Recordings") 
            os.makedirs(save_dir, exist_ok=True)
            default_path = os.path.join(save_dir, default_filename)

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save EEG Data", default_path,
                "CSV files (*.csv);;NumPy files (*.npy);;All files (*.*)"
            )
            if file_path:
                samples, timestamps = zip(*self.pylsl_recording_data)
                samples_array = np.array(samples)
                timestamps_array = np.array(timestamps)
                
                if file_path.endswith('.npy'):
                    data_dict = {'samples': samples_array, 'timestamps': timestamps_array, 'fs': self.current_sample_rate}
                    np.save(file_path, data_dict)
                elif file_path.endswith('.csv'):
                    # For CSV, it might be better to have timestamps as the first column
                    # and then channel data. Pandas can help here.
                    try:
                        import pandas as pd
                        df_data = pd.DataFrame(samples_array)
                        df_timestamps = pd.DataFrame(timestamps_array, columns=['Timestamp'])
                        # Determine channel columns for data
                        if samples_array.ndim > 1:
                            df_data.columns = [f'Ch{i+1}' for i in range(samples_array.shape[1])]
                        else:
                            df_data.columns = ['EEG_Data']
                        # Concatenate timestamps and data
                        df_final = pd.concat([df_timestamps, df_data], axis=1)
                        df_final.to_csv(file_path, index=False)
                    except ImportError:
                        QMessageBox.warning(self, "Pandas not found", "Saving to CSV requires pandas. Please install it (`pip install pandas`) or choose .npy format.")
                        # Fallback to simple numpy save if pandas is not there, though .npy is preferred then.
                        header = "Timestamp,"+ ",".join([f'Ch{i+1}' for i in range(samples_array.shape[1])])
                        data_to_save = np.hstack((timestamps_array.reshape(-1,1), samples_array))
                        np.savetxt(file_path, data_to_save, delimiter=",", header=header, comments='')
                else: # For other formats, default to numpy save with .npy extension
                    if not file_path.endswith('.npy'): file_path += '.npy'
                    data_dict = {'samples': samples_array, 'timestamps': timestamps_array, 'fs': self.current_sample_rate}
                    np.save(file_path, data_dict)
                
                QMessageBox.information(self, "Data Saved", f"EEG data saved to:\n{file_path}")
                self.pylsl_save_btn.setEnabled(False)
                self.pylsl_recording_data = [] # Clear data after saving
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}\n{traceback.format_exc()}")

    def start_inference(self):
        """
        Starts real-time inference on the EEG stream using the selected model.

        Initializes the inference processor with the selected model and starts
        the inference timer. Updates UI elements to reflect the inference state.
        """
        if not MODELS_AVAILABLE or self.selected_model_info is None:
            QMessageBox.warning(self, "Model Not Selected", "Please select a valid model for inference")
            return
        
        try:
            # Get model information
            model_type = self.selected_model_info.get("type", "unknown")
            model_name = self.selected_model_info.get("name", "Unnamed Model")
            model_path = self.selected_model_info.get("path")
            n_channels = self.selected_model_info.get("n_channels", 16)
            is_demo = self.selected_model_info.get("demo", False)
            
            # Initialize the model
            model = None
            if model_type == "EEGInceptionERP":
                model = EEGInceptionERPModel()
                if not is_demo and model_path and os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"Loaded trained model from {model_path}")
                else:
                    print(f"Using demo model: {model_name}")
            elif model_type == "EEGITNet":
                model = EEGITNetModel()
                if not is_demo and model_path and os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"Loaded trained model from {model_path}")
                else:
                    print(f"Using demo model: {model_name}")
            else:
                QMessageBox.warning(self, "Unsupported Model", f"The selected model type '{model_type}' is not supported for inference")
                return
            
            # Initialize the RealTimeInferenceProcessor
            self.inference_processor = RealTimeInferenceProcessor(
                model=model,
                n_channels=n_channels,
                sample_rate=self.current_sample_rate,
                window_size=400,
                filter_enabled=True,
                l_freq=0.5,
                h_freq=50.0
            )
            
            # Setup inference timer for 30ms intervals (as requested)
            self.inference_timer = QTimer(self)
            self.inference_timer.setInterval(30)  # 30ms for real-time inference
            self.inference_timer.timeout.connect(self.process_inference)
            self.inference_timer.start()
            
            # Update UI
            self.inference_status.setText(f"Inference: Active ({model_name})")
            self.inference_status.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            self.start_inference_btn.setEnabled(False)
            self.stop_inference_btn.setEnabled(True)
            self.inference_progress.setValue(0)
            
            print(f"Started real-time inference with {model_name}")
            print(f"Buffer size: 400 samples, Sample rate: {self.current_sample_rate} Hz")
            print(f"Inference interval: 30ms")
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Failed to start inference: {str(e)}\n{traceback.format_exc()}")

    def stop_inference(self):
        """
        Stops the real-time inference.

        Stops the inference timer and processor, and updates UI elements to reflect
        the stopped state.
        """
        try:
            if hasattr(self, 'inference_timer'):
                self.inference_timer.stop()
                del self.inference_timer
            
            if hasattr(self, 'inference_processor'):
                # Properly close or delete the inference processor if needed
                del self.inference_processor
            
            self.inference_status.setText("Inference: Not started")
            self.inference_status.setStyleSheet("padding: 5px;")
            self.start_inference_btn.setEnabled(True)
            self.stop_inference_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Stop Inference Error", f"Failed to stop inference: {str(e)}\n{traceback.format_exc()}")

    def process_inference(self):
        """
        Processes a single step of inference.

        Pulls the latest EEG data, runs it through the inference processor, and
        updates the inference results display. This is called periodically by the
        inference timer.
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None or not hasattr(self, 'inference_processor'):
            return
        try:
            # Pull latest samples from LSL stream
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.0, max_samples=10)
            
            if samples:
                # Convert samples to numpy array with proper shape
                samples_array = np.array(samples)
                if samples_array.ndim == 1:
                    samples_array = samples_array.reshape(1, -1)
                
                # Add each sample to the inference processor buffer
                for sample in samples_array:
                    # Run inference with the new sample
                    results = self.inference_processor.predict(sample.reshape(1, -1))
                    
                    # Update results display only if we get a successful inference
                    if results.get('status') == 'success':
                        self._update_inference_results(results)
                        
                        # Update progress bar based on confidence
                        confidence = results.get('confidence', 0)
                        self.inference_progress.setValue(int(confidence * 100))
            
        except Exception as e:
            print(f"DEBUG: Inference processing error: {e}")
            # traceback.print_exc() # Enable for detailed trace

    def _preprocess_samples(self, samples):
        """
        Preprocesses the raw EEG samples before inference.

        Applies any necessary filtering, normalization, or reshaping to the samples
        to prepare them for the inference model.

        Args:
            samples (list): The raw EEG samples to preprocess.

        Returns:
            np.ndarray: The preprocessed samples as a NumPy array.
        """
        if not samples or len(samples) == 0:
            return np.array([])
        try:
            # Example preprocessing: Convert to NumPy array and normalize
            samples_array = np.array(samples)
            if samples_array.ndim == 1:
                samples_array = samples_array.reshape(1, -1) # Reshape for single sample
            
            # Apply any model-specific preprocessing here
            if hasattr(self, 'inference_processor') and hasattr(self.inference_processor, 'preprocess'):
                samples_array = self.inference_processor.preprocess(samples_array)
            
            return samples_array
        except Exception as e:
            print(f"DEBUG: Preprocessing error: {e}")
            return np.array(samples) # Fallback to raw samples on error

    def _update_inference_results(self, results):
        """
        Updates the inference results display with the latest results.

        Formats and displays the inference results in the results QTextEdit.

        Args:
            results (Any): The raw results from the inference processor.
        """
        if results is None:
            return
        try:
            # Example: Format results as JSON string for display
            formatted_results = json.dumps(results, indent=2)
            self.inference_results_display.setPlainText(formatted_results)
        except Exception as e:
            print(f"DEBUG: Results formatting error: {e}")

    def clear_resources(self):
        """
        Cleans up resources like timers and LSL inlets.

        This method should be called when the tab or application is closing
        to ensure that the LSL stream is properly closed and the timer is stopped,
        preventing potential issues or resource leaks.
        """
        print("PylslTab: Clearing resources...")
        try:
            self.pylsl_timer.stop()
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream()
            if hasattr(self, 'inference_timer'):
                self.inference_timer.stop()
            print("PylslTab: Resources cleared.")
        except Exception as e:
            print(f"PylslTab: Error clearing resources: {e}")

    # It's good practice to ensure resources are cleaned up.
    # This might be called from MainWindow's closeEvent or when the tab is removed.
    def closeEvent(self, event):
        """
        Handles the close event for the widget.

        Ensures that `clear_resources` is called before the widget is closed.

        Args:
            event (QCloseEvent): The close event.
        """
        self.clear_resources()
        super().closeEvent(event) # Important to call the parent's closeEvent
