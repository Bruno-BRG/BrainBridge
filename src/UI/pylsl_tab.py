"""
File:    pylsl_tab.py
Class:   PylslTab
Purpose: Provides the user interface tab for discovering, connecting to,
         and visualizing EEG data streams using the Lab Streaming
         Layer (LSL) library.
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
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QHBoxLayout,
    QMessageBox
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
    StreamInlet = None # Define for type hinting if not available
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
        self.pylsl_plot_canvas = PlotCanvas(self, width=10, height=12, dpi=80) # Adjusted height
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
        
        layout.addStretch()

        # Initialize PyLSL variables
        self.pylsl_inlet: Optional['StreamInlet'] = None
        self.pylsl_buffer: Optional[deque] = None
        self.pylsl_time_buffer: Optional[deque] = None # For timestamped data if needed by plot
        self.pylsl_timer = QTimer(self)
        self.pylsl_timer.timeout.connect(self.update_pylsl_plot)
        self.current_sample_rate = 125  # Default sample rate, updated on stream connection
        
        # Connect signals
        self.pylsl_start_btn.clicked.connect(self.start_pylsl_stream)
        self.pylsl_stop_btn.clicked.connect(self.stop_pylsl_stream)
        self.pylsl_refresh_btn.clicked.connect(self.refresh_pylsl_streams)
        
        self.refresh_pylsl_streams() # Auto-refresh streams on load

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
            self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0)
        except Exception as e:
            self.pylsl_info_text.setPlainText(f"Error refreshing streams: {str(e)}")
            self.pylsl_start_btn.setEnabled(False)

    def start_pylsl_stream(self):
        """
        Starts an LSL EEG stream.
        Connects to the first available EEG stream, initializes buffers,
        and starts the plotting timer.
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
            
            # Connect to the first EEG stream found
            self.pylsl_inlet = StreamInlet(eeg_streams[0], max_buflen=360) # max_buflen adjusted, e.g., for 3s at 120Hz
            info = self.pylsl_inlet.info()
            n_channels = info.channel_count()
            # Use nominal_srate, ensure it's a positive value
            sample_rate = info.nominal_srate()
            if sample_rate <= 0: # If nominal_srate is 0 or irregular, use a default or estimate
                print(f"Warning: Stream {info.name()} reported nominal_srate of {sample_rate}. Using default 125 Hz.")
                self.current_sample_rate = 125.0
            else:
                self.current_sample_rate = float(sample_rate)

            # Buffer size set to display 400 samples
            buffer_size = 400
            
            self.pylsl_buffer = deque(maxlen=buffer_size)
            self.pylsl_time_buffer = deque(maxlen=buffer_size) # Store timestamps
            
            self.pylsl_channel_display.setText(f"Channels: {n_channels}")
            self.pylsl_sample_rate.setText(f"Sample Rate: {self.current_sample_rate:.2f} Hz")
            self.pylsl_buffer_size.setText(f"Buffer: {buffer_size} samples")
            self.pylsl_status_label.setText("Status: Connected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            self.pylsl_start_btn.setEnabled(False)
            self.pylsl_stop_btn.setEnabled(True)
            
            self.pylsl_timer.start(int(1000 / 25))  # Target ~25 FPS plot updates (40ms interval)
            QMessageBox.information(self, "Stream Started", f"Successfully connected to EEG stream: {info.name()}\\nChannels: {n_channels}\\nSample Rate: {self.current_sample_rate:.2f} Hz")
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", f"Failed to start stream: {str(e)}\n{traceback.format_exc()}")

    def stop_pylsl_stream(self):
        """
        Stops the currently active LSL stream.
        Closes the LSL inlet, stops the plotting timer, and resets UI elements.
        """
        try:
            self.pylsl_timer.stop()
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream() # Properly close the inlet
            self.pylsl_inlet = None
            self.pylsl_buffer = None # Clear buffer
            self.pylsl_time_buffer = None # Clear time buffer
            
            self.pylsl_status_label.setText("Status: Disconnected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
            self.pylsl_plot_canvas.clear_plot() # Use the method from PlotCanvas
            
            self.pylsl_start_btn.setEnabled(True)
            self.pylsl_stop_btn.setEnabled(False)
            
            QMessageBox.information(self, "Stream Stopped", "LSL stream disconnected.")
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Error stopping stream: {str(e)}\n{traceback.format_exc()}")

    def update_pylsl_plot(self):
        """
        Pulls new data from the LSL stream and updates the plot.
        This method is connected to the `pylsl_timer`.
        """
        if not self.pylsl_inlet or self.pylsl_buffer is None:
            return
        try:
            # Pull a chunk of samples. Adjust max_samples based on desired update rate and sample rate.
            # e.g., if timer is 40ms (25Hz), pull enough samples for that duration.
            samples_to_pull = int(self.current_sample_rate * (self.pylsl_timer.interval() / 1000.0))
            # Ensure at least 1 sample is pulled, timeout is 0.0 for non-blocking
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.0, max_samples=max(1, samples_to_pull)) 
            
            if samples:
                # samples is a list of lists (chunk_size x n_channels)
                for sample_group in samples: 
                    # Each 'sample_group' is a single time point with multiple channels
                    self.pylsl_buffer.append(sample_group) 
                # Timestamps are also available in 'timestamps' list, one for each sample_group
                # self.pylsl_time_buffer could be updated here if needed for plotting time axes
                self._update_visualization_plot() # Update plot with new data
        except Exception as e:
            print(f"DEBUG: Plot update error in pull_chunk or buffer append: {e}") 
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

                # Convert list of samples (each sample is a list/tuple of channel values)
                # to a NumPy array of shape (num_samples, num_channels)
                plot_data_samples_channels = np.array(current_buffer_list)

                if plot_data_samples_channels.size == 0:
                    self.pylsl_plot_canvas.clear_plot()
                    return

                # Transpose to (num_channels, num_samples) for the PlotCanvas.plot method
                plot_data_channels_samples = plot_data_samples_channels.T
                
                # The PlotCanvas.plot method can handle 1D (single channel) or 2D (multiple channels)
                # where the first dimension is channels if 2D.
                # If plot_data_channels_samples is (1, num_samples), it's treated as 2D with one channel.
                # If it somehow became 1D (e.g. (num_samples,)), PlotCanvas.plot also handles it.

                max_channels_to_display = 16 # Make this configurable if needed
                
                if plot_data_channels_samples.shape[0] > max_channels_to_display:
                    data_to_plot = plot_data_channels_samples[:max_channels_to_display, :]
                else:
                    data_to_plot = plot_data_channels_samples
                
                if data_to_plot.size > 0:
                    self.pylsl_plot_canvas.plot(data_to_plot, title="Real-time EEG Data")
                else:
                    # This might happen if max_channels_to_display is 0 or data becomes empty after slicing
                    self.pylsl_plot_canvas.clear_plot()
            except Exception as e:
                print(f"Error during plot data preparation: {str(e)}")
                traceback.print_exc()
                self.pylsl_plot_canvas.clear_plot() # Fallback to clear plot on error
        else:
            self.pylsl_plot_canvas.clear_plot() # Buffer is empty or None

    def closeEvent(self, event):
        """
        Handles the tab being closed or the application quitting.
        Ensures the LSL stream is stopped.
        """
        self.stop_pylsl_stream() # Ensure stream is stopped
        super().closeEvent(event) # Call base class closeEvent

# Example of how PlotCanvas might need to be adapted or used:
# (This is a conceptual note, actual PlotCanvas implementation might vary)
# class PlotCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super().__init__(fig)
#         self.setParent(parent)
#
#     def update_plot(self, data_matrix, sample_rate): # data_matrix: channels x samples
#         self.axes.clear()
#         if data_matrix is not None and data_matrix.shape[1] > 0:
#             num_channels, num_samples = data_matrix.shape
#             time_vector = np.arange(num_samples) / sample_rate
#             offset_scale = np.std(data_matrix) * 3 if np.std(data_matrix) > 0 else 1 # Heuristic for scaling
#
#             for i in range(num_channels):
#                 offset = i * offset_scale
#                 self.axes.plot(time_vector, data_matrix[i, :] - offset)
#
#             self.axes.set_xlabel("Time (s)")
#             self.axes.set_yticks([]) # Remove y-ticks for cleaner multi-channel display
#             self.axes.set_title("Real-time EEG Data")
#         else:
#             self.axes.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center')
#         self.draw()
#
#     def clear_plot(self):
#         self.axes.clear()
#         self.axes.text(0.5, 0.5, "Stream stopped or no data", horizontalalignment='center', verticalalignment='center')
#         self.draw()

# Make sure to adjust PlotCanvas if its methods were named differently or expect different data.
# The key methods used here are `update_plot(data_matrix, sample_rate)` and `clear_plot()`.
