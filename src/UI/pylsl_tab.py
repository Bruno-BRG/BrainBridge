import os
import sys
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QHBoxLayout, QMessageBox, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt
from collections import deque
from datetime import datetime

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

class PylslTab(QWidget):
    def __init__(self, parent_main_window):
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

        layout.addStretch()

        # Initialize PyLSL variables
        self.pylsl_inlet = None
        self.pylsl_buffer = None
        self.pylsl_time_buffer = None # Added for consistency
        self.pylsl_timer = QTimer(self)
        self.pylsl_timer.timeout.connect(self.update_pylsl_plot)
        self.pylsl_recording_data = []
        self.pylsl_is_recording = False
        self.current_sample_rate = 125  # Default sample rate
        
        # Connect signals
        self.pylsl_start_btn.clicked.connect(self.start_pylsl_stream)
        self.pylsl_stop_btn.clicked.connect(self.stop_pylsl_stream)
        self.pylsl_refresh_btn.clicked.connect(self.refresh_pylsl_streams)
        self.pylsl_record_btn.clicked.connect(self.start_pylsl_recording)
        self.pylsl_stop_record_btn.clicked.connect(self.stop_pylsl_recording)
        self.pylsl_save_btn.clicked.connect(self.save_pylsl_data)
        
        self.refresh_pylsl_streams() # Auto-refresh streams on load

    def refresh_pylsl_streams(self):
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
        self.pylsl_recording_data = []
        self.pylsl_is_recording = True
        self.pylsl_recording_status.setText("Recording: Active")
        self.pylsl_recording_status.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        self.pylsl_record_btn.setEnabled(False)
        self.pylsl_stop_record_btn.setEnabled(True)
        self.pylsl_save_btn.setEnabled(False) # Disable save until recording stops
        QMessageBox.information(self, "Recording Started", "EEG data recording started")

    def stop_pylsl_recording(self, show_message=True):
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

    def clear_resources(self):
        """Clean up resources like timers and inlets when the tab or window is closed."""
        print("PylslTab: Clearing resources...")
        try:
            self.pylsl_timer.stop()
            if self.pylsl_inlet:
                self.pylsl_inlet.close_stream()
            print("PylslTab: Resources cleared.")
        except Exception as e:
            print(f"PylslTab: Error clearing resources: {e}")

    # It's good practice to ensure resources are cleaned up.
    # This might be called from MainWindow's closeEvent or when the tab is removed.
    def closeEvent(self, event):
        self.clear_resources()
        super().closeEvent(event) # Important to call the parent's closeEvent
