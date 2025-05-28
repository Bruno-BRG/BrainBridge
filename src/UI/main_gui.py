import sys
import os
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTabWidget, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QGroupBox, QTextEdit, QHBoxLayout,
    QRadioButton, QSpinBox, QDoubleSpinBox, QMessageBox # Added QMessageBox
)
from PyQt5.QtGui import QPixmap # Import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal # Added QThread, pyqtSignal
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

# Import the StreamingWidget for PyLSL functionality
# Attempt relative import first, then absolute if within a package structure
try:
    from .StreamingWidget import StreamingWidget # Corrected import path to match potential capitalization
except ImportError:
    try:
        from src.UI.StreamingWidget import StreamingWidget # Corrected import path to match potential capitalization
    except ImportError:
        StreamingWidget = None
        print("Warning: StreamingWidget could not be imported. PyLSL tab might not work correctly.")

# Import tab classes
from .data_management_tab import DataManagementTab
from .training_tab import TrainingTab, TrainingThread # Import TrainingTab and TrainingThread


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, data, title=""):
        self.axes.clear()
        if data.ndim == 1: # Single channel
            self.axes.plot(data)
        elif data.ndim == 2: # Multiple channels (channels, timepoints)
            # Offset channels for visibility
            # Adjust offset based on data range to prevent overlap if necessary
            offset_scale = np.max(np.abs(data)) * 1.5 if np.max(np.abs(data)) > 0 else 1.0
            for i in range(data.shape[0]):
                self.axes.plot(data[i, :] + i * offset_scale)
        self.axes.set_title(title)
        self.axes.grid(True)
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_title("No Data") # Or some other placeholder
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Application")
        self.setGeometry(100, 100, 1000, 800) # Adjusted size
        # self.training_thread = None # Moved to TrainingTab
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
        self.main_layout.addWidget(self.tabs)

        # Create tabs
        self.data_tab = DataManagementTab(self) # Use the new class
        self.training_tab = TrainingTab(self) # Use the new TrainingTab class
        self.pylsl_tab = QWidget()

        self.tabs.addTab(self.data_tab, "Data Management")
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.pylsl_tab, "OpenBCI Live (PyLSL)")

        # Populate tabs
        # self.setup_data_tab() # This is now handled by DataManagementTab's __init__
        # self.setup_training_tab() # This is now handled by TrainingTab's __init__
        self.setup_pylsl_tab()

        # Exit button
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.close)
        self.main_layout.addWidget(self.exit_button)

    def setup_pylsl_tab(self):
        layout = QVBoxLayout(self.pylsl_tab)
        
        if not PYLSL_AVAILABLE:
            # Display error message if PyLSL is not available
            error_label = QLabel("PyLSL is not installed. Please install it to use this feature:")
            error_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            
            install_label = QLabel("pip install pylsl")
            install_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; font-family: monospace;")
            
            layout.addWidget(error_label)
            layout.addWidget(install_label)
            layout.addStretch()
            return
        
        # Create PyLSL streaming widget
        if StreamingWidget is not None:
            try:
                self.streaming_widget = StreamingWidget(self)
                layout.addWidget(self.streaming_widget)
            except Exception as e:
                # Fallback to manual implementation if StreamingWidget fails
                self.setup_manual_pylsl_tab(layout, str(e))
        else:
            # Manual implementation if StreamingWidget is not available
            self.setup_manual_pylsl_tab(layout, "StreamingWidget not found")
    
    def setup_manual_pylsl_tab(self, layout, error_msg=""):
        """Manual PyLSL implementation as fallback"""
        
        # Show error if any
        if error_msg:
            error_label = QLabel(f"StreamingWidget error: {error_msg}")
            error_label.setStyleSheet("color: orange; font-size: 12px;")
            layout.addWidget(error_label)
        
        # Stream connection group
        connection_group = QGroupBox("Stream Connection")
        connection_layout = QVBoxLayout()
        
        # Stream controls
        stream_controls_layout = QHBoxLayout()
        self.pylsl_start_btn = QPushButton("Start LSL Stream")
        self.pylsl_stop_btn = QPushButton("Stop Stream")
        self.pylsl_refresh_btn = QPushButton("Refresh Streams")
        self.pylsl_stop_btn.setEnabled(False)
        
        stream_controls_layout.addWidget(self.pylsl_start_btn)
        stream_controls_layout.addWidget(self.pylsl_stop_btn)
        stream_controls_layout.addWidget(self.pylsl_refresh_btn)
        connection_layout.addLayout(stream_controls_layout)
        
        # Stream status
        self.pylsl_status_label = QLabel("Status: Disconnected")
        self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        connection_layout.addWidget(self.pylsl_status_label)
        
        # Stream info
        self.pylsl_info_text = QTextEdit()
        self.pylsl_info_text.setMaximumHeight(100)
        self.pylsl_info_text.setPlainText("No stream information available")
        connection_layout.addWidget(self.pylsl_info_text)
        
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # Real-time visualization group
        visualization_group = QGroupBox("Real-time EEG Visualization")
        visualization_layout = QVBoxLayout()
          # Plot canvas for real-time data
        self.pylsl_plot_canvas = PlotCanvas(self, width=10, height=12, dpi=80)
        visualization_layout.addWidget(self.pylsl_plot_canvas)
        
        # Visualization controls
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
        
        # Recording controls
        recording_controls_layout = QHBoxLayout()
        self.pylsl_record_btn = QPushButton("Start Recording")
        self.pylsl_stop_record_btn = QPushButton("Stop Recording")
        self.pylsl_save_btn = QPushButton("Save Data")
        self.pylsl_stop_record_btn.setEnabled(False)
        self.pylsl_save_btn.setEnabled(False)
        
        recording_controls_layout.addWidget(self.pylsl_record_btn)
        recording_controls_layout.addWidget(self.pylsl_stop_record_btn)
        recording_controls_layout.addWidget(self.pylsl_save_btn)
        recording_layout.addLayout(recording_controls_layout)
        
        # Recording status
        self.pylsl_recording_status = QLabel("Recording: Not started")
        self.pylsl_recording_status.setStyleSheet("padding: 5px;")
        recording_layout.addWidget(self.pylsl_recording_status)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)
          # Initialize PyLSL variables
        self.pylsl_inlet = None
        self.pylsl_buffer = None
        self.pylsl_timer = QTimer()
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
        
        # Auto-refresh streams on load
        self.refresh_pylsl_streams()
    
    def refresh_pylsl_streams(self):
        """Refresh available LSL streams"""
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
            
            # Enable start button if EEG streams are available
            self.pylsl_start_btn.setEnabled(len(eeg_streams) > 0)
            
        except Exception as e:
            self.pylsl_info_text.setPlainText(f"Error refreshing streams: {str(e)}")
            self.pylsl_start_btn.setEnabled(False)
    
    def start_pylsl_stream(self):
        """Start PyLSL streaming"""
        if not PYLSL_AVAILABLE:
            QMessageBox.warning(self, "PyLSL Not Available", 
                              "PyLSL is not installed. Please install it with: pip install pylsl")
            return
        
        try:
            streams = resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type() == 'EEG']
            
            if not eeg_streams:
                QMessageBox.warning(self, "No EEG Streams", 
                                  "No EEG streams found. Make sure your device is streaming.")
                return
            
            # Connect to the first EEG stream with larger buffer
            self.pylsl_inlet = StreamInlet(eeg_streams[0], max_buflen=10)
            info = self.pylsl_inlet.info()
            
            # Get stream info
            n_channels = info.channel_count()
            sample_rate = int(info.nominal_srate())
              # Store sample rate for later use
            self.current_sample_rate = sample_rate if sample_rate > 0 else 125
            
            # Initialize buffer (store last 400 samples of data)
            buffer_size = 400
            self.pylsl_buffer = deque(maxlen=buffer_size)
            self.pylsl_time_buffer = deque(maxlen=buffer_size)  # Separate time buffer
            
            print(f"DEBUG: Stream info - Channels: {n_channels}, Sample rate: {self.current_sample_rate}")
            print(f"DEBUG: Buffer initialized with size: {buffer_size}")
            
            # Update UI
            self.pylsl_channel_display.setText(f"Channels: {n_channels}")
            self.pylsl_sample_rate.setText(f"Sample Rate: {self.current_sample_rate} Hz")
            self.pylsl_buffer_size.setText(f"Buffer: {buffer_size} samples")
            self.pylsl_status_label.setText("Status: Connected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ccffcc;")
            
            # Enable/disable buttons
            self.pylsl_start_btn.setEnabled(False)
            self.pylsl_stop_btn.setEnabled(True)
            self.pylsl_record_btn.setEnabled(True)
            
            # Start updating plot with faster refresh
            print("DEBUG: Starting timer for plot updates")
            self.pylsl_timer.start(40)  # Update every 40ms (25 Hz)
            
            QMessageBox.information(self, "Stream Started", 
                                  f"Successfully connected to EEG stream:\n"
                                  f"Channels: {n_channels}\n"
                                  f"Sample Rate: {self.current_sample_rate} Hz")
            
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", f"Failed to start stream: {str(e)}")
    
    def stop_pylsl_stream(self):
        """Stop PyLSL streaming"""
        try:
            self.pylsl_timer.stop()
            self.pylsl_inlet = None
            self.pylsl_buffer = None
            
            # Update UI
            self.pylsl_status_label.setText("Status: Disconnected")
            self.pylsl_status_label.setStyleSheet("padding: 5px; background-color: #ffcccc;")
            
            # Clear plot
            self.pylsl_plot_canvas.axes.clear()
            self.pylsl_plot_canvas.draw()
            
            # Enable/disable buttons
            self.pylsl_start_btn.setEnabled(True)
            self.pylsl_stop_btn.setEnabled(False)
            self.pylsl_record_btn.setEnabled(False)
            
            # Stop recording if active
            if self.pylsl_is_recording:
                self.stop_pylsl_recording()
            
            QMessageBox.information(self, "Stream Stopped", "LSL stream disconnected")
            
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Error stopping stream: {str(e)}")

    def update_pylsl_plot(self):
        """Update real-time plot with simple static visualization"""
        if not self.pylsl_inlet:
            print("DEBUG: No inlet available")
            return
            
        if self.pylsl_buffer is None:
            print("DEBUG: Buffer is None")
            return
            
        try:
            # Pull new samples
            samples, timestamps = self.pylsl_inlet.pull_chunk(timeout=0.0, max_samples=32)
            
            if samples:
                print(f"DEBUG: Received {len(samples)} samples")
                
                # Add to buffer (simple samples only, no timestamps)
                for sample in samples:
                    self.pylsl_buffer.append(sample)

                # Add to recording if active
                if self.pylsl_is_recording and timestamps:
                    self.pylsl_recording_data.extend([(sample, timestamp) for sample, timestamp in zip(samples, timestamps)])
                  # Update simple plot
                self._update_simple_plot()
            else:
                print("DEBUG: No new samples received")
        
        except Exception as e:
            print(f"DEBUG: Plot update error: {e}")
            import traceback
            traceback.print_exc()

    def _update_simple_plot(self):
        """Create simple static plot visualization"""
        if len(self.pylsl_buffer) == 0:
            print("DEBUG: Buffer is empty")
            return
        
        try:
            print(f"DEBUG: Buffer has {len(self.pylsl_buffer)} samples")
            
            # Convert buffer to numpy array
            if len(self.pylsl_buffer) > 0:
                data_array = np.array(list(self.pylsl_buffer))
                print(f"DEBUG: Data array shape: {data_array.shape}")
                
                # Clear and setup plot
                self.pylsl_plot_canvas.axes.clear()
                
                # Determine number of channels
                n_channels = data_array.shape[1] if len(data_array.shape) > 1 else 1
                print(f"DEBUG: Number of channels: {n_channels}")
                  # Get all available data for display (up to 400 samples)
                display_samples = min(400, len(data_array))
                data_to_plot = data_array[-display_samples:]                # Create time axis
                time_axis = np.arange(len(data_to_plot))
                
                if len(data_to_plot.shape) > 1:
                    # Multiple channels - show all channels with channel-based Y-axis
                    channels_to_show = min(n_channels, 16)  # Show up to 16 channels
                    for ch in range(channels_to_show):
                        channel_data = data_to_plot[:, ch]
                        # Normalize each channel and scale down to fit within channel spacing
                        normalized_data = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
                        # Scale the normalized data to be much smaller (0.15 units high per channel)
                        scaled_data = normalized_data * 0.15
                        self.pylsl_plot_canvas.axes.plot(time_axis, scaled_data + ch, 
                                                        label=f'Ch {ch+1}', linewidth=0.8)
                else:
                    # Single channel
                    self.pylsl_plot_canvas.axes.plot(time_axis, data_to_plot, 
                                                    label='EEG', linewidth=1)
                
                # Set up simple visualization with channel-based Y-axis
                self.pylsl_plot_canvas.axes.set_title("Real-time EEG Data")
                self.pylsl_plot_canvas.axes.set_xlabel("Sample Points")
                self.pylsl_plot_canvas.axes.set_ylabel("Channels")
                
                # Add grid for better readability
                self.pylsl_plot_canvas.axes.grid(True, alpha=0.3)
                  # Set Y-axis to show channel numbers with proper spacing
                if len(data_to_plot.shape) > 1 and n_channels > 1:
                    channels_shown = min(n_channels, 16)
                    # Add more margin between channels by expanding Y-limits
                    self.pylsl_plot_canvas.axes.set_ylim(-0.7, channels_shown - 0.3)
                    # Set Y-tick labels to show channel numbers
                    self.pylsl_plot_canvas.axes.set_yticks(range(channels_shown))
                    self.pylsl_plot_canvas.axes.set_yticklabels([f'Ch {i+1}' for i in range(channels_shown)])
                
                # Show legend only for reasonable number of channels
                if n_channels <= 4:
                    self.pylsl_plot_canvas.axes.legend(loc='upper right', fontsize=8)
                
                # Auto-scale X-axis for better visibility
                self.pylsl_plot_canvas.axes.set_xlim(0, len(data_to_plot))
                
                # Redraw the plot
                self.pylsl_plot_canvas.draw()
                print("DEBUG: Plot updated successfully")
                
        except Exception as e:
            print(f"DEBUG: Simple plot error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_pylsl_recording(self):
        """Start recording PyLSL data"""
        self.pylsl_recording_data = []
        self.pylsl_is_recording = True
        
        self.pylsl_recording_status.setText("Recording: Active")
        self.pylsl_recording_status.setStyleSheet("padding: 5px; background-color: #ffcccc;")
        
        self.pylsl_record_btn.setEnabled(False)
        self.pylsl_stop_record_btn.setEnabled(True)
        
        QMessageBox.information(self, "Recording Started", "EEG data recording started")
    
    def stop_pylsl_recording(self):
        """Stop recording PyLSL data"""
        self.pylsl_is_recording = False
        
        samples_recorded = len(self.pylsl_recording_data)
        self.pylsl_recording_status.setText(f"Recording: Stopped ({samples_recorded} samples)")
        self.pylsl_recording_status.setStyleSheet("padding: 5px; background-color: #ccffcc;")
        
        self.pylsl_record_btn.setEnabled(True)
        self.pylsl_stop_record_btn.setEnabled(False)
        self.pylsl_save_btn.setEnabled(samples_recorded > 0)
        
        QMessageBox.information(self, "Recording Stopped", 
                              f"Recording stopped. {samples_recorded} samples recorded.")
    
    def save_pylsl_data(self):
        """Save recorded PyLSL data to file"""
        if not self.pylsl_recording_data:
            QMessageBox.warning(self, "No Data", "No recorded data to save")
            return
        
        try:
            # Open save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save EEG Data", 
                f"eeg_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV files (*.csv);;NumPy files (*.npy);;All files (*.*)"
            )
            
            if file_path:
                # Extract samples and timestamps
                samples = [item[0] for item in self.pylsl_recording_data]
                timestamps = [item[1] for item in self.pylsl_recording_data]
                
                if file_path.endswith('.npy'):
                    # Save as NumPy array
                    data_dict = {
                        'samples': np.array(samples),
                        'timestamps': np.array(timestamps)
                    }
                    np.save(file_path, data_dict)
                else:
                    # Save as CSV
                    data_array = np.array(samples)
                    
                    # Create DataFrame
                    import pandas as pd
                    if len(data_array.shape) > 1:
                        columns = [f'Ch_{i+1}' for i in range(data_array.shape[1])]
                        df = pd.DataFrame(data_array, columns=columns)
                    else:
                        df = pd.DataFrame(data_array, columns=['EEG'])
                    
                    df['Timestamp'] = timestamps
                    df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Data Saved", f"EEG data saved to:\n{file_path}")
                self.pylsl_save_btn.setEnabled(False)
        
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")


# Matplotlib Canvas Widget
class TrainingPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(TrainingPlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, data_window, title="EEG Sample", num_points=500):
        self.axes.clear()
        # Data window shape is (channels, timepoints)
        # We want to plot first 500 timepoints if longer
        timepoints_to_plot = min(data_window.shape[1], num_points)
        
        for i in range(data_window.shape[0]): # Iterate over channels
            self.axes.plot(data_window[i, :timepoints_to_plot], label=f'Ch {i+1}')
        
        self.axes.set_title(title)
        self.axes.set_xlabel(f"Timepoints (first {timepoints_to_plot})")
        self.axes.set_ylabel("Amplitude")
        # self.ax.legend(loc='upper right', fontsize='small') # Optional: legend can be crowded
        self.axes.grid(True)
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_title("No Data to Display")
        self.axes.set_xlabel("Timepoints")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True)
        self.draw()


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
