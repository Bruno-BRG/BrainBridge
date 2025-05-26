import sys
import os # Add os import
import numpy as np # Add numpy import
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QTabWidget, QLabel, QFileDialog,
                             QLineEdit, QFormLayout, QGroupBox, QTextEdit, QHBoxLayout) # Added QHBoxLayout
from PyQt5.QtCore import Qt # Added Qt
# Add project root to Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import BCIDataLoader # Add BCIDataLoader import

# Matplotlib imports for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Application")
        self.setGeometry(100, 100, 900, 700)  # Adjusted size for plot

        # Initialize data cache
        self.data_cache = {
            "windows": None,
            "labels": None,
            "subject_ids": None,
            "data_summary": "No data loaded yet.",
            "data_path": "eeg_data", # Default data path
            "subjects_list": None, # Store subject list for reuse
            "current_plot_index": 0,
            "label_map": {0: "Right Hand", 1: "Left Hand"} # Assuming 0 for Right, 1 for Left
        }

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tab widget for different sections
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Create tabs
        self.data_tab = QWidget()
        self.training_tab = QWidget()
        self.pylsl_tab = QWidget()

        self.tabs.addTab(self.data_tab, "Data Management")
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.pylsl_tab, "OpenBCI Live (PyLSL)")

        # Populate tabs
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_pylsl_tab()

        # Exit button
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.close)
        self.main_layout.addWidget(self.exit_button)

    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        # --- Data Loading Group ---
        data_loading_group = QGroupBox("Load EEG Data")
        data_loading_layout = QFormLayout()

        self.data_path_label = QLabel("Selected Path: Not selected")
        self.btn_browse_data_path = QPushButton("Browse EEG Data Directory")
        self.btn_browse_data_path.clicked.connect(self.browse_data_directory)
        data_loading_layout.addRow(self.btn_browse_data_path, self.data_path_label)

        self.subject_ids_input = QLineEdit()
        self.subject_ids_input.setPlaceholderText("e.g., 1,2,3 or all")
        data_loading_layout.addRow(QLabel("Subject IDs:"), self.subject_ids_input)

        self.btn_load_data = QPushButton("Load and Process Data")
        self.btn_load_data.clicked.connect(self.load_data_action) # Placeholder action
        data_loading_layout.addWidget(self.btn_load_data)
        
        data_loading_group.setLayout(data_loading_layout)
        layout.addWidget(data_loading_group)

        # --- Data Summary Group ---
        data_summary_group = QGroupBox("Data Summary")
        data_summary_layout = QVBoxLayout()
        self.data_summary_display = QTextEdit()
        self.data_summary_display.setReadOnly(True)
        self.data_summary_display.setText(self.data_cache["data_summary"]) # Use cached summary
        data_summary_layout.addWidget(self.data_summary_display)
        data_summary_group.setLayout(data_summary_layout)
        layout.addWidget(data_summary_group)

        # --- EEG Plotting Group ---
        plot_group = QGroupBox("EEG Data Plot")
        plot_layout = QVBoxLayout()

        self.plot_canvas = PlotCanvas(self, width=8, height=4) # Adjusted size
        plot_layout.addWidget(self.plot_canvas)

        # Navigation buttons for plot
        nav_layout = QHBoxLayout()
        self.btn_prev_sample = QPushButton("Previous Sample")
        self.btn_prev_sample.clicked.connect(self.previous_sample)
        self.btn_prev_sample.setEnabled(False) # Disabled initially
        nav_layout.addWidget(self.btn_prev_sample)

        self.current_sample_label = QLabel("Sample: -/-")
        self.current_sample_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.current_sample_label)

        self.btn_next_sample = QPushButton("Next Sample")
        self.btn_next_sample.clicked.connect(self.next_sample)
        self.btn_next_sample.setEnabled(False) # Disabled initially
        nav_layout.addWidget(self.btn_next_sample)
        plot_layout.addLayout(nav_layout)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)


        layout.addStretch() # Add stretch to push elements to the top

    def browse_data_directory(self):
        # Placeholder for browsing directory logic
        dir_path = QFileDialog.getExistingDirectory(self, "Select EEG Data Directory", self.data_cache["data_path"])
        if dir_path:
            self.data_path_label.setText(f"Selected Path: {dir_path}")
            self.data_cache["data_path"] = dir_path # Store this path for later use
            print(f"Selected data directory: {dir_path}")

    def load_data_action(self):
        subjects_str = self.subject_ids_input.text().strip()
        data_path = self.data_cache["data_path"]

        if not data_path or data_path == "Not selected":
            self.data_summary_display.setText("Error: Please select a data directory first.")
            return

        # Validate data_path
        if not os.path.isdir(data_path):
            potential_path = os.path.join(project_root, data_path)
            if os.path.isdir(potential_path):
                data_path = potential_path
                self.data_cache["data_path"] = data_path # Update cache if relative path was resolved
                self.data_path_label.setText(f"Selected Path: {data_path}") # Update label
            else:
                self.data_summary_display.setText(f"Error: Data directory '{data_path}' not found.")
                return

        subjects_list_for_loader = None
        current_subjects_display = "all" # Default display if no specific subjects are entered

        if not subjects_str: # User pressed Enter or field is empty
            subjects_list_for_loader = self.data_cache["subjects_list"] # Use cached
            if self.data_cache["subjects_list"] is not None:
                current_subjects_display = ",".join(map(str, self.data_cache["subjects_list"]))
            self.data_summary_display.setText(f"Using previously specified subjects: {current_subjects_display}")
        elif subjects_str.lower() == 'all':
            subjects_list_for_loader = None
            self.data_cache["subjects_list"] = None # Update cache
            current_subjects_display = "all"
        else:
            try:
                subjects_list_for_loader = [int(s.strip()) for s in subjects_str.split(',') if s.strip()]
                if not subjects_list_for_loader: # Handle empty strings after split (e.g. "1,,2")
                    self.data_summary_display.setText("Error: Invalid subject IDs. Please enter comma-separated numbers or 'all'.")
                    return
                self.data_cache["subjects_list"] = subjects_list_for_loader # Update cache
                current_subjects_display = subjects_str
            except ValueError:
                self.data_summary_display.setText("Error: Invalid subject IDs. Please enter comma-separated numbers or 'all'.")
                return
        
        self.data_summary_display.setText(f"Loading data from: {data_path}\\nSubjects: {current_subjects_display}...")

        try:
            loader = BCIDataLoader(data_path=data_path, subjects=subjects_list_for_loader)
            windows, labels, subject_ids_loaded = loader.load_all_subjects() # Renamed to avoid conflict

            if windows.size == 0:
                summary = "No data was loaded. Please check data path and subject IDs."
                self.data_cache["data_summary"] = summary
            else:
                self.data_cache["windows"] = windows
                self.data_cache["labels"] = labels
                self.data_cache["subject_ids"] = subject_ids_loaded # Use renamed variable
                
                summary = (
                    f"Data loaded successfully!\\n"
                    f"  Total windows: {windows.shape[0]}\\n"
                    f"  Window shape: ({windows.shape[1]}, {windows.shape[2]}) (channels, timepoints)\\n"
                    f"  Number of unique subjects: {len(np.unique(subject_ids_loaded))}\\n"
                    f"  Class distribution: {np.bincount(labels.astype(int))}" # Ensure labels are int for bincount
                )
                self.data_cache["data_summary"] = summary
            
            self.data_summary_display.setText(summary)
            print(summary)

        except FileNotFoundError as e:
            error_msg = f"Error: {e}"
            self.data_summary_display.setText(error_msg)
            self.data_cache["data_summary"] = error_msg
            print(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            self.data_summary_display.setText(error_msg)
            self.data_cache["data_summary"] = error_msg
            print(error_msg)
        finally:
            # Attempt to plot data if loaded, or clear plot if not
            self.data_cache["current_plot_index"] = 0
            self.update_plot()
            self.update_plot_navigation()


    def update_plot(self):
        self.plot_canvas.clear_plot()
        if self.data_cache["windows"] is not None and self.data_cache["windows"].size > 0:
            idx = self.data_cache["current_plot_index"]
            
            if 0 <= idx < len(self.data_cache["windows"]):
                sample_window = self.data_cache["windows"][idx]
                label_val = self.data_cache["labels"][idx]
                subject_id_val = self.data_cache["subject_ids"][idx] # Assuming subject_ids are per window

                # Determine label text
                label_text = self.data_cache["label_map"].get(label_val, f"Unknown Label ({label_val})")
                
                plot_title = f"Subject: {subject_id_val} - Label: {label_text} (Sample {idx + 1})"
                
                # Ensure sample_window is 2D (channels, timepoints)
                if sample_window.ndim == 3 and sample_window.shape[0] == 1: # (1, channels, timepoints)
                    sample_window = sample_window.squeeze(0)
                elif sample_window.ndim != 2:
                    self.data_summary_display.append(f"\\nSkipping plot for sample {idx+1}: Unexpected window dimensions {sample_window.shape}")
                    self.plot_canvas.ax.set_title(f"Error: Cannot plot sample {idx+1}")
                    self.plot_canvas.draw()
                    return

                self.plot_canvas.plot(sample_window, title=plot_title)
            else:
                self.plot_canvas.ax.set_title("No data at current index")
                self.plot_canvas.draw()
        else:
            self.plot_canvas.ax.set_title("No EEG Data Loaded")
            self.plot_canvas.draw()
        self.update_plot_navigation()

    def update_plot_navigation(self):
        num_samples = len(self.data_cache["windows"]) if self.data_cache["windows"] is not None else 0
        current_idx = self.data_cache["current_plot_index"]

        if num_samples > 0:
            self.current_sample_label.setText(f"Sample: {current_idx + 1}/{num_samples}")
            self.btn_prev_sample.setEnabled(current_idx > 0)
            self.btn_next_sample.setEnabled(current_idx < num_samples - 1)
        else:
            self.current_sample_label.setText("Sample: -/-")
            self.btn_prev_sample.setEnabled(False)
            self.btn_next_sample.setEnabled(False)

    def next_sample(self):
        if self.data_cache["windows"] is not None:
            num_samples = len(self.data_cache["windows"])
            if self.data_cache["current_plot_index"] < num_samples - 1:
                self.data_cache["current_plot_index"] += 1
                self.update_plot()

    def previous_sample(self):
        if self.data_cache["windows"] is not None:
            if self.data_cache["current_plot_index"] > 0:
                self.data_cache["current_plot_index"] -= 1
                self.update_plot()

    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        # Add widgets for training parameters, start training, display results etc.
        layout.addWidget(QLabel("Model training features will be here."))
        # Example: Button to start training
        # start_training_button = QPushButton("Start Training")
        # layout.addWidget(start_training_button)

    def setup_pylsl_tab(self):
        layout = QVBoxLayout(self.pylsl_tab)
        layout.addWidget(QLabel("PyLSL integration for OpenBCI live data will be here."))


# Matplotlib Canvas Widget
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.clear_plot() # Initial clear

    def plot(self, data_window, title="EEG Data"):
        self.ax.clear()
        # Data window shape is (channels, timepoints)
        # We want to plot first 500 timepoints if longer
        timepoints_to_plot = min(data_window.shape[1], 500)
        
        for i in range(data_window.shape[0]): # Iterate over channels
            self.ax.plot(data_window[i, :timepoints_to_plot], label=f'Ch {i+1}')
        
        self.ax.set_title(title)
        self.ax.set_xlabel(f"Timepoints (first {timepoints_to_plot})")
        self.ax.set_ylabel("Amplitude")
        # self.ax.legend(loc='upper right', fontsize='small') # Optional: legend can be crowded
        self.ax.grid(True)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("No Data to Display")
        self.ax.set_xlabel("Timepoints")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.draw()


def start_gui():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    start_gui()
