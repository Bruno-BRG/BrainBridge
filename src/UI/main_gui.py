import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTabWidget, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QGroupBox, QTextEdit, QHBoxLayout,
    QRadioButton, QSpinBox, QDoubleSpinBox, QMessageBox # Added QMessageBox
)
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
import matplotlib.pyplot as plt

class TrainingThread(QThread):
    training_finished = pyqtSignal(object) # Signal to emit when training is done (can pass results)
    training_error = pyqtSignal(str)    # Signal to emit if an error occurs
    log_message = pyqtSignal(str)       # Signal to emit for log messages

    def __init__(self, training_params, data_cache):
        super().__init__()
        self.training_params = training_params
        self.data_cache = data_cache

    def run(self):
        try:
            # Prepare arguments for train_main_script
            # This assumes train_main_script can handle these params directly
            # or we adapt them here.
            self.log_message.emit("Starting training...")
            
            subjects_to_use_str = self.training_params.get("train_subject_ids", "all")
            subjects_to_use = None
            if subjects_to_use_str.lower() != "all":
                try:
                    subjects_to_use = [int(s.strip()) for s in subjects_to_use_str.split(',')]
                except ValueError:
                    self.training_error.emit(f"Invalid subject IDs format: {subjects_to_use_str}. Use comma-separated numbers or 'all'.")
                    return
            
            # Ensure data is loaded
            if self.data_cache["windows"] is None or self.data_cache["labels"] is None:
                self.training_error.emit("No data loaded. Please load data before starting training.")
                return

            # Call the training script
            # Note: train_main_script might need adjustments to accept all these GUI params
            # and to provide progress/results back in a way the GUI can use.
            # For now, we'll assume it runs and we can capture stdout/stderr or it writes to a log file.
            
            # Redirect stdout/stderr to capture logs from train_main_script
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = self # Redirect stdout to this thread object
            sys.stderr = self # Redirect stderr to this thread object

            results = train_main_script(
                subjects_to_use=subjects_to_use, # This needs to be handled by train_main_script
                num_epochs_per_fold=self.training_params["epochs"],
                num_k_folds=self.training_params["k_folds"],
                learning_rate=self.training_params["learning_rate"],
                early_stopping_patience=self.training_params["early_stopping_patience"],
                batch_size=self.training_params["batch_size"],
                test_split_ratio=self.training_params["test_split_size"],
                data_base_path=self.data_cache["data_path"], # Pass the data path used for loading
                # We need to pass the actual loaded data, not just the path again, 
                # or train_main_script needs to be able to use pre-loaded data.
                # This is a simplification for now.
                # Ideally, train_main_script would accept windows, labels, subject_ids directly.
            )
            
            sys.stdout = original_stdout # Restore stdout
            sys.stderr = original_stderr # Restore stderr

            self.log_message.emit("Training finished.")
            self.training_finished.emit(results) # Pass results back
        except Exception as e:
            sys.stdout = original_stdout # Restore stdout in case of error
            sys.stderr = original_stderr # Restore stderr in case of error
            self.training_error.emit(f"An error occurred during training: {str(e)}")

    # Implement write and flush to capture print statements from train_main_script
    def write(self, message):
        self.log_message.emit(message.strip())

    def flush(self):
        pass # Usually not needed for simple redirection

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Application")
        self.setGeometry(100, 100, 900, 700)
        self.training_thread = None # Initialize training thread attribute

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

    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)

        # --- Parameter Configuration Group ---
        param_config_group = QGroupBox("Parameter Configuration")
        param_config_layout = QVBoxLayout()

        self.rb_default_params = QRadioButton("Use Default Training Parameters")
        self.rb_default_params.setChecked(self.training_params_config["use_default_params"])
        self.rb_default_params.toggled.connect(self.toggle_custom_params_group)
        param_config_layout.addWidget(self.rb_default_params)

        self.rb_custom_params = QRadioButton("Use Custom Training Parameters")
        self.rb_custom_params.setChecked(not self.training_params_config["use_default_params"])
        self.rb_custom_params.toggled.connect(self.toggle_custom_params_group) # Ensure this is also connected
        param_config_layout.addWidget(self.rb_custom_params)

        param_config_group.setLayout(param_config_layout)
        layout.addWidget(param_config_group)

        # --- Custom Parameters Group (initially disabled) ---
        self.custom_params_group = QGroupBox("Custom Training Parameters")
        custom_params_layout = QFormLayout()

        # Epochs
        self.custom_param_inputs["epochs"] = QSpinBox()
        self.custom_param_inputs["epochs"].setRange(1, 10000)
        self.custom_param_inputs["epochs"].setValue(self.training_params_config["epochs"])
        custom_params_layout.addRow(QLabel("Epochs:"), self.custom_param_inputs["epochs"])

        # K-Folds
        self.custom_param_inputs["k_folds"] = QSpinBox()
        self.custom_param_inputs["k_folds"].setRange(1, 100)
        self.custom_param_inputs["k_folds"].setValue(self.training_params_config["k_folds"])
        custom_params_layout.addRow(QLabel("K-Folds:"), self.custom_param_inputs["k_folds"])

        # Learning Rate
        self.custom_param_inputs["learning_rate"] = QDoubleSpinBox()
        self.custom_param_inputs["learning_rate"].setDecimals(5)
        self.custom_param_inputs["learning_rate"].setRange(0.00001, 1.0)
        self.custom_param_inputs["learning_rate"].setSingleStep(0.0001)
        self.custom_param_inputs["learning_rate"].setValue(self.training_params_config["learning_rate"])
        custom_params_layout.addRow(QLabel("Learning Rate:"), self.custom_param_inputs["learning_rate"])

        # Early Stopping Patience
        self.custom_param_inputs["early_stopping_patience"] = QSpinBox()
        self.custom_param_inputs["early_stopping_patience"].setRange(1, 1000)
        self.custom_param_inputs["early_stopping_patience"].setValue(self.training_params_config["early_stopping_patience"])
        custom_params_layout.addRow(QLabel("Early Stopping Patience:"), self.custom_param_inputs["early_stopping_patience"])

        # Batch Size
        self.custom_param_inputs["batch_size"] = QSpinBox()
        self.custom_param_inputs["batch_size"].setRange(1, 1024)
        self.custom_param_inputs["batch_size"].setValue(self.training_params_config["batch_size"])
        custom_params_layout.addRow(QLabel("Batch Size:"), self.custom_param_inputs["batch_size"])

        # Test Split Size
        self.custom_param_inputs["test_split_size"] = QDoubleSpinBox()
        self.custom_param_inputs["test_split_size"].setDecimals(2)
        self.custom_param_inputs["test_split_size"].setRange(0.01, 0.99)
        self.custom_param_inputs["test_split_size"].setSingleStep(0.01)
        self.custom_param_inputs["test_split_size"].setValue(self.training_params_config["test_split_size"])
        custom_params_layout.addRow(QLabel("Test Split Ratio:"), self.custom_param_inputs["test_split_size"])

        self.custom_params_group.setLayout(custom_params_layout)
        layout.addWidget(self.custom_params_group)

        # --- Training Subject Input Group ---
        training_subject_group = QGroupBox("Training Subjects")
        training_subject_layout = QFormLayout()
        self.training_subject_ids_input = QLineEdit()
        self.training_subject_ids_input.setPlaceholderText("e.g., 1,2,3 or all (loaded data)")
        self.training_subject_ids_input.setText(self.training_params_config["train_subject_ids"])
        self.training_subject_ids_input.textChanged.connect(self.update_training_params_from_custom_fields) # Connect to update config
        training_subject_layout.addRow(QLabel("Subject IDs for Training:"), self.training_subject_ids_input)
        training_subject_group.setLayout(training_subject_layout)
        layout.addWidget(training_subject_group)
        
        # --- Training Action Group ---
        training_action_group = QGroupBox("Train Model")
        training_action_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.clicked.connect(self.start_training_action) # Connect the button
        training_action_layout.addWidget(self.btn_start_training)
        training_action_group.setLayout(training_action_layout)
        layout.addWidget(training_action_group)

        # --- Training Progress/Results Display ---
        training_results_group = QGroupBox("Training Log & Results")
        training_results_layout = QVBoxLayout()
        self.training_results_display = QTextEdit()
        self.training_results_display.setReadOnly(True)
        self.training_results_display.setPlaceholderText("Training logs and results will appear here.")
        training_results_layout.addWidget(self.training_results_display)
        training_results_group.setLayout(training_results_layout)
        layout.addWidget(training_results_group)

        layout.addStretch()
        self.training_tab.setLayout(layout)

        # Initialize state of custom params group
        self.toggle_custom_params_group()

    def toggle_custom_params_group(self):
        # Ensure only one radio button's toggle signal is primarily used or check sender
        use_custom = self.rb_custom_params.isChecked()
        self.custom_params_group.setEnabled(use_custom)
        self.training_params_config["use_default_params"] = not use_custom

        if not use_custom:
            self.training_results_display.append("Switched to default training parameters.")
            # Reset custom fields to defaults if needed or just use the stored default config
            self.update_training_params_from_custom_fields() # Update to reflect default if necessary
        else:
            self.training_results_display.append("Switched to custom training parameters.")
            self.update_training_params_from_custom_fields() # Ensure config reflects custom field values

    def update_training_params_from_custom_fields(self):
        if not self.training_params_config["use_default_params"]:
            for name, widget in self.custom_param_inputs.items():
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    self.training_params_config[name] = widget.value()
                elif isinstance(widget, QLineEdit):
                     self.training_params_config[name] = widget.text()
            self.training_params_config["train_subject_ids"] = self.training_subject_ids_input.text()
        else: # Using default params, so reset to defaults
            default_config = {
                "epochs": 50, "k_folds": 5, "learning_rate": 0.001,
                "early_stopping_patience": 5, "batch_size": 32, "test_split_size": 0.2,
                "train_subject_ids": "all"
            }
            for name, value in default_config.items():
                self.training_params_config[name] = value
                if name in self.custom_param_inputs: # Update UI fields as well
                    self.custom_param_inputs[name].setValue(value)
                if name == "train_subject_ids":
                    self.training_subject_ids_input.setText(value)
        # print(f"Updated training_params_config: {self.training_params_config}") # For debugging

    def start_training_action(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training In Progress", "A training process is already running.")
            return

        self.update_training_params_from_custom_fields() # Ensure params are current
        self.training_results_display.clear() # Clear previous logs
        self.training_results_display.append(f"Preparing to train with parameters: {self.training_params_config}")

        # Create and start the training thread
        self.training_thread = TrainingThread(self.training_params_config.copy(), self.data_cache.copy()) # Pass copies
        self.training_thread.log_message.connect(self.append_log_message)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        self.btn_start_training.setEnabled(False) # Disable button during training
        self.training_thread.start()

    def append_log_message(self, message):
        self.training_results_display.append(message)

    def on_training_finished(self, results):
        self.training_results_display.append("--- Training Successfully Completed ---")
        # self.training_results_display.append(f"Results: {results}") # Or format nicely
        QMessageBox.information(self, "Training Complete", "Model training finished successfully.")
        self.btn_start_training.setEnabled(True) # Re-enable button

    def on_training_error(self, error_message):
        self.training_results_display.append(f"--- Training Error ---")
        self.training_results_display.append(error_message)
        QMessageBox.critical(self, "Training Error", error_message)
        self.btn_start_training.setEnabled(True) # Re-enable button

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
        if self.data_cache["windows"] is not None and self.data_cache["windows"].size > 0:
            total_samples = len(self.data_cache["windows"])
            current_idx_display = self.data_cache["current_plot_index"] + 1
            self.current_sample_label.setText(f"Sample: {current_idx_display}/{total_samples}")

            self.btn_prev_sample.setEnabled(self.data_cache["current_plot_index"] > 0)
            self.btn_next_sample.setEnabled(self.data_cache["current_plot_index"] < total_samples - 1)
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

    def setup_pylsl_tab(self):
        layout = QVBoxLayout(self.pylsl_tab)
        layout.addWidget(QLabel("PyLSL integration for OpenBCI live data will be here."))


# Matplotlib Canvas Widget
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)
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
