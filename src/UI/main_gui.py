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
import matplotlib.pyplot as plt

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
try:
    from .StreamingWidget import StreamingWidget
except ImportError:
    try:
        from src.UI.StreamingWidget import StreamingWidget
    except ImportError:
        # Fallback if StreamingWidget is not available
        StreamingWidget = None

class TrainingThread(QThread):
    training_finished = pyqtSignal(object) # Signal to emit when training is done (can pass results)
    training_error = pyqtSignal(str)    # Signal to emit if an error occurs
    log_message = pyqtSignal(str)       # Signal to emit for log messages

    def __init__(self, training_params, data_cache, model_name): # Added model_name
        super().__init__()
        self.training_params = training_params
        self.data_cache = data_cache
        self.model_name = model_name # Store model_name

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

            # Redirect stdout/stderr to capture logs from train_main_script
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = self # Redirect stdout to this thread object
            sys.stderr = self # Redirect stderr to this thread object

            results = train_main_script(
                subjects_to_use=subjects_to_use,
                num_epochs_per_fold=self.training_params["epochs"],
                num_k_folds=self.training_params["k_folds"],
                learning_rate=self.training_params["learning_rate"],
                early_stopping_patience=self.training_params["early_stopping_patience"],
                batch_size=self.training_params["batch_size"],
                test_split_ratio=self.training_params["test_split_size"],
                data_base_path=self.data_cache["data_path"],
                model_name=self.model_name # Pass model_name to the training script
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
        self.setGeometry(100, 100, 1000, 800) # Adjusted size
        self.training_thread = None
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

        # --- Model Naming Group ---
        model_naming_group = QGroupBox("Model Naming")
        model_naming_layout = QFormLayout()
        self.model_name_input = QLineEdit(self.training_params_config["model_name"])
        self.model_name_input.setPlaceholderText("Enter a name for your model")
        model_naming_layout.addRow(QLabel("Model Name:"), self.model_name_input)
        model_naming_group.setLayout(model_naming_layout)
        layout.addWidget(model_naming_group)

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
                "model_name": "unnamed_model",
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
            QMessageBox.warning(self, "Training in Progress", "A training process is already running.")
            return

        # Update model name from input field
        self.current_model_name = self.model_name_input.text().strip()
        if not self.current_model_name:
            self.current_model_name = "unnamed_model" # Default if empty
            self.model_name_input.setText(self.current_model_name)
        
        self.training_params_config["model_name"] = self.current_model_name


        # Update params from custom fields if "custom" is selected
        if self.rb_custom_params.isChecked():
            self.update_training_params_from_custom_fields()
        else: # Reset to defaults if default is selected (or ensure defaults are used)
            self.training_params_config["epochs"] = 50 
            self.training_params_config["k_folds"] = 5
            self.training_params_config["learning_rate"] = 0.001
            self.training_params_config["early_stopping_patience"] = 5
            self.training_params_config["batch_size"] = 32
            self.training_params_config["test_split_size"] = 0.2
        
        current_model_name = self.model_name_input.text().strip()
        if not current_model_name:
            current_model_name = "unnamed_model" # Fallback if empty
            self.model_name_input.setText(current_model_name)
            QMessageBox.information(self, "Model Name", f"Model name was empty, defaulting to '{current_model_name}'.")


        self.training_params_config["train_subject_ids"] = self.training_subject_ids_input.text()

        self.append_log_message(f"Model Name: {current_model_name}")
        self.append_log_message(f"Using {'Custom' if self.rb_custom_params.isChecked() else 'Default'} parameters.")
        self.append_log_message(f"Parameters: {self.training_params_config}")
        self.append_log_message(f"Data path: {self.data_cache['data_path']}")
        
        self.btn_start_training.setEnabled(False)

        # Pass the current model name from the input field
        self.training_thread = TrainingThread(self.training_params_config, self.data_cache, current_model_name)
        self.training_thread.log_message.connect(self.append_log_message)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        self.training_thread.start()

    def append_log_message(self, message):
        self.training_results_display.append(message)

    def on_training_finished(self, results):
        self.btn_start_training.setEnabled(True)
        if results:
            # ... existing code ...
            QMessageBox.information(self, "Training Complete", f"Training for model '{self.current_model_name}' finished. Check logs for details.\\nMean CV Accuracy: {results.get('cv_mean_accuracy', 'N/A'):.4f}\\nFinal Test Accuracy: {results.get('final_test_accuracy', 'N/A'):.4f}")
            # Display the plot
            plot_path = results.get("plot_path")
            if plot_path and os.path.exists(plot_path):
                pixmap = QPixmap(plot_path)
                # self.training_plot_label.setPixmap(pixmap.scaled(self.training_plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                # For now, just log that the plot is available. Displaying it requires a QLabel.
                self.append_log_message(f"Plots saved to: {os.path.abspath(plot_path)}")
            else:
                self.append_log_message("Training plot image not found or path not returned.")

        else:
            QMessageBox.information(self, "Training Complete", f"Training for model '{self.current_model_name}' finished, but no results dictionary was returned. Check logs.")
        self.append_log_message(f"--- Training for model {self.current_model_name} ended ---")

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
