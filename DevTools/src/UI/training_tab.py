"""
Classes: TrainingThread, TrainingTab
Purpose: Provides the UI tab for configuring and running model training sessions.
         TrainingThread handles the training process in a separate thread to keep the UI responsive.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Integrates with the train_model script for backend training logic.
"""

import os
import sys
import traceback 
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QRadioButton, QSpinBox,
    QDoubleSpinBox, QPushButton, QTextEdit, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.train_model import main as train_main_script

class TrainingThread(QThread):
    training_finished = pyqtSignal(object)
    training_error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, training_params, data_cache, model_name):
        super().__init__()
        self.training_params = training_params
        self.data_cache = data_cache
        self.model_name = model_name  # Add missing model_name attribute
        
    def run(self):
        try:
            self.log_message.emit(f"Starting training for model: {self.model_name}...")
            
            subjects_to_use_str = self.training_params.get("train_subject_ids", "all")
            
            subjects_to_use_str = self.training_params.get("train_subject_ids", "all")
            subjects_to_use = None
            if subjects_to_use_str.lower() != "all":
                try:
                    subjects_to_use = [int(s.strip()) for s in subjects_to_use_str.split(',') if s.strip()]
                except ValueError:
                    self.training_error.emit(f"Invalid subject IDs format: {subjects_to_use_str}. Use comma-separated numbers or 'all'.")
                    return
            
            if self.data_cache["windows"] is None or self.data_cache["labels"] is None:
                self.training_error.emit("No data loaded. Please load data before starting training.")
                return

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = self
            sys.stderr = self

            results = train_main_script(
                subjects_to_use=subjects_to_use,
                num_epochs_per_fold=self.training_params["epochs"],
                num_k_folds=self.training_params["k_folds"], # Added k_folds
                batch_size=self.training_params["batch_size"],
                test_split_ratio=self.training_params["test_split_size"],
                learning_rate=self.training_params["learning_rate"],
                early_stopping_patience=self.training_params["early_stopping_patience"],
                data_base_path=self.data_cache["data_path"],
                model_name=self.model_name
                # model_params_json is not set from UI, will default to None in train_main_script
            )
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            self.log_message.emit("Training finished.")
            self.training_finished.emit(results)
        except Exception as e:
            sys.stdout = original_stdout # Restore stdout in case of error
            sys.stderr = original_stderr # Restore stderr in case of error
            self.training_error.emit(f"An error occurred during training: {str(e)}\n{traceback.format_exc()}")

    def write(self, message):
        self.log_message.emit(message.strip())

    def flush(self):
        pass

class TrainingTab(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.main_window = parent_main_window
        self.training_thread = None

        layout = QVBoxLayout(self)

        # --- Model Naming Group ---
        model_naming_group = QGroupBox("Model Naming")
        model_naming_layout = QFormLayout()
        self.model_name_input = QLineEdit(self.main_window.training_params_config["model_name"])
        self.model_name_input.setPlaceholderText("Enter a name for your model")
        self.model_name_input.textChanged.connect(self.update_model_name_config)
        model_naming_layout.addRow(QLabel("Model Name:"), self.model_name_input)
        model_naming_group.setLayout(model_naming_layout)
        layout.addWidget(model_naming_group)

        # --- Parameter Configuration Group ---
        param_config_group = QGroupBox("Parameter Configuration")
        param_config_layout = QVBoxLayout()

        self.rb_default_params = QRadioButton("Use Default Training Parameters")
        self.rb_default_params.setChecked(self.main_window.training_params_config["use_default_params"])
        self.rb_default_params.toggled.connect(self.toggle_custom_params_group)
        param_config_layout.addWidget(self.rb_default_params)

        self.rb_custom_params = QRadioButton("Use Custom Training Parameters")
        self.rb_custom_params.setChecked(not self.main_window.training_params_config["use_default_params"])
        self.rb_custom_params.toggled.connect(self.toggle_custom_params_group)
        param_config_layout.addWidget(self.rb_custom_params)

        param_config_group.setLayout(param_config_layout)
        layout.addWidget(param_config_group)

        # --- Custom Parameters Group ---
        self.custom_params_group = QGroupBox("Custom Training Parameters")
        custom_params_layout = QFormLayout()
        self.main_window.custom_param_inputs = {} # Store inputs for easy access

        # Epochs
        self.main_window.custom_param_inputs["epochs"] = QSpinBox()
        self.main_window.custom_param_inputs["epochs"].setRange(1, 10000)
        self.main_window.custom_param_inputs["epochs"].setValue(self.main_window.training_params_config["epochs"])
        custom_params_layout.addRow(QLabel("Epochs:"), self.main_window.custom_param_inputs["epochs"])

        # K-Folds
        self.main_window.custom_param_inputs["k_folds"] = QSpinBox()
        self.main_window.custom_param_inputs["k_folds"].setRange(1, 100)
        self.main_window.custom_param_inputs["k_folds"].setValue(self.main_window.training_params_config["k_folds"])
        custom_params_layout.addRow(QLabel("K-Folds:"), self.main_window.custom_param_inputs["k_folds"])

        # Learning Rate
        self.main_window.custom_param_inputs["learning_rate"] = QDoubleSpinBox()
        self.main_window.custom_param_inputs["learning_rate"].setDecimals(5)
        self.main_window.custom_param_inputs["learning_rate"].setRange(0.00001, 1.0)
        self.main_window.custom_param_inputs["learning_rate"].setSingleStep(0.0001)
        self.main_window.custom_param_inputs["learning_rate"].setValue(self.main_window.training_params_config["learning_rate"])
        custom_params_layout.addRow(QLabel("Learning Rate:"), self.main_window.custom_param_inputs["learning_rate"])

        # Early Stopping Patience
        self.main_window.custom_param_inputs["early_stopping_patience"] = QSpinBox()
        self.main_window.custom_param_inputs["early_stopping_patience"].setRange(1, 1000)
        self.main_window.custom_param_inputs["early_stopping_patience"].setValue(self.main_window.training_params_config["early_stopping_patience"])
        custom_params_layout.addRow(QLabel("Early Stopping Patience:"), self.main_window.custom_param_inputs["early_stopping_patience"])

        # Batch Size
        self.main_window.custom_param_inputs["batch_size"] = QSpinBox()
        self.main_window.custom_param_inputs["batch_size"].setRange(1, 1024)
        self.main_window.custom_param_inputs["batch_size"].setValue(self.main_window.training_params_config["batch_size"])
        custom_params_layout.addRow(QLabel("Batch Size:"), self.main_window.custom_param_inputs["batch_size"])

        # Test Split Size
        self.main_window.custom_param_inputs["test_split_size"] = QDoubleSpinBox()
        self.main_window.custom_param_inputs["test_split_size"].setDecimals(2)
        self.main_window.custom_param_inputs["test_split_size"].setRange(0.01, 0.99)
        self.main_window.custom_param_inputs["test_split_size"].setSingleStep(0.01)
        self.main_window.custom_param_inputs["test_split_size"].setValue(self.main_window.training_params_config["test_split_size"])
        custom_params_layout.addRow(QLabel("Test Split Ratio:"), self.main_window.custom_param_inputs["test_split_size"])
        
        # Train Subject IDs
        self.main_window.custom_param_inputs["train_subject_ids"] = QLineEdit()
        self.main_window.custom_param_inputs["train_subject_ids"].setPlaceholderText("e.g., 1,2,3 or all")
        self.main_window.custom_param_inputs["train_subject_ids"].setText(self.main_window.training_params_config["train_subject_ids"])
        custom_params_layout.addRow(QLabel("Train Subject IDs:"), self.main_window.custom_param_inputs["train_subject_ids"])


        self.custom_params_group.setLayout(custom_params_layout)
        layout.addWidget(self.custom_params_group)
        self.toggle_custom_params_group() # Set initial state

        # --- Training Action Group ---
        training_action_group = QGroupBox("Training Actions")
        training_action_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.clicked.connect(self.start_training_action)
        training_action_layout.addWidget(self.btn_start_training)
        training_action_group.setLayout(training_action_layout)
        layout.addWidget(training_action_group)

        # --- Training Log Group ---
        training_log_group = QGroupBox("Training Log & Results")
        training_log_layout = QVBoxLayout()
        self.training_log_display = QTextEdit()
        self.training_log_display.setReadOnly(True)
        layout.addStretch()
        self.setLayout(layout)
    
    def update_model_name_config(self, text):
        self.main_window.training_params_config["model_name"] = text
        self.main_window.current_model_name = text # Also update MainWindow's current_model_name
        layout.addStretch()
        self.setLayout(layout)

    def update_model_name_config(self, text):
        self.main_window.training_params_config["model_name"] = text
        self.main_window.current_model_name = text # Also update MainWindow's current_model_name

    def toggle_custom_params_group(self):
        use_custom = self.rb_custom_params.isChecked()
        self.custom_params_group.setEnabled(use_custom)
        self.main_window.training_params_config["use_default_params"] = not use_custom

    def start_training_action(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training in Progress", "A training process is already running.")
            return

        # Update params from UI
        current_params = {}
        if self.main_window.training_params_config["use_default_params"]:
            # Use default values (already in training_params_config)
            current_params = {
                "epochs": self.main_window.training_params_config["epochs"],
                "k_folds": self.main_window.training_params_config["k_folds"],
                "learning_rate": self.main_window.training_params_config["learning_rate"],
                "early_stopping_patience": self.main_window.training_params_config["early_stopping_patience"],
                "batch_size": self.main_window.training_params_config["batch_size"],
                "test_split_size": self.main_window.training_params_config["test_split_size"],
                "train_subject_ids": self.main_window.training_params_config["train_subject_ids"]
            }
        else:
            # Use custom values from input fields
            try:
                current_params["epochs"] = self.main_window.custom_param_inputs["epochs"].value()
                current_params["k_folds"] = self.main_window.custom_param_inputs["k_folds"].value()
                current_params["learning_rate"] = self.main_window.custom_param_inputs["learning_rate"].value()
                current_params["early_stopping_patience"] = self.main_window.custom_param_inputs["early_stopping_patience"].value()
                current_params["batch_size"] = self.main_window.custom_param_inputs["batch_size"].value() # Added
                current_params["test_split_size"] = self.main_window.custom_param_inputs["test_split_size"].value() # Added
                current_params["train_subject_ids"] = self.main_window.custom_param_inputs["train_subject_ids"].text() # Added
                
                # Update the main config as well with all custom values
                self.main_window.training_params_config.update(current_params)
            except Exception as e:
                self.training_log_display.append(f"Error reading custom parameters: {e}")
                return

        model_name = self.model_name_input.text().strip()
        if not model_name:
            model_name = "unnamed_model"
            self.model_name_input.setText(model_name) # Update UI if empty
        self.main_window.current_model_name = model_name # Update main window's model name

        self.training_log_display.clear()
        self.training_log_display.append(f"Preparing to train model: {model_name}")
        self.training_log_display.append(f"Parameters: {current_params}")

        self.training_thread = TrainingThread(current_params, self.main_window.data_cache, model_name)
        self.training_thread.log_message.connect(self.append_log_message)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        
        self.btn_start_training.setEnabled(False)
        self.training_thread.start()

    def append_log_message(self, message):
        self.training_log_display.append(message)

    def on_training_finished(self, results):
        self.training_log_display.append("\n--- Training Complete ---")
        if results:
            self.training_log_display.append("Results:")
            # Assuming results is a dictionary or an object with a __str__ method
            # For more detailed display, you might need to format `results`
            if isinstance(results, dict):
                for key, value in results.items():
                    self.training_log_display.append(f"  {key}: {value}")
            else:
                self.training_log_display.append(str(results))
        else:
            self.training_log_display.append("Training finished, but no results were returned.")
        self.btn_start_training.setEnabled(True)
        QMessageBox.information(self, "Training Complete", f"Training for model '{self.main_window.current_model_name}' finished successfully.")


    def on_training_error(self, error_message):
        self.training_log_display.append(f"--- TRAINING ERROR ---")
        self.training_log_display.append(error_message)
        self.btn_start_training.setEnabled(True)
        QMessageBox.critical(self, "Training Error", f"An error occurred during training for model '{self.main_window.current_model_name}'.\nCheck logs for details.")


