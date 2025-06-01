"""
Class:   DataManagementTab
Purpose: Provides the UI tab for managing EEG data, including loading, viewing, and navigating through data samples.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Integrates with BCIDataLoader and PlotCanvas for data handling and visualization.
"""

import sys
import os
import traceback
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt

# Assuming project_root and BCIDataLoader are accessible
# This might require adjustments based on your project structure
# Add project root to Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import BCIDataLoader
from .plot_canvas import PlotCanvas # Import PlotCanvas

class DataManagementTab(QWidget):
    def __init__(self, parent_main_window):
        super().__init__()
        self.main_window = parent_main_window # Reference to MainWindow to access data_cache and methods

        layout = QVBoxLayout(self)

        # --- Data Loading Group ---
        data_loading_group = QGroupBox("Load EEG Data")
        data_loading_layout = QFormLayout()

        self.data_path_label = QLabel(f"Selected Path: {self.main_window.data_cache['data_path']}")
        self.btn_browse_data_path = QPushButton("Browse EEG Data Directory")
        self.btn_browse_data_path.clicked.connect(self.browse_data_directory)
        data_loading_layout.addRow(self.btn_browse_data_path, self.data_path_label)

        self.subject_ids_input = QLineEdit()
        self.subject_ids_input.setPlaceholderText("e.g., 1,2,3 or all")
        data_loading_layout.addRow(QLabel("Subject IDs:"), self.subject_ids_input)

        self.btn_load_data = QPushButton("Load and Process Data")
        self.btn_load_data.clicked.connect(self.load_data_action)
        data_loading_layout.addWidget(self.btn_load_data)
        
        data_loading_group.setLayout(data_loading_layout)
        layout.addWidget(data_loading_group)

        # --- Data Summary Group ---
        data_summary_group = QGroupBox("Data Summary")
        data_summary_layout = QVBoxLayout()
        self.data_summary_display = QTextEdit()
        self.data_summary_display.setReadOnly(True)
        self.data_summary_display.setText(self.main_window.data_cache["data_summary"])
        data_summary_layout.addWidget(self.data_summary_display)
        data_summary_group.setLayout(data_summary_layout)
        layout.addWidget(data_summary_group)

        # --- EEG Plotting Group ---
        plot_group = QGroupBox("EEG Data Plot")
        plot_layout = QVBoxLayout()

        self.plot_canvas = PlotCanvas(self, width=8, height=4)
        plot_layout.addWidget(self.plot_canvas)

        nav_layout = QHBoxLayout()
        self.btn_prev_sample = QPushButton("Previous Sample")
        self.btn_prev_sample.clicked.connect(self.previous_sample)
        self.btn_prev_sample.setEnabled(False)
        nav_layout.addWidget(self.btn_prev_sample)

        self.current_sample_label = QLabel("Sample: -/-")
        self.current_sample_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.current_sample_label)

        self.btn_next_sample = QPushButton("Next Sample")
        self.btn_next_sample.clicked.connect(self.next_sample)
        self.btn_next_sample.setEnabled(False)
        nav_layout.addWidget(self.btn_next_sample)
        plot_layout.addLayout(nav_layout)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        layout.addStretch()
        self.setLayout(layout)
        self.update_plot() # Initial plot update

    def browse_data_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select EEG Data Directory", self.main_window.data_cache["data_path"])
        if dir_path:
            self.data_path_label.setText(f"Selected Path: {dir_path}")
            self.main_window.data_cache["data_path"] = dir_path
            print(f"Selected data directory: {dir_path}")

    def load_data_action(self):
        subjects_str = self.subject_ids_input.text().strip()
        data_path = self.main_window.data_cache["data_path"]

        if not data_path or data_path == "Not selected":
            self.data_summary_display.setText("Error: Please select a data directory first.")
            return

        if not os.path.isdir(data_path):
            potential_path = os.path.join(project_root, data_path)
            if os.path.isdir(potential_path):
                data_path = potential_path
                self.main_window.data_cache["data_path"] = data_path
                self.data_path_label.setText(f"Selected Path: {data_path}")
            else:
                self.data_summary_display.setText(f"Error: Data directory '{data_path}' not found.")
                return

        subjects_list_for_loader = None
        current_subjects_display = "all"

        if not subjects_str:
            subjects_list_for_loader = self.main_window.data_cache["subjects_list"]
            if self.main_window.data_cache["subjects_list"] is not None:
                current_subjects_display = ",".join(map(str, self.main_window.data_cache["subjects_list"]))
            self.data_summary_display.setText(f"Using previously specified subjects: {current_subjects_display}")
        elif subjects_str.lower() == 'all':
            subjects_list_for_loader = None
            self.main_window.data_cache["subjects_list"] = None
            current_subjects_display = "all"
        else:
            try:
                subjects_list_for_loader = [int(s.strip()) for s in subjects_str.split(',') if s.strip()]
                if not subjects_list_for_loader:
                    self.data_summary_display.setText("Error: Invalid subject IDs. Please enter comma-separated numbers or 'all'.")
                    return
                self.main_window.data_cache["subjects_list"] = subjects_list_for_loader
                current_subjects_display = subjects_str
            except ValueError:
                self.data_summary_display.setText("Error: Invalid subject IDs. Please enter comma-separated numbers or 'all'.")
                return
        
        self.data_summary_display.setText(f"Loading data from: {data_path}\nSubjects: {current_subjects_display}...")

        try:
            loader = BCIDataLoader(data_path=data_path, subjects=subjects_list_for_loader)
            windows, labels, subject_ids_loaded = loader.load_all_subjects()

            if windows.size == 0:
                summary = "No data was loaded. Please check data path and subject IDs."
                self.main_window.data_cache["data_summary"] = summary
            else:
                self.main_window.data_cache["windows"] = windows
                self.main_window.data_cache["labels"] = labels
                self.main_window.data_cache["subject_ids"] = subject_ids_loaded
                
                summary = (
                    f"Data loaded successfully!\n"
                    f"  Total windows: {windows.shape[0]}\n"
                    f"  Window shape: ({windows.shape[1]}, {windows.shape[2]}) (channels, timepoints)\n"
                    f"  Number of unique subjects: {len(np.unique(subject_ids_loaded))}\n"
                    f"  Class distribution: {np.bincount(labels.astype(int))}"
                )
                self.main_window.data_cache["data_summary"] = summary
            
            self.data_summary_display.setText(summary)
            print(summary)

        except FileNotFoundError as e:
            error_msg = f"Error: {e}"
            self.data_summary_display.setText(error_msg)
            self.main_window.data_cache["data_summary"] = error_msg
            print(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            self.data_summary_display.setText(error_msg)
            self.main_window.data_cache["data_summary"] = error_msg
            print(error_msg)
            traceback.print_exc()
        finally:
            self.main_window.data_cache["current_plot_index"] = 0
            self.update_plot()
            self.update_plot_navigation()

    def update_plot(self):
        self.plot_canvas.clear_plot()
        if self.main_window.data_cache["windows"] is not None and self.main_window.data_cache["windows"].size > 0:
            idx = self.main_window.data_cache["current_plot_index"]
            
            if 0 <= idx < len(self.main_window.data_cache["windows"]):
                sample_window = self.main_window.data_cache["windows"][idx]
                label_val = self.main_window.data_cache["labels"][idx]
                subject_id_val = self.main_window.data_cache["subject_ids"][idx]

                label_text = self.main_window.data_cache["label_map"].get(label_val, f"Unknown Label ({label_val})")
                plot_title = f"Subject: {subject_id_val} - Label: {label_text} (Sample {idx + 1})"
                
                if sample_window.ndim == 3 and sample_window.shape[0] == 1:
                    sample_window = sample_window.squeeze(0)
                elif sample_window.ndim != 2:
                    self.data_summary_display.append(f"\nSkipping plot for sample {idx+1}: Unexpected window dimensions {sample_window.shape}")
                    self.plot_canvas.axes.set_title(f"Error: Cannot plot sample {idx+1}")
                    self.plot_canvas.draw()
                    return

                self.plot_canvas.plot(sample_window, title=plot_title)
            else:
                self.plot_canvas.axes.set_title("No data at current index")
                self.plot_canvas.draw()
        else:
            self.plot_canvas.axes.set_title("No EEG Data Loaded")
            self.plot_canvas.draw()
        self.update_plot_navigation()

    def update_plot_navigation(self):
        if self.main_window.data_cache["windows"] is not None and self.main_window.data_cache["windows"].size > 0:
            total_samples = len(self.main_window.data_cache["windows"])
            current_idx_display = self.main_window.data_cache["current_plot_index"] + 1
            self.current_sample_label.setText(f"Sample: {current_idx_display}/{total_samples}")

            self.btn_prev_sample.setEnabled(self.main_window.data_cache["current_plot_index"] > 0)
            self.btn_next_sample.setEnabled(self.main_window.data_cache["current_plot_index"] < total_samples - 1)
        else:
            self.current_sample_label.setText("Sample: -/-")
            self.btn_prev_sample.setEnabled(False)
            self.btn_next_sample.setEnabled(False)

    def next_sample(self):
        if self.main_window.data_cache["windows"] is not None:
            num_samples = len(self.main_window.data_cache["windows"])
            if self.main_window.data_cache["current_plot_index"] < num_samples - 1:
                self.main_window.data_cache["current_plot_index"] += 1
                self.update_plot()

    def previous_sample(self):
        if self.main_window.data_cache["windows"] is not None:
            if self.main_window.data_cache["current_plot_index"] > 0:
                self.main_window.data_cache["current_plot_index"] -= 1
                self.update_plot()
