"""
BCI Motor Imagery Classification GUI Application

A PyQt5-based graphical user interface for the EEG motor imagery classification system.
Provides functionality for data loading, model training, testing, and visualization.

Features:
1. Data Loading Tab - Select subjects and configure data loading
2. Training Tab - Configure and run model training with visualization
3. Testing Tab - Load trained models and test on data windows
4. Additional functionality as needed

Author: GitHub Copilot
License: BSD (3-clause)
"""

import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget,
    QGroupBox, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox,
    QSlider, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Update imports to use relative imports
from .PlotCanvas import PlotCanvas  # Change from UI.PlotCanvas to .PlotCanvas
from .data_tab.DataLoadThread import DataLoadThread  # Change from UI.data_tab.DataLoadThread to .data_tab.DataLoadThread
from src.data.data_loader import BCIDataLoader, create_data_loaders  # This remains the same as it's from src package
from .training_tab.TrainingManager import TrainingManager  # Now just handles UI

# Set the style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BCIMainWindow(QMainWindow):
    
    """Main window for BCI GUI application with tabs for data loading, training, and testing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BCI Motor Imagery Classification")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Add tabs (basic placeholders for now)
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data Loading")
        self._create_data_loading_tab() # Placeholder for tab content

        self.training_tab = QWidget()
        self.tabs.addTab(self.training_tab, "Training")
        self._create_training_tab() # Placeholder for tab content

        self.testing_tab = QWidget()
        self.tabs.addTab(self.testing_tab, "Testing")
        self._create_testing_tab() # Placeholder for tab content
        
        # Status Bar
        self.statusBar().showMessage("Ready")
    def _create_data_loading_tab(self):
        """Create comprehensive data loading tab with subject selection and visualization"""
        main_layout = QHBoxLayout(self.data_tab)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Subject Selection Group
        subject_group = QGroupBox("Subject Selection")
        subject_layout = QVBoxLayout()
        
        # Quick select buttons
        quick_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Select None")
        self.select_first_10_btn = QPushButton("First 10")
        quick_layout.addWidget(self.select_all_btn)
        quick_layout.addWidget(self.select_none_btn)
        quick_layout.addWidget(self.select_first_10_btn)
        subject_layout.addLayout(quick_layout)
        
        # Subject checkboxes in scrollable area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.subject_layout = QVBoxLayout(scroll_widget)
        self.subject_checkboxes = {}
        
        # Initialize available subjects (1-109 based on the codebase)
        available_subjects = list(range(1, 110))
        
        for subject_id in available_subjects:
            checkbox = QCheckBox(f"Subject {subject_id:03d}")
            checkbox.setChecked(subject_id <= 10)  # Default: first 10 selected
            self.subject_checkboxes[subject_id] = checkbox
            self.subject_layout.addWidget(checkbox)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(300)
        subject_layout.addWidget(scroll_area)
        
        # Load button
        self.load_data_btn = QPushButton("Load Selected Subjects")
        self.load_data_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        subject_layout.addWidget(self.load_data_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        subject_layout.addWidget(self.progress_bar)
        
        subject_group.setLayout(subject_layout)
        left_layout.addWidget(subject_group)
        
        # Data Info Group
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        
        self.total_samples_label = QLabel("Total Samples: 0")
        self.current_sample_label = QLabel("Current Sample: N/A")
        self.sample_label_info = QLabel("Label: N/A")
        self.sample_subject_info = QLabel("Subject: N/A")
        # self.sample_run_info = QLabel("Run: N/A") # Removed
        
        info_layout.addWidget(self.total_samples_label)
        info_layout.addWidget(self.current_sample_label)
        info_layout.addWidget(self.sample_label_info)
        info_layout.addWidget(self.sample_subject_info)
        # info_layout.addWidget(self.sample_run_info) # Removed
        
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        # Navigation Group
        nav_group = QGroupBox("Sample Navigation")
        nav_layout = QVBoxLayout()
        
        # Navigation buttons
        nav_btn_layout = QHBoxLayout()
        self.first_btn = QPushButton("⏮ First")
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.last_btn = QPushButton("Last ⏭")
        
        nav_btn_layout.addWidget(self.first_btn)
        nav_btn_layout.addWidget(self.prev_btn)
        nav_btn_layout.addWidget(self.next_btn)
        nav_btn_layout.addWidget(self.last_btn)
        nav_layout.addLayout(nav_btn_layout)
        
        # Sample slider
        self.sample_slider = QSlider(Qt.Horizontal)
        self.sample_slider.setMinimum(0)
        self.sample_slider.setMaximum(0)
        self.sample_slider.setValue(0)
        nav_layout.addWidget(QLabel("Sample Index:"))
        nav_layout.addWidget(self.sample_slider)
        
        # Direct navigation
        direct_layout = QHBoxLayout()
        direct_layout.addWidget(QLabel("Go to:"))
        self.goto_spinbox = QSpinBox()
        self.goto_spinbox.setMinimum(1)
        self.goto_spinbox.setMaximum(1)
        self.goto_btn = QPushButton("Go")
        direct_layout.addWidget(self.goto_spinbox)
        direct_layout.addWidget(self.goto_btn)
        nav_layout.addLayout(direct_layout)
        
        nav_group.setLayout(nav_layout)
        left_layout.addWidget(nav_group)
        
        # Filter/Display Options Group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_channels_spinbox = QSpinBox()
        self.show_channels_spinbox.setMinimum(1)
        self.show_channels_spinbox.setMaximum(16)
        self.show_channels_spinbox.setValue(16) # Changed default to 16
        
        display_layout.addWidget(QLabel("Channels to display:"))
        display_layout.addWidget(self.show_channels_spinbox)
        
        self.filter_by_label = QComboBox()
        self.filter_by_label.addItems(["All Samples", "Left Hand Only", "Right Hand Only"])
        display_layout.addWidget(QLabel("Filter by label:"))
        display_layout.addWidget(self.filter_by_label)
        
        display_group.setLayout(display_layout)
        left_layout.addWidget(display_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Plot area
        self.plot_canvas = PlotCanvas(width=10, height=8)
        right_layout.addWidget(self.plot_canvas)
        
        main_layout.addWidget(right_panel)
        
        # Initialize data storage
        self.loaded_data = None
        self.current_sample_idx = 0
        self.filtered_indices = []
        
        # Connect signals
        self.select_all_btn.clicked.connect(self._select_all_subjects)
        self.select_none_btn.clicked.connect(self._select_no_subjects)
        self.select_first_10_btn.clicked.connect(self._select_first_10_subjects)
        self.load_data_btn.clicked.connect(self._load_data)
        
        self.first_btn.clicked.connect(self._go_to_first)
        self.prev_btn.clicked.connect(self._go_to_previous)
        self.next_btn.clicked.connect(self._go_to_next)
        self.last_btn.clicked.connect(self._go_to_last)
        self.goto_btn.clicked.connect(self._go_to_sample)
        
        self.sample_slider.valueChanged.connect(self._slider_changed)
        self.show_channels_spinbox.valueChanged.connect(self._update_plot)
        self.filter_by_label.currentTextChanged.connect(self._apply_filter)
        
        # Disable navigation initially
        self._set_navigation_enabled(False)    
        
    def _create_training_tab(self):
        """Create training tab with comprehensive training controls"""
        layout = QVBoxLayout(self.training_tab)
        
        # Initialize training manager
        self.training_manager = TrainingManager()
        
        # Top section for configuration
        config_section = QHBoxLayout()
        
        # Model configuration group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["EEGInceptionERP", "EEGNetv4"])
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)
        
        model_group.setLayout(model_layout)
        config_section.addWidget(model_group)
        
        # Training parameters group
        train_group = QGroupBox("Training Parameters")
        train_layout = QVBoxLayout()
        
        self.batch_size_spin = self._create_param_spinner("Batch Size:", 32, 1, 256)
        self.epochs_spin = self._create_param_spinner("Epochs:", 50, 1, 1000)
        self.learning_rate_spin = self._create_param_spinner("Learning Rate:", 0.001, 0.0001, 0.1, 0.0001)
        self.k_folds_spin = self._create_param_spinner("K-Folds:", 5, 1, 10)
        self.patience_spin = self._create_param_spinner("Early Stopping Patience:", 5, 1, 20)
        
        for param in [self.batch_size_spin, self.epochs_spin, self.learning_rate_spin, 
                     self.k_folds_spin, self.patience_spin]:
            train_layout.addLayout(param)
        
        train_group.setLayout(train_layout)
        config_section.addWidget(train_group)
        
        layout.addLayout(config_section)
        
        # Middle section for controls and progress
        control_section = QHBoxLayout()
        
        # Training controls
        control_group = QGroupBox("Training Controls")
        control_layout = QVBoxLayout()
        
        # Progress display
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        control_layout.addWidget(self.training_progress)
        
        # Status label
        self.training_status = QLabel("Ready")
        control_layout.addWidget(self.training_status)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.stop_training_btn = QPushButton("Stop")
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setStyleSheet("background-color: #f44336; color: white;")
        
        button_layout.addWidget(self.start_training_btn)
        button_layout.addWidget(self.stop_training_btn)
        control_layout.addLayout(button_layout)
        
        control_group.setLayout(control_layout)
        control_section.addWidget(control_group)
        
        # Current fold info
        fold_group = QGroupBox("Current Fold")
        fold_layout = QVBoxLayout()
        
        self.current_fold_label = QLabel("Fold: N/A")
        self.current_epoch_label = QLabel("Epoch: N/A")
        self.current_loss_label = QLabel("Loss: N/A")
        self.current_acc_label = QLabel("Accuracy: N/A")
        
        for label in [self.current_fold_label, self.current_epoch_label, 
                     self.current_loss_label, self.current_acc_label]:
            fold_layout.addWidget(label)
        
        fold_group.setLayout(fold_layout)
        control_section.addWidget(fold_group)
        
        layout.addLayout(control_section)
        
        # Bottom section for visualization
        viz_group = QGroupBox("Training Visualization")
        viz_layout = QVBoxLayout()
        
        # Add plot canvas
        self.training_plot = PlotCanvas(width=8, height=4)
        viz_layout.addWidget(self.training_plot)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Connect signals
        self.start_training_btn.clicked.connect(self._start_training)
        self.stop_training_btn.clicked.connect(self._stop_training)
        
        # Connect training manager signals
        self.training_manager.training_started.connect(self._on_training_started)
        self.training_manager.fold_started.connect(self._on_fold_started)
        self.training_manager.epoch_completed.connect(self._on_epoch_completed)
        self.training_manager.fold_completed.connect(self._on_fold_completed)
        self.training_manager.training_completed.connect(self._on_training_completed)
        self.training_manager.training_error.connect(self._on_training_error)

    def _create_param_spinner(self, label, default, minimum, maximum, step=1):
        """Helper to create parameter spinners with labels"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        spinner = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
        spinner.setValue(default)
        spinner.setMinimum(minimum)
        spinner.setMaximum(maximum)
        spinner.setSingleStep(step)
        layout.addWidget(spinner)
        return layout

    def _create_testing_tab(self):
        """Create testing tab with placeholder content"""
        layout = QVBoxLayout(self.testing_tab)
        
        # Main container
        container = QWidget()
        container_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Model Testing")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)
        
        # Placeholder content
        placeholder = QLabel("Testing functionality will be implemented here.\n\n"
                            "This will include:\n"
                            "• Model loading from checkpoints\n"
                            "• Test data selection\n"
                            "• Prediction generation\n"
                            "• Performance metrics calculation\n"
                            "• Confusion matrix and ROC curves\n"
                            "• Real-time classification testing")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("QLabel { padding: 20px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; }")
        container_layout.addWidget(placeholder)
        
        container_layout.addStretch()
        container.setLayout(container_layout)
        layout.addWidget(container)

    def _select_all_subjects(self):
        """Select all subject checkboxes"""
        for checkbox in self.subject_checkboxes.values():
            checkbox.setChecked(True)
    
    def _select_no_subjects(self):
        """Deselect all subject checkboxes"""
        for checkbox in self.subject_checkboxes.values():
            checkbox.setChecked(False)
    
    def _select_first_10_subjects(self):
        """Select only the first 10 subjects"""
        for subject_id, checkbox in self.subject_checkboxes.items():
            checkbox.setChecked(subject_id <= 10)
    
    def _load_data(self):
        """Load data for selected subjects"""
        # Get selected subjects
        selected_subjects = [
            subject_id for subject_id, checkbox in self.subject_checkboxes.items()
            if checkbox.isChecked()
        ]
        
        if not selected_subjects:
            QMessageBox.warning(self, "Warning", "Please select at least one subject!")
            return
        
        # Show progress bar and disable load button
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.load_data_btn.setEnabled(False)
        self.statusBar().showMessage("Loading data...")
        
        # Start loading thread
        self.data_load_thread = DataLoadThread("eeg_data", selected_subjects, [4, 8, 12])
        self.data_load_thread.progress_updated.connect(self.progress_bar.setValue)
        self.data_load_thread.data_loaded.connect(self._on_data_loaded)
        self.data_load_thread.error_occurred.connect(self._on_data_load_error)
        self.data_load_thread.start()
    
    def _on_data_loaded(self, data_dict):
        """Handle successful data loading"""
        self.loaded_data = data_dict
        self.progress_bar.setVisible(False)
        self.load_data_btn.setEnabled(True)
        
        # Apply current filter
        self._apply_filter()
        
        # Update UI
        self._update_data_info()
        self._set_navigation_enabled(True)
        
        # Show first sample
        if self.filtered_indices:
            self.current_sample_idx = 0
            self._update_plot()
        
        self.statusBar().showMessage(f"Loaded {len(self.loaded_data['windows'])} samples from {len(set(self.loaded_data['subject_ids']))} subjects")
    
    def _on_data_load_error(self, error_msg):
        """Handle data loading error"""
        self.progress_bar.setVisible(False)
        self.load_data_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to load data:\n{error_msg}")
        self.statusBar().showMessage("Error loading data")
    
    def _apply_filter(self):
        """Apply label filter to samples"""
        if self.loaded_data is None:
            return
        
        filter_text = self.filter_by_label.currentText()
        all_indices = list(range(len(self.loaded_data['windows'])))
        
        if filter_text == "All Samples":
            self.filtered_indices = all_indices
        elif filter_text == "Left Hand Only":
            self.filtered_indices = [i for i in all_indices if self.loaded_data['labels'][i] == 0]
        elif filter_text == "Right Hand Only":
            self.filtered_indices = [i for i in all_indices if self.loaded_data['labels'][i] == 1]
        
        # Update navigation controls
        if self.filtered_indices:
            self.current_sample_idx = 0
            self.sample_slider.setMaximum(len(self.filtered_indices) - 1)
            self.goto_spinbox.setMaximum(len(self.filtered_indices))
            self._update_plot()
            self._update_data_info()
        else:
            self._set_navigation_enabled(False)
    
    def _update_data_info(self):
        """Update data information labels"""
        if self.loaded_data and self.filtered_indices:
            actual_idx = self.filtered_indices[self.current_sample_idx]
            self.total_samples_label.setText(f"Total Samples: {len(self.filtered_indices)} (filtered)")
            self.current_sample_label.setText(f"Current Sample: {self.current_sample_idx + 1} / {len(self.filtered_indices)}")
            
            label_map = {0: "Left Hand", 1: "Right Hand"} # Example mapping
            label_val = self.loaded_data['labels'][actual_idx]
            self.sample_label_info.setText(f"Label: {label_map.get(label_val, f'Raw: {label_val}')}")
            
            if 'subject_ids' in self.loaded_data and self.loaded_data['subject_ids'] is not None:
                self.sample_subject_info.setText(f"Subject: {self.loaded_data['subject_ids'][actual_idx]}")
            else:
                self.sample_subject_info.setText("Subject: N/A")
            
            # self.sample_run_info.setText(f"Run: {self.loaded_data['run_ids'][actual_idx]}") # Removed, ensure 'run_ids' existed if re-adding
        else:
            self.total_samples_label.setText("Total Samples: 0")
            self.current_sample_label.setText("Current Sample: N/A")
            self.sample_label_info.setText("Label: N/A")
            self.sample_subject_info.setText("Subject: N/A")
            # self.sample_run_info.setText("Run: N/A") # Removed
    
    def _update_plot(self):
        """Update the EEG plot with current sample"""
        if self.loaded_data is None or not self.filtered_indices:
            self.plot_canvas.plot_eeg_data(None, sample_rate=125) # Clear plot if no data
            return
        
        actual_idx = self.filtered_indices[self.current_sample_idx]
        sample_data = self.loaded_data['windows'][actual_idx]
        label = self.loaded_data['labels'][actual_idx]
        subject_id = self.loaded_data['subject_ids'][actual_idx]
        
        # Get number of channels to display
        n_channels_to_show = self.show_channels_spinbox.value()
        channels_to_plot = list(range(min(n_channels_to_show, sample_data.shape[0])))
        
        title = f"Sample {self.current_sample_idx + 1}/{len(self.filtered_indices)} - Subject {subject_id:03d} - {'Left Hand' if label == 0 else 'Right Hand'}"
        
        # Pass the sample_rate from the data_loader
        # Use standard EEG sample rate of 125 Hz
        self.plot_canvas.plot_eeg_data(sample_data, 
                                       sample_rate=125, 
                                       channels=channels_to_plot, 
                                       title=title)
    
    def _set_navigation_enabled(self, enabled):
        """Enable/disable navigation controls"""
        controls = [
            self.first_btn, self.prev_btn, self.next_btn, self.last_btn,
            self.sample_slider, self.goto_spinbox, self.goto_btn
        ]
        for control in controls:
            control.setEnabled(enabled)
    
    def _go_to_first(self):
        """Navigate to first sample"""
        if self.filtered_indices:
            self.current_sample_idx = 0
            self.sample_slider.setValue(0)
            self._update_plot()
    
    def _go_to_previous(self):
        """Navigate to previous sample"""
        if self.filtered_indices and self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.sample_slider.setValue(self.current_sample_idx)
            self._update_plot()
    
    def _go_to_next(self):
        """Navigate to next sample"""
        if self.filtered_indices and self.current_sample_idx < len(self.filtered_indices) - 1:
            self.current_sample_idx += 1
            self.sample_slider.setValue(self.current_sample_idx)
            self._update_plot()
    
    def _go_to_last(self):
        """Navigate to last sample"""
        if self.filtered_indices:
            self.current_sample_idx = len(self.filtered_indices) - 1
            self.sample_slider.setValue(self.current_sample_idx)
            self._update_plot()
    
    def _go_to_sample(self):
        """Navigate to specific sample number"""
        if self.filtered_indices:
            target_sample = self.goto_spinbox.value() - 1  # Convert to 0-based index
            if 0 <= target_sample < len(self.filtered_indices):
                self.current_sample_idx = target_sample
                self.sample_slider.setValue(self.current_sample_idx)
                self._update_plot()
    
    def _slider_changed(self, value):
        """Handle slider value change"""
        if self.filtered_indices and 0 <= value < len(self.filtered_indices):
            self.current_sample_idx = value
            self._update_plot()
    
    def _on_training_started(self):
        self.statusBar().showMessage("Training started...")
        
    def _on_fold_started(self, fold):
        self.statusBar().showMessage(f"Training fold {fold}...")
        
    def _on_epoch_completed(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.statusBar().showMessage(f"Epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        
    def _on_fold_completed(self, fold, val_acc):
        self.statusBar().showMessage(f"Fold {fold} completed: val_acc={val_acc:.3f}")
        
    def _on_training_completed(self, results):
        self.statusBar().showMessage("Training completed!")
        
    def _on_training_error(self, error_msg):
        QMessageBox.critical(self, "Training Error", error_msg)

    def _start_training(self):
        """Start the training process using parameters from the UI"""
        if not self.loaded_data:
            QMessageBox.warning(self, "Data Missing", "Please load data before starting training.")
            return

        model_type = self.model_type_combo.currentText()
        batch_size = self.batch_size_spin.findChild(QSpinBox).value()
        num_epochs = self.epochs_spin.findChild(QSpinBox).value()
        learning_rate = self.learning_rate_spin.findChild(QDoubleSpinBox).value()
        k_folds = self.k_folds_spin.findChild(QSpinBox).value()
        patience = self.patience_spin.findChild(QSpinBox).value()

        # Prepare model configuration
        model_config = {
            'model_type': model_type,
            # 'n_filters': n_filters, # Removed
            # 'dropout': dropout, # Removed
            # Assuming EEGInceptionERP specific defaults or that ModelTrainer handles missing keys
            'n_channels': self.loaded_data['windows'].shape[1] if self.loaded_data else 16, # Default if no data
            'n_classes': len(set(self.loaded_data['labels'])) if self.loaded_data else 2, # Default if no data
            'input_window_samples': self.loaded_data['windows'].shape[2] if self.loaded_data else 500 # Default
        }

        # Prepare training configuration
        training_config = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'k_folds': k_folds,
            'patience': patience,
            'model_save_path': project_root # Base path, ModelTrainer will append specifics
        }
        
        self.training_manager.configure(model_config, training_config)
        
        # Ensure data is prepared in the TrainingManager
        # This uses the data loaded in the Data Loading Tab
        self.training_manager.prepare_data(
            self.loaded_data['windows'], 
            self.loaded_data['labels'], 
            self.loaded_data.get('subject_ids') # Pass subject_ids if available
        )

        self.training_manager.start_training()
        
        # Update UI
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_progress.setVisible(True)

    def _stop_training(self):
        """Stop the training process"""
        self.training_manager.stop_training()
        self.stop_training_btn.setEnabled(False)
        self.start_training_btn.setEnabled(True)
        self.training_status.setText("Training stopped")

def main():
    app = QApplication([]) # Use sys.argv in a standalone script
    # app.setStyle("Fusion") # Optional: set a style

    # Palette for dark mode (optional, can be customized)
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # # ... (set other colors for dark mode)
    # app.setPalette(dark_palette)

    window = BCIMainWindow()
    window.show()
    # sys.exit(app.exec_()) # Use this if running bci_gui_app.py directly

if __name__ == "__main__":
    # This part is usually for direct execution.
    # If test_gui.py is the entry point, it will handle app.exec_()
    import sys
    app = QApplication(sys.argv)
    window = BCIMainWindow()
    window.show()
    sys.exit(app.exec_())
