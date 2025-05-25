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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
    QGroupBox, QProgressBar, QComboBox, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QSplitter, QMessageBox,
    QGridLayout, QFrame, QSlider, QDoubleSpinBox, QListWidget,
    QListWidgetItem, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor

# Import our custom modules
from src.model.eeg_inception_erp import EEGModel
from src.data.data_loader import BCIDataLoader, BCIDataset

# Set the style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingThread(QThread):
    """Thread for training models in background"""
    progress_updated = pyqtSignal(int)
    epoch_completed = pyqtSignal(int, float, float, float, float)  # epoch, train_loss, train_acc, val_loss, val_acc
    training_completed = pyqtSignal(object, str)  # model, save_path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, X_train, y_train, X_val, y_val, model_config, training_config, save_path):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_config = model_config
        self.training_config = training_config
        self.save_path = save_path
        
    def run(self):
        try:
            # Initialize model
            model = EEGModel(**self.model_config)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Prepare data
            train_dataset = BCIDataset(self.X_train, self.y_train)
            val_dataset = BCIDataset(self.X_val, self.y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
            
            # Initialize training components
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.training_config['learning_rate'])
            
            num_epochs = self.training_config['num_epochs']
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                # Emit epoch results
                self.epoch_completed.emit(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.training_config['patience']:
                        break
                
                # Update progress
                progress = int((epoch + 1) / num_epochs * 100)
                self.progress_updated.emit(progress)
            
            self.training_completed.emit(model, self.save_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Training error: {str(e)}")


class ModelLoadThread(QThread):
    """Thread for loading models in background"""
    progress_updated = pyqtSignal(int)
    model_loaded = pyqtSignal(object, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, model_config):
        super().__init__()
        self.model_path = model_path
        self.model_config = model_config
    
    def run(self):
        try:
            self.progress_updated.emit(25)
            
            # Create model instance
            model = EEGModel(**self.model_config)
            self.progress_updated.emit(50)
            
            # Load state dict
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location='cpu')
            
            self.progress_updated.emit(75)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            self.progress_updated.emit(100)
            self.model_loaded.emit(model, self.model_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading model: {str(e)}")


class DataLoadThread(QThread):
    """Thread for loading and preprocessing data in background"""
    progress_updated = pyqtSignal(int)
    data_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data_path, subjects, runs):
        super().__init__()
        self.data_path = data_path
        self.subjects = subjects
        self.runs = runs
    
    def run(self):
        try:
            self.progress_updated.emit(20)
            
            # Initialize data loader
            data_loader = BCIDataLoader(
                data_path=self.data_path,
                subjects=self.subjects,
                runs=self.runs
            )
            
            self.progress_updated.emit(60)
            
            # Load and preprocess data
            windows, labels, subject_ids = data_loader.load_all_subjects()
            
            # Create data dictionary
            data_dict = {
                'windows': windows,
                'labels': labels,
                'subject_ids': subject_ids
            }
            
            self.progress_updated.emit(100)
            self.data_loaded.emit(data_dict)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading data: {str(e)}")


class PlotCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initial empty plot
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title("No data to display")
        self.axes.grid(True)
        
    def plot_eeg_data(self, data, sfreq=160.0, channels=None, title="EEG Data"):
        """Plot EEG data with overlapping channels for better amplitude visibility"""
        self.axes.clear()
        
        if data is None:
            self.axes.set_title("No data to display")
            self.axes.grid(True)
            self.draw()
            return
            
        if len(data.shape) == 3:  # (trials, channels, time_points)
            # Plot first trial
            data = data[0]
        
        n_channels, n_times = data.shape
        time = np.arange(n_times) / sfreq
        
        # Select channels to plot (max 10 for clarity)
        if channels is None:
            channels = list(range(min(10, n_channels)))
        
        # Define colors for all channels
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))
        
        # Normalize data to enhance amplitude visibility
        for i, ch in enumerate(channels):
            if ch < n_channels:
                # Normalize each channel to standard deviation for better amplitude visibility
                channel_data = data[ch]
                # Standardize the data to make amplitude variations more visible
                channel_std = np.std(channel_data)
                if channel_std > 0:
                    normalized_data = (channel_data - np.mean(channel_data)) / channel_std
                else:
                    normalized_data = channel_data
                
                # Small offset for slight separation while maintaining overlap
                offset = i * 3  # Much smaller offset for overlapping effect
                
                self.axes.plot(time, normalized_data + offset, color=colors[i], 
                             label=f'Channel {ch}', linewidth=2.0, alpha=0.8)
        
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Normalized Amplitude')
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_training_progress(self, epochs, train_loss, train_acc, val_loss, val_acc):
        """Plot training progress"""
        self.axes.clear()
        
        if not epochs:
            self.axes.set_title("No training data")
            self.axes.grid(True)
            self.draw()
            return
        
        # Create subplots
        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        
        # Plot loss
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_prediction_results(self, data, predictions, true_labels, confidences):
        """Plot prediction results"""
        self.axes.clear()
        
        if len(predictions) == 0:
            self.axes.set_title("No predictions to display")
            self.axes.grid(True)
            self.draw()
            return
        
        # Plot accuracy over time
        correct = np.array(predictions) == np.array(true_labels)
        accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        
        windows = np.arange(1, len(predictions) + 1)
        
        self.axes.plot(windows, accuracy, 'g-', linewidth=2, label='Cumulative Accuracy')
        self.axes.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Chance Level')
        
        self.axes.set_xlabel('Window Number')
        self.axes.set_ylabel('Accuracy')
        self.axes.set_title(f'Prediction Accuracy (Overall: {accuracy[-1]:.3f})')
        self.axes.legend()
        self.axes.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()


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
        
        # Initialize data loader to get available subjects
        self.data_loader = BCIDataLoader(data_path="eeg_data")
        available_subjects = self.data_loader.get_available_subjects()
        
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
        self.sample_run_info = QLabel("Run: N/A")
        
        info_layout.addWidget(self.total_samples_label)
        info_layout.addWidget(self.current_sample_label)
        info_layout.addWidget(self.sample_label_info)
        info_layout.addWidget(self.sample_subject_info)
        info_layout.addWidget(self.sample_run_info)
        
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
        self.show_channels_spinbox.setValue(10)
        
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
        """Create training tab with placeholder content"""
        layout = QVBoxLayout(self.training_tab)
        
        # Main container
        container = QWidget()
        container_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Configuration")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)
        
        # Placeholder content
        placeholder = QLabel("Training functionality will be implemented here.\n\n"
                            "This will include:\n"
                            "• Model architecture selection\n"
                            "• Training parameter configuration\n"
                            "• K-fold cross-validation setup\n"
                            "• Training progress visualization\n"
                            "• Model saving and checkpointing")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("QLabel { padding: 20px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; }")
        container_layout.addWidget(placeholder)
        
        container_layout.addStretch()
        container.setLayout(container_layout)
        layout.addWidget(container)

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
        if self.loaded_data is None:
            return
        
        total_samples = len(self.filtered_indices)
        self.total_samples_label.setText(f"Total Samples: {total_samples}")
        
        if total_samples > 0:
            actual_idx = self.filtered_indices[self.current_sample_idx]
            label = self.loaded_data['labels'][actual_idx]
            subject_id = self.loaded_data['subject_ids'][actual_idx]
            
            self.current_sample_label.setText(f"Current Sample: {self.current_sample_idx + 1}/{total_samples}")
            self.sample_label_info.setText(f"Label: {'Left Hand' if label == 0 else 'Right Hand'}")
            self.sample_subject_info.setText(f"Subject: {subject_id:03d}")
            # Note: run info would need to be added to the data structure
            self.sample_run_info.setText("Run: N/A")
    
    def _update_plot(self):
        """Update the EEG plot with current sample"""
        if self.loaded_data is None or not self.filtered_indices:
            return
        
        actual_idx = self.filtered_indices[self.current_sample_idx]
        sample_data = self.loaded_data['windows'][actual_idx]
        label = self.loaded_data['labels'][actual_idx]
        subject_id = self.loaded_data['subject_ids'][actual_idx]
        
        # Get number of channels to display
        n_channels_to_show = self.show_channels_spinbox.value()
        channels_to_plot = list(range(min(n_channels_to_show, sample_data.shape[0])))
        
        title = f"Sample {self.current_sample_idx + 1}/{len(self.filtered_indices)} - Subject {subject_id:03d} - {'Left Hand' if label == 0 else 'Right Hand'}"
        
        self.plot_canvas.plot_eeg_data(sample_data, channels=channels_to_plot, title=title)
        self._update_data_info()
    
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

    # ...existing code...
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
