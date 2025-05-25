"""
Braindecode-based EEG Training System
Main application for training and evaluating EEG models using Braindecode library.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Braindecode imports
from braindecode import EEGRegressor, EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor
)
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds

# Sklearn imports
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# MNE imports
import mne
from mne.io import RawArray
from mne.datasets import eegbci
from mne import events_from_annotations

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QComboBox, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QTabWidget, QFileDialog, QMessageBox,
    QScrollArea, QSplitter, QFrame, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CopyableErrorDialog(QDialog):
    """
    Custom error dialog that allows copying error messages to clipboard.
    """
    
    def __init__(self, parent=None, title="Error", message="", icon_type="error"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(500, 300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Icon and title layout
        header_layout = QHBoxLayout()
        
        # Icon label
        icon_label = QLabel()
        if icon_type == "error":
            icon_label.setText("❌")
        elif icon_type == "warning":
            icon_label.setText("⚠️")
        elif icon_type == "info":
            icon_label.setText("ℹ️")
        else:
            icon_label.setText("💡")
        
        icon_label.setStyleSheet("font-size: 24px; margin-right: 10px;")
        header_layout.addWidget(icon_label)
        
        # Title label
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Message text area (copyable)
        self.message_text = QTextEdit()
        self.message_text.setPlainText(message)
        self.message_text.setReadOnly(True)
        self.message_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.message_text)
        
        # Copy button and OK button
        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(copy_button)
        
        button_layout.addStretch()
        
        # Standard buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_layout.addWidget(button_box)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def copy_to_clipboard(self):
        """Copy the error message to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.message_text.toPlainText())
        
        # Temporarily change button text to show feedback
        copy_button = self.sender()
        original_text = copy_button.text()
        copy_button.setText("Copied! ✓")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        # Reset button after 2 seconds
        QTimer.singleShot(2000, lambda: self.reset_copy_button(copy_button, original_text))
        
    def reset_copy_button(self, button, original_text):
        """Reset copy button to original state."""
        button.setText(original_text)
        button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)


def show_copyable_error(parent=None, title="Error", message="", icon_type="error"):
    """
    Show a copyable error dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message
        icon_type: Type of icon ('error', 'warning', 'info', 'question')
    """
    dialog = CopyableErrorDialog(parent, title, message, icon_type)
    return dialog.exec_()


def show_copyable_warning(parent=None, title="Warning", message=""):
    """Show a copyable warning dialog."""
    return show_copyable_error(parent, title, message, "warning")


def show_copyable_info(parent=None, title="Information", message=""):
    """Show a copyable information dialog."""
    return show_copyable_error(parent, title, message, "info")


class EEGDataManager:
    """
    Manages EEG data loading and preprocessing using local CSV files and Braindecode.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize EEG Data Manager.
        
        Args:
            data_path: Path to the CSV data directory
        """
        self.data_path = data_path
        self.subjects_data: Dict[int, Dict[str, Any]] = {}
        self.channel_names = [
            'C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8',
            'T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2'
        ]
        self.sfreq = 125  # OpenBCI sampling rate
        
    def load_subject_data(self, subject_id: int) -> Optional[Dict[str, Any]]:
        """
        Load and preprocess data for a specific subject.
        
        Args:
            subject_id: Subject ID to load
            
        Returns:
            Dictionary containing processed epochs, labels, and metadata
        """
        assert 1 <= subject_id <= 109, f"Subject ID must be between 1-109, got {subject_id}"
        
        subject_dir = os.path.join(
            self.data_path, 'MNE-eegbci-data', 'files', 
            'eegmmidb', '1.0.0', f'S{subject_id:03d}'
        )
        
        if not os.path.exists(subject_dir):
            return None
            
        epochs_list = []
        labels_list = []
        runs = [4, 8, 12]  # Motor imagery runs
        
        for run in runs:
            csv_file = os.path.join(subject_dir, f'S{subject_id:03d}R{run:02d}_csv_openbci.csv')
            
            if not os.path.exists(csv_file):
                continue
                
            try:
                # Load CSV data
                df = pd.read_csv(csv_file, comment='%', engine='python', on_bad_lines='skip')
                
                # Extract EEG channels
                eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
                if len(eeg_cols) < 16:
                    continue
                    
                eeg_data = df[eeg_cols[:16]].values.T  # Shape: (channels, samples)
                annotations = df['Annotations'].fillna('')
                
                # Find events (T1=left, T2=right)
                event_indices = []
                event_types = []
                
                for idx, ann in enumerate(annotations):
                    if ann in ['T1', 'T2']:
                        event_indices.append(idx)
                        event_types.append(0 if ann == 'T1' else 1)  # 0=left, 1=right
                
                # Extract epochs (3.1 seconds starting 1 second after event)
                samples_per_epoch = int(3.1 * self.sfreq)
                start_offset = int(1.0 * self.sfreq)
                
                for evt_idx, evt_type in zip(event_indices, event_types):
                    start = evt_idx + start_offset
                    end = start + samples_per_epoch
                    
                    if end <= eeg_data.shape[1]:
                        epoch = eeg_data[:, start:end]
                        epochs_list.append(epoch)
                        labels_list.append(evt_type)
                        
            except Exception as e:
                print(f"Error loading run {run} for subject {subject_id}: {e}")
                continue
        
        if not epochs_list:
            return None
            
        epochs = np.stack(epochs_list)  # Shape: (trials, channels, samples)
        labels = np.array(labels_list)
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sfreq,
            ch_types='eeg'
        )
        
        # Create epochs object for braindecode compatibility
        mne_epochs = []
        for i, epoch in enumerate(epochs):
            raw = RawArray(epoch, info)
            mne_epochs.append(raw)
            
        subject_data = {
            'epochs': epochs,
            'labels': labels,
            'info': info,
            'mne_epochs': mne_epochs,
            'n_trials': len(epochs),
            'n_channels': epochs.shape[1],
            'n_samples': epochs.shape[2]
        }
        
        self.subjects_data[subject_id] = subject_data
        return subject_data
    
    def get_multiple_subjects_data(self, subject_ids: List[int]) -> Dict[str, Any]:
        """
        Load and combine data from multiple subjects.
        
        Args:
            subject_ids: List of subject IDs to load
            
        Returns:
            Combined dataset dictionary
        """
        all_epochs = []
        all_labels = []
        successful_subjects = []
        
        for subject_id in subject_ids:
            print(f"Loading subject {subject_id}...")
            data = self.load_subject_data(subject_id)
            
            if data is not None:
                all_epochs.append(data['epochs'])
                all_labels.append(data['labels'])
                successful_subjects.append(subject_id)
            else:
                print(f"Failed to load subject {subject_id}")
        
        if not all_epochs:
            raise ValueError("No valid subject data found")
            
        # Combine all data
        combined_epochs = np.concatenate(all_epochs, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        return {
            'epochs': combined_epochs,
            'labels': combined_labels,
            'successful_subjects': successful_subjects,
            'n_subjects': len(successful_subjects),
            'n_trials': len(combined_epochs),
            'n_channels': combined_epochs.shape[1],
            'n_samples': combined_epochs.shape[2]
        }


class BrainDecodeTrainer:
    """
    Handles model training and evaluation using Braindecode framework.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the trainer.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.clf = None
        self.history = {}
        self.metrics = {}
        
    def create_model(self, model_type: str, n_channels: int, n_samples: int, n_classes: int = 2) -> nn.Module:
        """
        Create a braindecode model.
        
        Args:
            model_type: Type of model ('ShallowFBCSPNet', 'EEGNetv4')
            n_channels: Number of EEG channels
            n_samples: Number of time samples
            n_classes: Number of classes
            
        Returns:
            Initialized model
        """
        assert model_type in ['ShallowFBCSPNet', 'EEGNetv4'], f"Unsupported model type: {model_type}"
        assert n_classes > 1, f"Number of classes must be > 1, got {n_classes}"
        
        if model_type == 'ShallowFBCSPNet':
            model = ShallowFBCSPNet(
                n_chans=n_channels,
                n_outputs=n_classes,
                input_window_samples=n_samples,
                n_filters_time=40,
                n_filters_spat=40,
                filter_time_length=25,
                pool_time_length=75,
                pool_time_stride=15,
                final_conv_length='auto',
                drop_prob=0.5
            )
        elif model_type == 'EEGNetv4':
            model = EEGNetv4(
                n_chans=n_channels,
                n_outputs=n_classes,
                input_window_samples=n_samples,
                F1=8,
                D=2,
                F2=16,
                kernel_length=64,
                drop_prob=0.25
            )
            
        self.model = model
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_type: str = 'EEGNetv4',
                   n_epochs: int = 100, 
                   batch_size: int = 64,
                   learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """
        Train the model using braindecode.
        
        Args:
            X_train: Training data (trials, channels, samples)
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            model_type: Type of model to use
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        assert len(X_train) > 0, "Training data cannot be empty"
        assert len(X_train) == len(y_train), "Training data and labels must have same length"
        assert X_train.shape[1:] == X_val.shape[1:], "Train and validation data must have same shape"
        
        n_channels, n_samples = X_train.shape[1], X_train.shape[2]
        n_classes = len(np.unique(y_train))
        
        # Set random seeds for reproducibility
        set_random_seeds(seed=42, cuda=torch.cuda.is_available())
        
        # Create model
        model = self.create_model(model_type, n_channels, n_samples, n_classes)        # Create EEGClassifier
        self.clf = EEGClassifier(
            model,
            criterion=CroppedLoss(F.nll_loss),
            optimizer=torch.optim.Adam,
            optimizer__lr=learning_rate,
            batch_size=batch_size,
            max_epochs=n_epochs,
            device=self.device,
            callbacks=None
        )
        
        # Convert data to torch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).long()
          # Create datasets
        from braindecode.datasets import create_from_X_y
        
        train_set = create_from_X_y(
            X_train_tensor, y_train_tensor, drop_last_window=False, sfreq=125
        )
        
        valid_set = create_from_X_y(
            X_val_tensor, y_val_tensor, drop_last_window=False, sfreq=125
        )
        
        # Train the model
        print(f"Training {model_type} model...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Device: {self.device}")
        
        self.clf.fit(train_set, y=None, valid_set=valid_set)
        
        # Extract training history
        self.history = {
            'train_loss': self.clf.history[:, 'train_loss'],
            'valid_loss': self.clf.history[:, 'valid_loss'],
            'train_accuracy': self.clf.history[:, 'train_accuracy'],
            'valid_accuracy': self.clf.history[:, 'valid_accuracy']
        }
        
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        assert self.clf is not None, "Model must be trained before evaluation"
        assert len(X_test) > 0, "Test data cannot be empty"
        assert len(X_test) == len(y_test), "Test data and labels must have same length"
        
        # Convert to tensors
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        # Create test dataset
        from braindecode.datasets import create_from_X_y
        test_set = create_from_X_y(
            X_test_tensor, y_test_tensor, drop_last_window=False
        )
        
        # Make predictions
        y_pred = self.clf.predict(test_set)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        self.metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        return self.metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        assert self.clf is not None, "No model to save"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save using pickle (braindecode standard)
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.clf, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        assert os.path.exists(filepath), f"Model file not found: {filepath}"
        
        import pickle
        with open(filepath, 'rb') as f:
            self.clf = pickle.load(f)
        
        print(f"Model loaded from: {filepath}")


class TrainingThread(QThread):
    """
    Separate thread for model training to avoid GUI freezing.
    """
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data_manager: EEGDataManager, trainer: BrainDecodeTrainer,
                 subject_ids: List[int], model_type: str, n_epochs: int,
                 batch_size: int, learning_rate: float):
        super().__init__()
        self.data_manager = data_manager
        self.trainer = trainer
        self.subject_ids = subject_ids
        self.model_type = model_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def run(self):
        """Run the training process."""
        try:
            # Load data
            self.status_updated.emit("Loading subject data...")
            self.progress_updated.emit(10)
            
            data = self.data_manager.get_multiple_subjects_data(self.subject_ids)
            
            self.status_updated.emit("Splitting data...")
            self.progress_updated.emit(30)
            
            # Split data
            X, y = data['epochs'], data['labels']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            self.status_updated.emit("Training model...")
            self.progress_updated.emit(50)
            
            # Train model
            history = self.trainer.train_model(
                X_train, y_train, X_val, y_val,
                model_type=self.model_type,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            
            self.status_updated.emit("Evaluating model...")
            self.progress_updated.emit(80)
            
            # Evaluate model
            metrics = self.trainer.evaluate_model(X_test, y_test)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Training completed!")
            
            # Emit results
            results = {
                'history': history,
                'metrics': metrics,
                'data_info': data
            }
            
            self.training_completed.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class MetricsVisualizationWidget(QWidget):
    """
    Widget for displaying training metrics and evaluation results.
    """
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_plots(self, history: Dict[str, List[float]], metrics: Dict[str, Any]):
        """
        Update the plots with training results.
        
        Args:
            history: Training history
            metrics: Evaluation metrics
        """
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Training loss
        ax1 = self.figure.add_subplot(gs[0, 0])
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training accuracy
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['valid_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confusion matrix
        ax3 = self.figure.add_subplot(gs[1, 0])
        conf_matrix = metrics['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Classification metrics
        ax4 = self.figure.add_subplot(gs[1, 1])
        report = metrics['classification_report']
        
        # Extract metrics for visualization
        classes = ['Left', 'Right']
        precision = [report['0']['precision'], report['1']['precision']]
        recall = [report['0']['recall'], report['1']['recall']]
        f1_score = [report['0']['f1-score'], report['1']['f1-score']]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax4.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax4.bar(x, recall, width, label='Recall', alpha=0.8)
        ax4.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax4.set_title('Classification Metrics by Class')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(classes)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.canvas.draw()


class MainApplicationWindow(QMainWindow):
    """
    Main application window for the Braindecode EEG Training System.
    """
    
    def __init__(self):
        super().__init__()
        self.data_manager = None
        self.trainer = None
        self.training_thread = None
        self.current_results = None
        
        self.init_ui()
        self.init_data_manager()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Braindecode EEG Training System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create left panel for controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Create right panel for visualization
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Data Configuration Group
        data_group = QGroupBox("Data Configuration")
        data_layout = QGridLayout()
        
        # Subject selection
        data_layout.addWidget(QLabel("Number of Subjects:"), 0, 0)
        self.num_subjects_spinbox = QSpinBox()
        self.num_subjects_spinbox.setRange(1, 50)
        self.num_subjects_spinbox.setValue(5)
        data_layout.addWidget(self.num_subjects_spinbox, 0, 1)
        
        # Subject ID selection (range)
        data_layout.addWidget(QLabel("Subject ID Range:"), 1, 0)
        self.subject_range_layout = QHBoxLayout()
        self.start_subject_spinbox = QSpinBox()
        self.start_subject_spinbox.setRange(1, 109)
        self.start_subject_spinbox.setValue(1)
        self.subject_range_layout.addWidget(QLabel("From:"))
        self.subject_range_layout.addWidget(self.start_subject_spinbox)
        
        self.end_subject_spinbox = QSpinBox()
        self.end_subject_spinbox.setRange(1, 109)
        self.end_subject_spinbox.setValue(10)
        self.subject_range_layout.addWidget(QLabel("To:"))
        self.subject_range_layout.addWidget(self.end_subject_spinbox)
        
        range_widget = QWidget()
        range_widget.setLayout(self.subject_range_layout)
        data_layout.addWidget(range_widget, 1, 1)
        
        # Data path
        data_layout.addWidget(QLabel("Data Path:"), 2, 0)
        self.data_path_button = QPushButton("Select Data Folder")
        self.data_path_button.clicked.connect(self.select_data_path)
        data_layout.addWidget(self.data_path_button, 2, 1)
        
        self.data_path_label = QLabel("No path selected")
        data_layout.addWidget(self.data_path_label, 3, 0, 1, 2)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Model Configuration Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        # Model type
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EEGNetv4", "ShallowFBCSPNet"])
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # Number of epochs
        model_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 500)
        self.epochs_spinbox.setValue(100)
        model_layout.addWidget(self.epochs_spinbox, 1, 1)
        
        # Batch size
        model_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(32)
        model_layout.addWidget(self.batch_size_spinbox, 2, 1)
        
        # Learning rate
        model_layout.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.001", "0.0001", "0.01", "0.005"])
        model_layout.addWidget(self.lr_combo, 3, 1)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training Control Group
        training_group = QGroupBox("Training Control")
        training_layout = QVBoxLayout()
        
        # Start training button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        training_layout.addWidget(self.train_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        training_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to start training")
        training_layout.addWidget(self.status_label)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Model Management Group
        management_group = QGroupBox("Model Management")
        management_layout = QVBoxLayout()
        
        # Save model button
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        management_layout.addWidget(self.save_button)
        
        # Load model button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        management_layout.addWidget(self.load_button)
        
        management_group.setLayout(management_layout)
        layout.addWidget(management_group)
        
        # Results Group
        results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout()
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def create_visualization_panel(self) -> QWidget:
        """Create the visualization panel."""
        # Create tab widget for different visualizations
        tab_widget = QTabWidget()
        
        # Training metrics tab
        self.metrics_widget = MetricsVisualizationWidget()
        tab_widget.addTab(self.metrics_widget, "Training Metrics")
        
        # Additional tabs can be added here for other visualizations
        
        return tab_widget
        
    def init_data_manager(self):
        """Initialize the data manager with default path."""
        default_path = r"C:\Users\Chari\OneDrive\Documentos\GitHub\projetoBCI\eeg_data"
        if os.path.exists(default_path):
            self.data_manager = EEGDataManager(default_path)
            self.data_path_label.setText(default_path)
            self.train_button.setEnabled(True)
    
    def select_data_path(self):
        """Select the data path."""
        path = QFileDialog.getExistingDirectory(self, "Select EEG Data Folder")
        if path:
            self.data_manager = EEGDataManager(path)
            self.data_path_label.setText(path)
            self.train_button.setEnabled(True)
            
    def start_training(self):
        """Start the training process."""
        if self.data_manager is None:
            show_copyable_warning(self, "Warning", "Please select a data path first.")
            return
            
        # Get parameters
        num_subjects = self.num_subjects_spinbox.value()
        model_type = self.model_combo.currentText()
        n_epochs = self.epochs_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        learning_rate = float(self.lr_combo.currentText())
        
        # Generate subject IDs (randomly select from available range)
        import random
        subject_ids = random.sample(range(1, 110), min(num_subjects, 109))
        
        # Create trainer
        self.trainer = BrainDecodeTrainer()
        
        # Create and start training thread
        self.training_thread = TrainingThread(
            self.data_manager, self.trainer, subject_ids,
            model_type, n_epochs, batch_size, learning_rate
        )
        
        # Connect signals
        self.training_thread.progress_updated.connect(self.progress_bar.setValue)
        self.training_thread.status_updated.connect(self.status_label.setText)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.error_occurred.connect(self.on_training_error)
        
        # Update UI
        self.train_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Start training
        self.training_thread.start()
        
    def on_training_completed(self, results: Dict[str, Any]):
        """Handle training completion."""
        self.current_results = results
        
        # Update UI
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Update metrics visualization
        self.metrics_widget.update_plots(
            results['history'], 
            results['metrics']
        )
        
        # Update results text
        metrics = results['metrics']
        data_info = results['data_info']
        
        results_text = f"""
Training Completed Successfully!

Dataset Information:
- Subjects used: {data_info['successful_subjects']}
- Total trials: {data_info['n_trials']}
- Channels: {data_info['n_channels']}
- Samples per trial: {data_info['n_samples']}

Model Performance:
- Test Accuracy: {metrics['accuracy']:.3f}

Classification Report:
- Left Hand (Class 0):
  * Precision: {metrics['classification_report']['0']['precision']:.3f}
  * Recall: {metrics['classification_report']['0']['recall']:.3f}
  * F1-Score: {metrics['classification_report']['0']['f1-score']:.3f}

- Right Hand (Class 1):
  * Precision: {metrics['classification_report']['1']['precision']:.3f}
  * Recall: {metrics['classification_report']['1']['recall']:.3f}
  * F1-Score: {metrics['classification_report']['1']['f1-score']:.3f}

Overall Metrics:
- Macro Average F1: {metrics['classification_report']['macro avg']['f1-score']:.3f}
- Weighted Average F1: {metrics['classification_report']['weighted avg']['f1-score']:.3f}
        """
        
        self.results_text.setText(results_text)
        
    def on_training_error(self, error_message: str):
        """Handle training error."""
        self.train_button.setEnabled(True)
        self.status_label.setText("Training failed!")
        QMessageBox.critical(self, "Training Error", f"Training failed with error:\n{error_message}")
        
    def save_model(self):
        """Save the trained model."""
        if self.trainer is None or self.trainer.clf is None:
            QMessageBox.warning(self, "Warning", "No trained model to save.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model", 
            "braindecode_model.pkl", 
            "Pickle Files (*.pkl)"
        )
        
        if filepath:
            try:
                self.trainer.save_model(filepath)
                QMessageBox.information(self, "Success", f"Model saved successfully to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
                
    def load_model(self):
        """Load a trained model."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", 
            "", 
            "Pickle Files (*.pkl)"
        )
        
        if filepath:
            try:
                if self.trainer is None:
                    self.trainer = BrainDecodeTrainer()
                self.trainer.load_model(filepath)
                self.save_button.setEnabled(True)
                QMessageBox.information(self, "Success", f"Model loaded successfully from:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Braindecode EEG Training System")
    app.setOrganizationName("BCI Research Lab")
    
    # Create and show main window
    window = MainApplicationWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
