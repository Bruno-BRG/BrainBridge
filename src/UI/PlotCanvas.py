import numpy as np
from typing import Optional
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

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
        
    def plot_eeg_data(self, data: np.ndarray, sample_rate: float, channels: Optional[list] = None, title: str = "EEG Data", baseline_duration_sec: float = 1.0):
        """Plot EEG data with sample index on x-axis and stimulus onset line."""
        self.axes.clear()
        
        if data is None:
            self.axes.set_title("No data to display")
            self.axes.grid(True)
            self.draw()
            return
            
        if len(data.shape) == 3:  # (trials, channels, time_points)
            # Plot first trial if multiple are passed, though typically one window is passed
            data = data[0]
        
        n_channels, n_times = data.shape
        sample_indices = np.arange(n_times)
        
        # Select channels to plot (max 10 for clarity or as specified)
        if channels is None:
            channels_to_plot = list(range(min(10, n_channels)))
        else:
            channels_to_plot = [ch for ch in channels if ch < n_channels]

        if not channels_to_plot: # If channels list is empty or all out of bounds
            self.axes.set_title("No valid channels to display")
            self.axes.grid(True)
            self.draw()
            return

        # Define colors for the selected channels
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels_to_plot)))
        
        # Normalize data to enhance amplitude visibility
        for i, ch_idx in enumerate(channels_to_plot):
            channel_data = data[ch_idx]
            channel_std = np.std(channel_data)
            if channel_std > 0:
                normalized_data = (channel_data - np.mean(channel_data)) / channel_std
            else:
                normalized_data = channel_data # Avoid division by zero if std is 0
            
            offset = i * 3  # Small offset for slight separation
            
            self.axes.plot(sample_indices, normalized_data + offset, color=colors[i], 
                           label=f'Channel {ch_idx}', linewidth=1.5, alpha=0.8) # Adjusted linewidth
        
        # Add vertical line for stimulus onset
        stimulus_onset_sample = int(baseline_duration_sec * sample_rate)
        if 0 < stimulus_onset_sample < n_times:
            self.axes.axvline(stimulus_onset_sample, color='r', linestyle='--', linewidth=1.5, label='Stimulus Onset')
        
        self.axes.set_xlabel('Sample Index')
        self.axes.set_ylabel('Normalized Amplitude (Offset)')
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
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

