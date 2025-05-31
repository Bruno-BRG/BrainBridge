"""
Classes: PlotCanvas, TrainingPlotCanvas
Purpose: Provide Matplotlib canvas widgets for embedding plots within the PyQt5 GUI.
         PlotCanvas is for general data plotting, TrainingPlotCanvas is specialized for EEG samples.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QMessageBox # QWidget is a common base, though not strictly necessary for FigureCanvas if only used as a canvas

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        if parent: # Ensure parent is set if provided
            self.setParent(parent)

    def plot(self, data, title=""):
        self.axes.clear()
        if data.ndim == 1:  # Single channel
            self.axes.plot(data)
        elif data.ndim == 2:  # Multiple channels (channels, timepoints)
            offset_scale = np.max(np.abs(data)) * 1.5 if np.max(np.abs(data)) > 0 else 1.0
            for i in range(data.shape[0]):
                self.axes.plot(data[i, :] + i * offset_scale)
        self.axes.set_title(title)
        self.axes.grid(True)
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_title("No Data")
        self.draw()

    def plot_and_save(self, data_to_plot: np.ndarray, filename: str, title: str):
        """Plots the given data with a title and saves it to a file."""
        # Plot the new data
        self.plot(data_to_plot, title=title) # This method already calls self.draw()

        try:
            self.figure.savefig(filename)
            parent_widget = self.parentWidget()
            if not parent_widget: parent_widget = self # Fallback
            QMessageBox.information(parent_widget, "Plot Saved", f"Plot saved as {filename}")
        except Exception as e:
            parent_widget = self.parentWidget()
            if not parent_widget: parent_widget = self
            QMessageBox.critical(parent_widget, "Save Error", f"Could not save plot: {str(e)}")

class TrainingPlotCanvas(FigureCanvas): # Also moving TrainingPlotCanvas here as it\'s a canvas
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(TrainingPlotCanvas, self).__init__(fig)
        if parent:
            self.setParent(parent)

    def plot(self, data_window, title="EEG Sample", num_points=500):
        self.axes.clear()
        timepoints_to_plot = min(data_window.shape[1], num_points)
        
        for i in range(data_window.shape[0]): # Iterate over channels
            self.axes.plot(data_window[i, :timepoints_to_plot], label=f'Ch {i+1}')
        
        self.axes.set_title(title)
        self.axes.set_xlabel(f"Timepoints (first {timepoints_to_plot})")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True)
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_title("No Data to Display")
        self.axes.set_xlabel("Timepoints")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True)
        self.draw()
