import os
import torch
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Any
from src.model.model_trainer import ModelTrainer
from .TrainingConfig import TrainingConfig

class TrainingManager(QObject):
    """
    UI manager for training process
    
    Signals:
        training_started: Emitted when training begins
        fold_started(int): Emitted when a new fold starts, passing fold number
        epoch_completed(int, float, float, float, float): Emitted after each epoch with:
            - epoch number
            - training loss
            - training accuracy
            - validation loss
            - validation accuracy
        fold_completed(int, float): Emitted when a fold completes with:
            - fold number
            - validation accuracy
        training_completed(dict): Emitted with final results dictionary
        training_error(str): Emitted when an error occurs during training
    """
    
    # Signals without type hints
    training_started = pyqtSignal()
    fold_started = pyqtSignal(int)
    epoch_completed = pyqtSignal(int, float, float, float, float)
    fold_completed = pyqtSignal(int, float)
    training_completed = pyqtSignal(dict)  # Changed from Dict[str, Any] to dict
    training_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.config = TrainingConfig()
        self.trainer = ModelTrainer()
    
    def configure(self, **kwargs) -> None:
        """Configure training parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update trainer config
        self.trainer.configure_training(self.config.__dict__)
    
    def prepare_data(self, windows, labels, subject_ids=None) -> None:
        """Prepare data for training"""
        self.trainer.prepare_data(windows, labels, subject_ids)
    
    def start_training(self) -> None:
        """Start the training process"""
        try:
            callbacks = {
                'training_started': lambda: self.training_started.emit(),
                'fold_started': lambda f: self.fold_started.emit(f),
                'epoch_completed': lambda *args: self.epoch_completed.emit(*args),
                'fold_completed': lambda f, acc: self.fold_completed.emit(f, acc),
                'training_completed': lambda r: self.training_completed.emit(r),
                'error_occurred': lambda msg: self.training_error.emit(msg)
            }
            
            if self.config.k_folds > 1:
                self.trainer.start_kfold_training(callbacks)
            else:
                self.trainer.start_single_training(callbacks)
                
        except Exception as e:
            self.training_error.emit(f"Error starting training: {str(e)}")
    
    def stop_training(self) -> None:
        """Stop the current training process"""
        self.trainer.stop_training()
