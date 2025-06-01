"""
Module:  model.__init__
Purpose: Initializes the model module for the BCI project, making model classes and training scripts accessible.
Author:  Copilot (NASA-style guidelines)
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
"""

# Model module for BCI project
from .base_model import Model
from .eeg_inception_erp import EEGInceptionERPModel
from .eeg_it_net import EEGITNetModel
from .train_model import main as train_main_script

__all__ = ['Model', 'EEGInceptionERPModel', 'EEGITNetModel', 'train_main_script']
