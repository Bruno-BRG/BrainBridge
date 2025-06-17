"""
Module:  model.__init__
Purpose: Initializes the model module for the BCI project.
Author:  Copilot (NASA-style guidelines)  
Created: 2025-05-28
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
"""

from .fine_tuning import ModelFineTuner, fine_tune_model_simple

__all__ = [
    'ModelFineTuner',
    'fine_tune_model_simple'
]
