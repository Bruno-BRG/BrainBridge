# Model module for BCI project
from .base_model import Model
from .eeg_inception_erp import EEGInceptionERPModel
from .train_model import main as train_main_script

__all__ = ['Model', 'EEGInceptionERPModel', 'train_main_script']
