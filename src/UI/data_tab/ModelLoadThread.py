import torch
from PyQt5.QtCore import QThread, pyqtSignal
from src.model.eeg_inception_erp import EEGModel

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
