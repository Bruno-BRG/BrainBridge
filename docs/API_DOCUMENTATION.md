# BCI System API Documentation

## Overview

This document provides comprehensive API documentation for the BCI EEG Motor Imagery Classification System. The APIs are organized by functional modules and include detailed examples for developers.

## Core Data APIs

### BCIDataLoader

The main class for loading and preprocessing EEG data from various sources.

```python
from src.data.data_loader import BCIDataLoader

class BCIDataLoader:
    """
    Load and preprocess EEG data from CSV files with PhysioNet format.
    
    Supports automatic filtering, windowing, and event extraction.
    """
    
    def __init__(self, data_dir: str, subjects: List[int], runs: List[int]):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the EEG data directory
            subjects (List[int]): List of subject IDs to load
            runs (List[int]): List of run numbers to load
            
        Example:
            >>> loader = BCIDataLoader("./eeg_data", [1, 2, 3], [4, 8, 12])
        """
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess all EEG data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (data, labels)
                - data: Shape (n_samples, n_channels, n_timepoints)
                - labels: Shape (n_samples,) with class labels
                
        Example:
            >>> data, labels = loader.load_data()
            >>> print(f"Loaded {data.shape[0]} samples")
        """
        
    def preprocess_signal(self, eeg_data: np.ndarray, sfreq: float = 125) -> np.ndarray:
        """
        Apply preprocessing filters to EEG data.
        
        Args:
            eeg_data (np.ndarray): Raw EEG data
            sfreq (float): Sampling frequency in Hz
            
        Returns:
            np.ndarray: Filtered EEG data
            
        Example:
            >>> filtered_data = loader.preprocess_signal(raw_data)
        """
```

### BCIDataset

PyTorch dataset class for EEG data with augmentation support.

```python
from src.data.data_loader import BCIDataset

class BCIDataset(Dataset):
    """
    PyTorch Dataset for EEG data with optional augmentation.
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 apply_augmentation: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data (np.ndarray): EEG data array
            labels (np.ndarray): Label array
            apply_augmentation (bool): Whether to apply data augmentation
            
        Example:
            >>> dataset = BCIDataset(data, labels, apply_augmentation=True)
            >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with optional augmentation.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (sample, label)
        """
        
    def apply_augmentation(self, data: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation techniques.
        
        Args:
            data (np.ndarray): Input EEG sample
            
        Returns:
            np.ndarray: Augmented EEG sample
            
        Augmentation methods:
        - Temporal jittering (±10% time shift)
        - Amplitude scaling (0.8-1.2x)
        - Gaussian noise addition (σ=0.1)
        """
```

## Model APIs

### EEGInceptionERPModel

The main model class implementing the EEGInceptionERP architecture.

```python
from src.model.eeg_inception_erp import EEGInceptionERPModel

class EEGInceptionERPModel(BaseModel):
    """
    EEGInceptionERP model wrapper with training and inference capabilities.
    """
    
    def __init__(self, n_chans: int = 16, n_outputs: int = 2, 
                 n_times: int = 400, sfreq: float = 125):
        """
        Initialize the model.
        
        Args:
            n_chans (int): Number of EEG channels
            n_outputs (int): Number of output classes
            n_times (int): Number of time points per sample
            sfreq (float): Sampling frequency
            
        Example:
            >>> model = EEGInceptionERPModel(n_chans=16, n_outputs=2)
        """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor shape (batch, channels, time)
            
        Returns:
            torch.Tensor: Output predictions shape (batch, n_classes)
            
        Example:
            >>> predictions = model(input_batch)
            >>> probabilities = torch.softmax(predictions, dim=1)
        """
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x (np.ndarray): Input data shape (n_samples, channels, time)
            
        Returns:
            np.ndarray: Predicted class probabilities
            
        Example:
            >>> probabilities = model.predict(test_data)
            >>> predictions = np.argmax(probabilities, axis=1)
        """
        
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            dict: Model configuration and parameters
            
        Example:
            >>> info = model.get_model_info()
            >>> print(f"Model has {info['total_params']} parameters")
        """
```

### EEGTrainer

Comprehensive training class with cross-validation and callbacks.

```python
from src.model.train_model import EEGTrainer

class EEGTrainer:
    """
    Complete training pipeline with K-fold cross-validation.
    """
    
    def __init__(self, model: EEGInceptionERPModel, config: dict = None):
        """
        Initialize the trainer.
        
        Args:
            model (EEGInceptionERPModel): Model to train
            config (dict): Training configuration
            
        Default config:
            {
                "epochs": 30,
                "k_folds": 10,
                "learning_rate": 0.001,
                "batch_size": 10,
                "early_stopping_patience": 8,
                "test_split_ratio": 0.2
            }
        """
        
    def train(self, data: np.ndarray, labels: np.ndarray) -> dict:
        """
        Train the model with cross-validation.
        
        Args:
            data (np.ndarray): Training data
            labels (np.ndarray): Training labels
            
        Returns:
            dict: Training results and statistics
            
        Example:
            >>> results = trainer.train(train_data, train_labels)
            >>> print(f"Best CV accuracy: {results['best_cv_accuracy']:.3f}")
        """
        
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            data (np.ndarray): Test data
            labels (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
            
        Example:
            >>> metrics = trainer.evaluate(test_data, test_labels)
            >>> print(f"Test accuracy: {metrics['accuracy']:.3f}")
        """
```

## Real-Time Processing APIs

### RealTimeInferenceProcessor

Handles real-time EEG processing with sliding window approach.

```python
from src.model.realtime_inference import RealTimeInferenceProcessor

class RealTimeInferenceProcessor:
    """
    Real-time EEG processing for live inference.
    """
    
    def __init__(self, model: EEGInceptionERPModel, window_size: int = 400):
        """
        Initialize the real-time processor.
        
        Args:
            model (EEGInceptionERPModel): Trained model for inference
            window_size (int): Size of sliding window
            
        Example:
            >>> processor = RealTimeInferenceProcessor(trained_model)
        """
        
    def process_sample(self, sample: np.ndarray) -> Optional[float]:
        """
        Process a single EEG sample.
        
        Args:
            sample (np.ndarray): Single EEG sample (n_channels,)
            
        Returns:
            Optional[float]: Prediction confidence or None if buffer not full
            
        Example:
            >>> confidence = processor.process_sample(new_eeg_sample)
            >>> if confidence is not None:
            ...     print(f"Prediction confidence: {confidence:.3f}")
        """
        
    def reset_buffer(self):
        """
        Reset the internal buffer.
        
        Example:
            >>> processor.reset_buffer()  # Start new session
        """
        
    def get_buffer_status(self) -> dict:
        """
        Get current buffer status information.
        
        Returns:
            dict: Buffer status including fill level and statistics
            
        Example:
            >>> status = processor.get_buffer_status()
            >>> print(f"Buffer fill: {status['fill_percentage']:.1f}%")
        """
```

### PyLSLStreamReader

Interface for Lab Streaming Layer EEG data streams.

```python
from src.utils.pylsl_utils import PyLSLStreamReader

class PyLSLStreamReader:
    """
    Handle PyLSL EEG data streams for real-time processing.
    """
    
    def __init__(self, stream_name: str = "EEG"):
        """
        Initialize the stream reader.
        
        Args:
            stream_name (str): Name of the LSL stream to connect to
            
        Example:
            >>> reader = PyLSLStreamReader("EEG")
        """
        
    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to the LSL stream.
        
        Args:
            timeout (float): Connection timeout in seconds
            
        Returns:
            bool: True if connection successful
            
        Example:
            >>> if reader.connect():
            ...     print("Connected to EEG stream")
        """
        
    def get_samples(self, max_samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latest samples from the stream.
        
        Args:
            max_samples (int): Maximum number of samples to retrieve
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (samples, timestamps)
            
        Example:
            >>> samples, timestamps = reader.get_samples()
            >>> if len(samples) > 0:
            ...     # Process new samples
        """
        
    def disconnect(self):
        """
        Disconnect from the LSL stream.
        
        Example:
            >>> reader.disconnect()
        """
```

## Patient Management APIs

### PatientDataManager

Manages patient-specific data organization and processing.

```python
from EndUser.src.data.patient_data_manager import PatientDataManager

class PatientDataManager:
    """
    Manage patient data, recordings, and model training.
    """
    
    def __init__(self, data_dir: str = "patient_data"):
        """
        Initialize the patient data manager.
        
        Args:
            data_dir (str): Base directory for patient data
            
        Example:
            >>> manager = PatientDataManager("./patient_data")
        """
        
    def register_patient(self, patient_info: dict) -> str:
        """
        Register a new patient.
        
        Args:
            patient_info (dict): Patient information dictionary
                Required keys: 'name', 'age', 'gender'
                Optional keys: 'affected_hand', 'onset_time', 'notes'
                
        Returns:
            str: Generated patient ID
            
        Example:
            >>> patient_id = manager.register_patient({
            ...     'name': 'John Doe',
            ...     'age': 45,
            ...     'gender': 'Male',
            ...     'affected_hand': 'Right',
            ...     'onset_time': '6 months'
            ... })
        """
        
    def get_patient_list(self) -> List[dict]:
        """
        Get list of all registered patients.
        
        Returns:
            List[dict]: List of patient information dictionaries
            
        Example:
            >>> patients = manager.get_patient_list()
            >>> for patient in patients:
            ...     print(f"{patient['id']}: {patient['name']}")
        """
        
    def get_patient_data(self, patient_id: str) -> dict:
        """
        Get complete patient data including recordings and models.
        
        Args:
            patient_id (str): Patient identifier
            
        Returns:
            dict: Complete patient data
            
        Example:
            >>> data = manager.get_patient_data("PAT001")
            >>> print(f"Patient has {len(data['recordings'])} recordings")
        """
        
    def add_recording(self, patient_id: str, recording_path: str, 
                     session_info: dict = None) -> str:
        """
        Add a new EEG recording for a patient.
        
        Args:
            patient_id (str): Patient identifier
            recording_path (str): Path to the EEG recording file
            session_info (dict): Optional session metadata
            
        Returns:
            str: Recording identifier
            
        Example:
            >>> recording_id = manager.add_recording(
            ...     "PAT001", 
            ...     "./recordings/session_1.csv",
            ...     {"date": "2025-01-15", "duration": "15 minutes"}
            ... )
        """
        
    def train_patient_model(self, patient_id: str, config: dict = None) -> dict:
        """
        Train a patient-specific model using their recordings.
        
        Args:
            patient_id (str): Patient identifier
            config (dict): Training configuration
            
        Returns:
            dict: Training results and model information
            
        Example:
            >>> results = manager.train_patient_model("PAT001")
            >>> print(f"Model accuracy: {results['accuracy']:.3f}")
        """
```

## Visualization APIs

### PlotCanvas

Advanced EEG visualization with matplotlib backend.

```python
from src.UI.plot_canvas import PlotCanvas

class PlotCanvas:
    """
    EEG signal visualization with multiple plot types.
    """
    
    def __init__(self, parent=None, width: int = 800, height: int = 600):
        """
        Initialize the plot canvas.
        
        Args:
            parent: Parent Qt widget
            width (int): Canvas width in pixels
            height (int): Canvas height in pixels
        """
        
    def plot_eeg_signals(self, data: np.ndarray, channels: List[str], 
                        sfreq: float = 125, title: str = "EEG Signals"):
        """
        Plot multi-channel EEG signals.
        
        Args:
            data (np.ndarray): EEG data shape (n_channels, n_timepoints)
            channels (List[str]): Channel names
            sfreq (float): Sampling frequency
            title (str): Plot title
            
        Example:
            >>> canvas.plot_eeg_signals(
            ...     eeg_data, 
            ...     ['C3', 'C4', 'Cz'], 
            ...     title="Motor Imagery EEG"
            ... )
        """
        
    def plot_spectrogram(self, data: np.ndarray, channel_idx: int = 0, 
                        sfreq: float = 125):
        """
        Plot spectrogram for a specific channel.
        
        Args:
            data (np.ndarray): EEG data
            channel_idx (int): Channel index to plot
            sfreq (float): Sampling frequency
            
        Example:
            >>> canvas.plot_spectrogram(eeg_data, channel_idx=0)
        """
        
    def plot_training_history(self, history: dict):
        """
        Plot training loss and accuracy curves.
        
        Args:
            history (dict): Training history with 'loss' and 'accuracy' keys
            
        Example:
            >>> canvas.plot_training_history(trainer.get_history())
        """
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None):
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Class names for labels
            
        Example:
            >>> canvas.plot_confusion_matrix(
            ...     true_labels, 
            ...     predictions, 
            ...     ['Left Hand', 'Right Hand']
            ... )
        """
```

## Utility APIs

### Configuration Management

```python
from src.utils.config import ConfigManager

class ConfigManager:
    """
    Manage application configuration and settings.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str): Path to configuration file
        """
        
    def get_config(self, section: str = None) -> dict:
        """
        Get configuration values.
        
        Args:
            section (str): Configuration section name, None for all
            
        Returns:
            dict: Configuration values
            
        Example:
            >>> config = ConfigManager()
            >>> model_config = config.get_config('model')
            >>> print(f"Learning rate: {model_config['learning_rate']}")
        """
        
    def set_config(self, key: str, value: Any, section: str = None):
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
            section (str): Configuration section
            
        Example:
            >>> config.set_config('learning_rate', 0.001, 'model')
        """
        
    def save_config(self):
        """
        Save configuration to file.
        
        Example:
            >>> config.save_config()
        """
```

### Logging Utilities

```python
from src.utils.logging import setup_logging, get_logger

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Set up application logging.
    
    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file (str): Optional log file path
        
    Example:
        >>> setup_logging('DEBUG', 'bci_app.log')
    """

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
```

## Error Handling

### Custom Exceptions

```python
class BCIException(Exception):
    """Base exception for BCI application."""
    pass

class DataLoadError(BCIException):
    """Raised when data loading fails."""
    pass

class ModelError(BCIException):
    """Raised when model operations fail."""
    pass

class StreamError(BCIException):
    """Raised when stream operations fail."""
    pass

class PatientError(BCIException):
    """Raised when patient operations fail."""
    pass

# Usage examples:
try:
    data, labels = loader.load_data()
except DataLoadError as e:
    logger.error(f"Failed to load data: {e}")
    
try:
    model.train(data, labels)
except ModelError as e:
    logger.error(f"Training failed: {e}")
```

## API Usage Examples

### Complete Training Pipeline

```python
from src.data.data_loader import BCIDataLoader
from src.model.eeg_inception_erp import EEGInceptionERPModel
from src.model.train_model import EEGTrainer

# Load data
loader = BCIDataLoader("./eeg_data", subjects=[1, 2, 3], runs=[4, 8, 12])
data, labels = loader.load_data()

# Create model
model = EEGInceptionERPModel(n_chans=16, n_outputs=2)

# Train model
trainer = EEGTrainer(model)
results = trainer.train(data, labels)

print(f"Training completed with {results['best_cv_accuracy']:.3f} accuracy")
```

### Real-Time Processing Setup

```python
from src.model.realtime_inference import RealTimeInferenceProcessor
from src.utils.pylsl_utils import PyLSLStreamReader

# Load trained model
model = EEGInceptionERPModel.load("trained_model.pth")

# Setup real-time processing
processor = RealTimeInferenceProcessor(model)
stream_reader = PyLSLStreamReader("EEG")

# Connect to stream
if stream_reader.connect():
    print("Connected to EEG stream")
    
    while True:
        samples, timestamps = stream_reader.get_samples()
        
        for sample in samples:
            prediction = processor.process_sample(sample)
            if prediction is not None:
                print(f"Prediction: {prediction:.3f}")
```

### Patient Management Workflow

```python
from EndUser.src.data.patient_data_manager import PatientDataManager

# Initialize patient manager
manager = PatientDataManager()

# Register new patient
patient_id = manager.register_patient({
    'name': 'Jane Doe',
    'age': 52,
    'gender': 'Female',
    'affected_hand': 'Left',
    'onset_time': '3 months'
})

# Add recording
recording_id = manager.add_recording(
    patient_id, 
    "./recordings/session_1.csv"
)

# Train patient-specific model
results = manager.train_patient_model(patient_id)
print(f"Patient model trained with {results['accuracy']:.3f} accuracy")
```

This API documentation provides comprehensive reference for all major components of the BCI system, enabling developers to integrate and extend the system effectively.
