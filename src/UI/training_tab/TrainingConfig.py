from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class TrainingConfig:
    """Configuration for the training process"""
    # Model parameters
    model_type: str = 'EEGInceptionERP'
    num_classes: int = 2 
    input_channels: int = 60 
    sfreq: float = 160.0
    input_window_samples: int = 1120 
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    optimizer_name: str = 'Adam'
    criterion_name: str = 'CrossEntropyLoss'
    
    # Data handling
    k_folds: int = 5
    test_split_ratio: float = 0.2  # If not using k-fold, or for final model test set
    use_subject_specific_split: bool = False # For subject-aware k-fold or train/val split

    # Regularization and Early Stopping
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    
    # Hardware and Logging
    device: str = 'cuda' # 'cuda' or 'cpu'
    log_interval: int = 10 # Log every N batches/epochs
    
    # Paths
    save_dir: str = 'models'
    plot_dir: str = 'plots'
    
    # Optional additional parameters for specific models or optimizers
    additional_model_params: Dict[str, Any] = field(default_factory=dict)
    additional_optimizer_params: Dict[str, Any] = field(default_factory=dict)

    # For GUI state, not core training logic
    selected_subjects: Optional[list] = field(default_factory=list) 
    data_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary, useful for logging or saving."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
