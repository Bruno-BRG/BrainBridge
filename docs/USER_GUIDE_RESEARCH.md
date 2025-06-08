# 🔬 Research User Guide

> **Comprehensive guide for researchers, data scientists, and developers using the BCI DevTools environment**

## 🎯 Overview

The DevTools interface provides a comprehensive research environment for EEG motor imagery analysis, featuring advanced machine learning capabilities, extensive data visualization, and sophisticated model development tools.

## 🚀 Getting Started

### System Launch

```bash
# Launch DevTools interface
python launch_bci.py --mode gui

# Advanced launch with GPU acceleration
python launch_bci.py --mode gui --gpu --verbose

# Batch processing mode (command line)
python launch_bci.py --mode train --config research_config.yaml
```

## 📊 Data Management Workflow

### 1. Dataset Preparation

#### PhysioNet EEG Motor Movement/Imagery Database

**Automatic Download and Setup:**

1. **Navigate to Data Management Tab**
   - First tab in DevTools interface
   - Shows current data path and status

2. **Configure Data Directory**
   ```
   Default Path: ./eeg_data/MNE-eegbci-data/files/eegmmidb/1.0.0/
   
   Browse Options:
   - Select existing PhysioNet data
   - Specify custom dataset location
   - Multiple dataset support
   ```

3. **Subject Selection**
   ```
   Subject IDs:
   - "all" (loads all 109 subjects)
   - "1,2,3" (specific subjects)
   - "1-10" (range selection)
   - "1,5,10-15" (mixed selection)
   ```

4. **Load and Process Data**
   - Click "Load and Process Data"
   - Automatic preprocessing pipeline execution
   - Progress monitoring with detailed logs

#### Custom Dataset Integration

**Supported Formats:**
- CSV files (OpenBCI format)
- EDF files (European Data Format)
- BDF files (BioSemi Data Format)
- FIF files (Neuromag/Elekta)

**Data Structure Requirements:**
```python
# CSV Format (OpenBCI compatible)
columns = [
    'Sample Index', 'Channel 1', 'Channel 2', ..., 'Channel N',
    'Timestamp', 'Event', 'Label'
]

# Event Codes
events = {
    'T1': 'Left Hand Motor Imagery',
    'T2': 'Right Hand Motor Imagery',
    'Rest': 'Baseline/Rest State'
}
```

### 2. Data Preprocessing Pipeline

#### Automatic Preprocessing Steps:

```python
# Signal Processing Pipeline
preprocessing_steps = [
    "Bandpass Filtering (8-30 Hz)",        # Motor imagery relevant frequencies
    "Notch Filtering (50/60 Hz)",          # Power line noise removal
    "Common Average Referencing",          # Spatial filtering
    "Artifact Rejection",                  # Automated artifact detection
    "Baseline Correction",                 # Zero-mean centering
    "Windowing (1-second epochs)",         # Temporal segmentation
    "Feature Normalization"                # Z-score standardization
]
```

#### Manual Preprocessing Controls:

**Filter Configuration:**
```python
filter_params = {
    'low_cutoff': 8.0,      # Hz - Motor imagery lower bound
    'high_cutoff': 30.0,    # Hz - Motor imagery upper bound
    'notch_freq': 50.0,     # Hz - Power line frequency
    'filter_order': 4,      # Butterworth filter order
    'zero_phase': True      # Forward-backward filtering
}
```

**Artifact Rejection:**
```python
artifact_params = {
    'amplitude_threshold': 100e-6,  # μV - Voltage threshold
    'gradient_threshold': 75e-6,    # μV - Gradient threshold
    'variance_threshold': 'auto',   # Automatic variance calculation
    'correlation_threshold': 0.8    # Inter-channel correlation
}
```

## 🧠 Machine Learning Workflow

### 1. Model Architecture Configuration

#### EEGInceptionERP Parameters:

```python
model_config = {
    'n_channels': 64,           # Number of EEG channels
    'n_classes': 2,             # Left vs. Right hand
    'input_window_samples': 250, # 1 second at 250 Hz
    'drop_prob': 0.5,           # Dropout probability
    'kernel_length': 64,        # Temporal kernel size
    'F1': 8,                    # First conv layer filters
    'D': 2,                     # Depth multiplier
    'F2': 16,                   # Second conv layer filters
    'norm_rate': 0.25,          # Batch normalization momentum
    'dropout_rate': 0.25        # Spatial dropout rate
}
```

#### Advanced Architecture Options:

**Custom Model Selection:**
- EEGInceptionERP (Default - State-of-the-art)
- EEGNet (Lightweight alternative)
- DeepConvNet (Deep learning baseline)
- ShallowConvNet (Classical approach)
- Custom architecture support

### 2. Training Configuration

#### Comprehensive Training Parameters:

```python
training_config = {
    # Data Split Configuration
    'test_size': 0.2,           # Hold-out test set
    'validation_size': 0.2,     # Validation during training
    'cross_validation_folds': 5, # K-fold CV
    'stratified_split': True,   # Balanced class distribution
    
    # Training Hyperparameters
    'batch_size': 32,           # Mini-batch size
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'weight_decay': 0.01,       # L2 regularization
    'epochs': 200,              # Maximum training epochs
    'early_stopping_patience': 15, # Early stopping patience
    
    # Learning Rate Scheduling
    'lr_scheduler': 'plateau',  # ReduceLROnPlateau
    'lr_factor': 0.5,          # LR reduction factor
    'lr_patience': 10,         # LR scheduler patience
    'lr_min': 1e-6,            # Minimum learning rate
    
    # Data Augmentation
    'augmentation_enabled': True,
    'noise_level': 0.05,       # Gaussian noise std
    'time_shift_max': 20,      # Maximum time shift (samples)
    'amplitude_scale_range': (0.9, 1.1) # Amplitude scaling
}
```

#### Advanced Training Features:

**Multi-GPU Support:**
```python
training_config.update({
    'device': 'cuda',           # GPU acceleration
    'multi_gpu': True,          # DataParallel training
    'mixed_precision': True,    # Automatic Mixed Precision
    'compile_model': True       # PyTorch 2.0 compilation
})
```

**Experiment Tracking:**
```python
experiment_config = {
    'experiment_name': 'motor_imagery_v2',
    'logging_enabled': True,
    'save_checkpoints': True,
    'checkpoint_frequency': 10,  # Save every 10 epochs
    'tensorboard_logging': True,
    'wandb_integration': False   # Optional: Weights & Biases
}
```

### 3. Training Execution

#### Step-by-Step Training Process:

1. **Configure Training Parameters**
   ```python
   # In Training Tab
   - Select model architecture
   - Set training hyperparameters
   - Configure cross-validation
   - Enable/disable augmentation
   ```

2. **Start Training**
   ```python
   # Training execution
   - Real-time loss monitoring
   - Validation accuracy tracking
   - Learning rate adaptation
   - Best model checkpointing
   ```

3. **Monitor Progress**
   ```python
   # Real-time monitoring
   - Training/validation loss curves
   - Accuracy progression
   - Learning rate evolution
   - Memory usage tracking
   ```

#### Training Output and Logging:

**Console Output:**
```
Epoch 25/200:
├── Train Loss: 0.3245 (↓0.0123)
├── Train Acc:  78.34% (↑1.23%)
├── Val Loss:   0.4156 (↓0.0089)
├── Val Acc:    73.21% (↑0.67%)
├── LR:         0.0005 (↓50%)
└── Time:       45.2s/epoch
```

**Saved Artifacts:**
```
models/
├── {model_name}_best.pth           # Best validation model
├── {model_name}_final.pth          # Final epoch model
├── training_history.json           # Complete training log
├── model_config.json               # Model configuration
└── performance_metrics.json        # Final performance
```

## 🎯 Advanced Model Fine-Tuning

### 1. Patient-Specific Fine-Tuning

#### Transfer Learning Workflow:

1. **Load Pre-trained Model**
   ```python
   # In Fine-Tuning Tab
   - Select base model (trained on PhysioNet)
   - Choose patient data directory
   - Configure patient data manager
   ```

2. **Fine-Tuning Strategy Selection**
   ```python
   fine_tuning_strategies = {
       'full_fine_tuning': {
           'freeze_layers': None,
           'learning_rate_ratio': 1.0
       },
       'feature_extraction': {
           'freeze_layers': ['conv_layers'],
           'learning_rate_ratio': 0.1
       },
       'gradual_unfreezing': {
           'freeze_layers': 'progressive',
           'unfreeze_schedule': [10, 20, 30]  # Epochs
       }
   }
   ```

3. **Patient Data Integration**
   ```python
   patient_data_config = {
       'recording_sessions': ['session_001', 'session_002'],
       'validation_split': 0.3,
       'augmentation_factor': 5,    # Data augmentation multiplier
       'quality_threshold': 0.85    # Signal quality minimum
   }
   ```

#### Advanced Fine-Tuning Features:

**Domain Adaptation:**
```python
domain_adaptation = {
    'method': 'adversarial',        # Adversarial domain adaptation
    'lambda_domain': 0.1,           # Domain loss weight
    'gradient_reversal': True,      # Gradient reversal layer
    'domain_classifier_depth': 2    # Domain discriminator layers
}
```

**Few-Shot Learning:**
```python
few_shot_config = {
    'meta_learning': True,          # MAML-style meta-learning
    'support_shots': 5,             # Shots per class
    'query_shots': 15,              # Query samples per class
    'meta_lr': 0.001,              # Meta-learning rate
    'inner_lr': 0.01               # Inner loop learning rate
}
```

### 2. Hyperparameter Optimization

#### Automated Hyperparameter Search:

```python
optimization_config = {
    'method': 'optuna',             # Optuna-based optimization
    'n_trials': 100,                # Number of trials
    'pruning_enabled': True,        # Early trial pruning
    'search_space': {
        'learning_rate': ('log', 1e-5, 1e-2),
        'batch_size': ('categorical', [16, 32, 64]),
        'dropout_rate': ('uniform', 0.1, 0.7),
        'weight_decay': ('log', 1e-5, 1e-1)
    }
}
```

## 📊 Advanced Visualization and Analysis

### 1. Real-Time Monitoring

#### Training Visualization:

```python
visualization_features = [
    "Real-time loss curves",
    "Accuracy progression plots",
    "Learning rate evolution",
    "Gradient norm tracking",
    "Weight histogram evolution",
    "Confusion matrix updates",
    "ROC curve progression",
    "Feature map visualization"
]
```

#### Signal Analysis:

```python
signal_analysis_tools = [
    "Multi-channel EEG plots",
    "Spectral density analysis",
    "Time-frequency decomposition",
    "Coherence analysis",
    "Event-related potential plots",
    "Topographic maps",
    "Source localization",
    "Connectivity analysis"
]
```

### 2. Advanced Analytics

#### Model Interpretability:

```python
interpretability_methods = {
    'gradient_based': {
        'vanilla_gradients': True,
        'integrated_gradients': True,
        'guided_backprop': True,
        'grad_cam': True
    },
    'perturbation_based': {
        'occlusion_analysis': True,
        'lime_explanation': True,
        'shap_values': True
    },
    'attention_analysis': {
        'attention_weights': True,
        'attention_rollout': True,
        'attention_flow': True
    }
}
```

#### Performance Analysis:

```python
performance_metrics = {
    'classification_metrics': [
        'accuracy', 'precision', 'recall', 'f1_score',
        'specificity', 'sensitivity', 'mcc', 'kappa'
    ],
    'probabilistic_metrics': [
        'log_loss', 'brier_score', 'calibration_error'
    ],
    'ranking_metrics': [
        'auc_roc', 'auc_pr', 'average_precision'
    ],
    'temporal_metrics': [
        'prediction_latency', 'inference_time',
        'memory_usage', 'throughput'
    ]
}
```

## 🔬 Research Protocols

### 1. Experimental Design

#### Cross-Subject Validation:

```python
cross_subject_protocol = {
    'leave_one_subject_out': True,
    'subject_independent': True,
    'cross_session_validation': True,
    'temporal_generalization': True
}
```

#### Statistical Analysis:

```python
statistical_analysis = {
    'significance_testing': {
        'paired_t_test': True,
        'wilcoxon_signed_rank': True,
        'mcnemar_test': True,
        'bootstrap_confidence': True
    },
    'effect_size': {
        'cohens_d': True,
        'eta_squared': True,
        'cliff_delta': True
    },
    'multiple_comparisons': {
        'bonferroni_correction': True,
        'fdr_correction': True,
        'holm_method': True
    }
}
```

### 2. Data Export and Sharing

#### Research Data Export:

```python
export_formats = {
    'raw_data': ['CSV', 'EDF', 'BDF', 'FIF'],
    'processed_data': ['NumPy', 'HDF5', 'Parquet'],
    'models': ['PyTorch', 'ONNX', 'TensorFlow'],
    'results': ['JSON', 'CSV', 'Excel', 'LaTeX']
}
```

#### BIDS Compliance:

```python
bids_export = {
    'structure_compliance': True,
    'metadata_generation': True,
    'participant_anonymization': True,
    'data_validation': True
}
```

## 🛠️ Advanced Development

### 1. Custom Model Development

#### Model Architecture Framework:

```python
class CustomEEGModel(nn.Module):
    """
    Template for custom EEG model development
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Custom architecture implementation
        
    def forward(self, x):
        # Forward pass implementation
        return output
        
    def compute_loss(self, predictions, targets):
        # Custom loss function
        return loss
```

#### Integration with DevTools:

```python
# Register custom model
from custom_models import CustomEEGModel

# In model selection dropdown
available_models = {
    'EEGInceptionERP': EEGInceptionERP,
    'CustomModel': CustomEEGModel,
    # Add more custom models
}
```

### 2. Advanced Signal Processing

#### Custom Preprocessing Pipeline:

```python
class CustomPreprocessor:
    """
    Custom preprocessing pipeline for specialized research needs
    """
    def __init__(self, config):
        self.config = config
        
    def preprocess(self, raw_data):
        # Custom preprocessing steps
        processed_data = self.apply_custom_filters(raw_data)
        processed_data = self.custom_artifact_removal(processed_data)
        processed_data = self.custom_feature_extraction(processed_data)
        return processed_data
```

## 📈 Performance Optimization

### 1. Computational Optimization

#### GPU Acceleration:

```python
optimization_settings = {
    'device': 'cuda',
    'mixed_precision': True,
    'gradient_checkpointing': True,
    'dataloader_workers': 8,
    'pin_memory': True,
    'persistent_workers': True
}
```

#### Memory Management:

```python
memory_optimization = {
    'gradient_accumulation_steps': 4,
    'max_batch_size_auto': True,
    'memory_efficient_attention': True,
    'activation_checkpointing': True
}
```

### 2. Parallel Processing

#### Multi-GPU Training:

```python
parallel_config = {
    'strategy': 'data_parallel',
    'devices': [0, 1, 2, 3],
    'find_unused_parameters': False,
    'gradient_as_bucket_view': True
}
```

#### Distributed Training:

```python
distributed_config = {
    'backend': 'nccl',
    'init_method': 'env://',
    'world_size': 4,
    'rank': 0
}
```

---

## 🔧 Configuration Files

### Research Configuration Template:

```yaml
# research_config.yaml
experiment:
  name: "motor_imagery_study_2025"
  description: "Advanced motor imagery classification"
  
data:
  dataset: "physionet_eegmmidb"
  subjects: "all"
  preprocessing:
    bandpass: [8, 30]
    notch: 50
    car: true
    
model:
  architecture: "EEGInceptionERP"
  parameters:
    n_channels: 64
    n_classes: 2
    dropout: 0.5
    
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 200
  cross_validation: 5
  
optimization:
  device: "cuda"
  mixed_precision: true
  gradient_accumulation: 1
```

This comprehensive guide enables researchers to leverage the full potential of the BCI DevTools environment for cutting-edge EEG motor imagery research.
