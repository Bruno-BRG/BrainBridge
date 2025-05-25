# EEG Motor Imagery Classification Project

This project implements a complete pipeline for EEG motor imagery classification using the EEGInceptionERP model from braindecode. The system can classify left vs right hand motor imagery from EEG signals.

## 🎯 Project Overview

- **Objective**: Classify motor imagery tasks (left vs right hand) from EEG signals
- **Dataset**: PhysioNet Motor Movement/Imagery Dataset (runs 4, 8, 12)
- **Model**: EEGInceptionERP - A state-of-the-art CNN architecture for EEG classification
- **Framework**: PyTorch with custom data loading and preprocessing

## 📁 Project Structure

```
projetoBCI/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # EEG data loading and preprocessing
│   └── model/
│       ├── __init__.py
│       └── eeg_inception_erp.py    # EEGInceptionERP model implementation
├── eeg_data/                       # EEG dataset directory
├── plots/                          # Training plots and visualizations
├── requirements.txt                # Python dependencies
├── train_model.py                  # Full training script with K-fold CV
├── minimal_train.py               # Simple training example
├── evaluate_model.py              # Model evaluation and inference
├── simple_test.py                 # Basic functionality test
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your EEG data in the following structure:
```
eeg_data/
└── MNE-eegbci-data/
    └── files/
        └── eegmmidb/
            └── 1.0.0/
                ├── S001/
                │   ├── S001R04_csv_openbci.csv
                │   ├── S001R08_csv_openbci.csv
                │   └── S001R12_csv_openbci.csv
                ├── S002/
                └── ...
```

### 3. Test the Setup

```bash
python simple_test.py
```

### 4. Train the Model

For a quick test with synthetic data:
```bash
python minimal_train.py
```

For full training with real EEG data:
```bash
python train_model.py
```

### 5. Evaluate the Model

```bash
python evaluate_model.py
```

## 🔧 Key Components

### Data Loader (`src/data/data_loader.py`)

- **BCIDataLoader**: Main class for loading EEG data from CSV files
- **BCIDataset**: PyTorch dataset class with data augmentation
- Features:
  - Bandpass filtering (0.5-50 Hz)
  - Notch filtering (50 Hz power line noise)
  - Z-score standardization
  - Windowing with configurable overlap
  - Event extraction from annotations

### Model (`src/model/eeg_inception_erp.py`)

- **EEGInceptionERP**: Implementation based on braindecode
- Features:
  - Multi-scale inception blocks
  - Depthwise separable convolutions
  - Batch normalization and dropout
  - Configurable architecture parameters

### Training Pipeline

- **K-fold Cross-Validation**: 5-fold cross-validation for robust performance evaluation
- **EEGTrainer**: Complete training class with:
  - Early stopping
  - Learning rate scheduling
  - Training history tracking
  - Model checkpointing
  - Cross-validation statistics

## 📊 Model Architecture

The EEGInceptionERP model uses:
- **Input**: Multi-channel EEG signals (16 channels × time points)
- **Inception blocks**: Multiple temporal scales (0.5s, 0.25s, 0.125s)
- **Depthwise convolutions**: Spatial filtering
- **Classification head**: Binary classification (left vs right hand)

## 🎯 Performance

The model achieves:
- Fast training convergence (typically <20 epochs)
- Good generalization with proper regularization
- Real-time inference capability

## 📈 Usage Examples

### Loading Data
```python
from src.data.data_loader import BCIDataLoader

loader = BCIDataLoader(
    data_path="eeg_data",
    subjects=[1, 2, 3],
    runs=[4, 8, 12]
)

windows, labels, _ = loader.load_all_subjects()
```

### Creating Model
```python
from src.model.eeg_inception_erp import EEGInceptionERP

model = EEGInceptionERP(
    n_chans=16,
    n_outputs=2,
    n_times=500,
    sfreq=125
)
```

### Making Predictions
```python
model.eval()
with torch.no_grad():
    output = model(eeg_data)
    predictions = torch.argmax(output, dim=1)
```

### K-fold Training
```python
# Run K-fold cross-validation training
python train_model.py

# This will:
# 1. Split data into train/test sets
# 2. Perform 5-fold CV on training data
# 3. Train final model on all training data
# 4. Evaluate on held-out test set
# 5. Generate comprehensive plots
```

## 🔬 Customization

### Model Parameters
- `n_filters`: Number of initial filters (default: 8)
- `drop_prob`: Dropout probability (default: 0.5)
- `scales_samples_s`: Temporal scales in seconds (default: (0.5, 0.25, 0.125))
- `pooling_sizes`: Pooling sizes (default: (4, 2, 2, 2))

### Data Processing
- `window_length`: Window duration in seconds (default: 4.0)
- `baseline_length`: Baseline period in seconds (default: 1.0)
- `overlap`: Window overlap ratio (default: 0.5)

### Training Parameters
- `K_FOLDS`: Number of cross-validation folds (default: 5)
- `TEST_SPLIT`: Test set proportion (default: 0.2)
- `EARLY_STOPPING_PATIENCE`: Early stopping patience (default: 5)
- `NUM_EPOCHS`: Maximum training epochs (default: 50)

## 📊 Performance Evaluation

The training script provides comprehensive evaluation through:

### K-fold Cross-Validation
- **Robust Performance Estimation**: 5-fold CV provides reliable performance metrics
- **Statistical Analysis**: Mean accuracy ± standard deviation across folds
- **Overfitting Detection**: Comparison of training vs validation performance

### Visualization
- Cross-validation accuracy distribution
- Average learning curves across folds
- Individual fold performance
- Final model training curves
- Comprehensive summary statistics

### Performance Metrics
- Individual fold accuracies
- Mean CV accuracy with confidence intervals
- Final test set accuracy
- Training convergence analysis

## 📝 Notes

- The model uses 'same' padding which may generate warnings on certain PyTorch versions
- For Windows users, set `num_workers=0` in DataLoader
- GPU acceleration is supported if CUDA is available
- Model checkpoints are automatically saved during training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the BSD 3-Clause License.

## 🙏 Acknowledgments

- [Braindecode](https://braindecode.org/) for the EEGInceptionERP architecture
- [PhysioNet](https://physionet.org/) for the EEG motor imagery dataset
- PyTorch team for the deep learning framework

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.
