"""
BCI Data Loader for PhysioNet Motor Imagery Dataset

This module provides data loading, preprocessing and windowing capabilities
for the PhysioNet motor imagery dataset (runs 4, 8, 12) suitable for CNN training.

Classes:
    BCIDataLoader: Main data loader class for loading and preprocessing EEG data
    BCIDataset: PyTorch dataset class for windowed EEG data
    
Functions:
    load_csv_data: Load EEG data from CSV files
    extract_events: Extract event markers and timing
    create_windows: Create time windows for CNN training
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCIDataLoader:
    """
    Data loader for PhysioNet Motor Imagery Dataset
    
    Handles loading CSV files, extracting events, and creating windowed data
    suitable for CNN training on left/right hand motor imagery tasks.
    """
    
    def __init__(self, 
                 data_path: str,
                 subjects: Optional[List[int]] = None,
                 runs: Optional[List[int]] = None,
                 sample_rate: int = 125,
                 n_channels: int = 16):
        """
        Initialize the BCI data loader
        
        Args:
            data_path: Path to the EEG data directory
            subjects: List of subject IDs to load (default: all available)
            runs: List of runs to load (default: [4, 8, 12])
            sample_rate: Sampling rate in Hz (default: 125)
            n_channels: Number of EEG channels (default: 16)
        """
        self.data_path = data_path
        self.subjects = subjects
        self.runs = runs if runs is not None else [4, 8, 12]
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        
        # Event mapping for PhysioNet motor imagery
        self.event_mapping = {
            'T0': 0,  # Rest/baseline
            'T1': 1,  # Left hand imagery  
            'T2': 2   # Right hand imagery
        }
        
        # EEG channel names (standard 10-20 system for first 16 channels)
        self.channel_names = [
            'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
            'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7',
            'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11',
            'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15'
        ]
        
        logger.info(f"Initialized BCIDataLoader for subjects: {self.subjects}, runs: {self.runs}")
    
    def get_available_subjects(self) -> List[int]:
        """Get list of available subject IDs in the dataset"""
        subjects = []
        base_path = os.path.join(self.data_path, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
        
        if not os.path.exists(base_path):
            logger.error(f"Data path not found: {base_path}")
            return subjects
            
        for item in os.listdir(base_path):
            if item.startswith('S') and item[1:].isdigit():
                subject_id = int(item[1:])
                subjects.append(subject_id)
                
        return sorted(subjects)
    
    def load_csv_data(self, subject_id: int, run: int) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        """
        Load EEG data from CSV file for a specific subject and run
        
        Args:
            subject_id: Subject ID (e.g., 1 for S001)
            run: Run number (4, 8, or 12)
            
        Returns:
            Tuple of (eeg_data, events) where:
            - eeg_data: numpy array of shape (n_samples, n_channels)
            - events: list of (sample_index, event_label) tuples
        """
        # Construct file path
        subject_str = f"S{subject_id:03d}"
        filename = f"{subject_str}R{run:02d}_csv_openbci.csv"
        file_path = os.path.join(
            self.data_path, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0", 
            subject_str, filename
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        # Read CSV file, skip header comments
        df = pd.read_csv(file_path, comment='%')
        
        # Extract EEG channels (first 16 columns after Sample Index)
        eeg_data = df[self.channel_names].values.astype(np.float32)
        
        # Extract events from annotations column
        events = []
        annotations = df['Annotations'].fillna('')
        
        for idx, annotation in enumerate(annotations):
            if annotation in self.event_mapping:
                events.append((idx, annotation))
        
        logger.info(f"Loaded {eeg_data.shape[0]} samples, {len(events)} events")
        return eeg_data, events
    
    def preprocess_data(self, 
                       eeg_data: np.ndarray, 
                       lowcut: float = 0.5, 
                       highcut: float = 50.0,
                       notch_freq: float = 50.0,
                       apply_standardization: bool = True) -> np.ndarray:
        """
        Preprocess EEG data with filtering and standardization
        
        Args:
            eeg_data: EEG data array of shape (n_samples, n_channels)
            lowcut: Low cutoff frequency for bandpass filter
            highcut: High cutoff frequency for bandpass filter  
            notch_freq: Notch filter frequency (power line noise)
            apply_standardization: Whether to apply z-score standardization
            
        Returns:
            Preprocessed EEG data
        """
        logger.info("Applying preprocessing...")
        
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low_norm = lowcut / nyquist
        high_norm = highcut / nyquist
        
        # Bandpass filter
        b_band, a_band = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Notch filter for power line noise
        b_notch, a_notch = signal.iirnotch(notch_freq, 30, self.sample_rate)
        
        # Apply filters to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            # Bandpass filter
            filtered_ch = signal.filtfilt(b_band, a_band, eeg_data[:, ch])
            # Notch filter
            filtered_ch = signal.filtfilt(b_notch, a_notch, filtered_ch)
            filtered_data[:, ch] = filtered_ch
        
        # Standardization (z-score normalization)
        if apply_standardization:
            scaler = StandardScaler()
            filtered_data = scaler.fit_transform(filtered_data)
        
        logger.info("Preprocessing completed")
        return filtered_data.astype(np.float32)
    
    def create_windows(self, 
                      eeg_data: np.ndarray, 
                      events: List[Tuple[int, str]],
                      window_length: float = 4.0,
                      baseline_length: float = 1.0,
                      overlap: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time windows around events for CNN training
        
        Args:
            eeg_data: Preprocessed EEG data of shape (n_samples, n_channels)
            events: List of (sample_index, event_label) tuples
            window_length: Length of each window in seconds
            baseline_length: Baseline period before event onset in seconds  
            overlap: Overlap between windows (0.0 = no overlap, 0.5 = 50% overlap)
            
        Returns:
            Tuple of (windows, labels) where:
            - windows: array of shape (n_windows, n_channels, n_timepoints)
            - labels: array of shape (n_windows,) with class labels
        """
        logger.info("Creating time windows...")
        
        # Convert time to samples
        window_samples = int(window_length * self.sample_rate)
        baseline_samples = int(baseline_length * self.sample_rate)
        
        windows = []
        labels = []
        
        for event_sample, event_label in events:
            # Skip rest events for motor imagery classification
            if event_label == 'T0':
                continue
                
            # Calculate window bounds
            start_sample = event_sample - baseline_samples
            end_sample = start_sample + window_samples
            
            # Check bounds
            if start_sample < 0 or end_sample >= eeg_data.shape[0]:
                logger.warning(f"Skipping event at sample {event_sample} - out of bounds")
                continue
            
            # Extract window and transpose to (channels, time)
            window = eeg_data[start_sample:end_sample, :].T
            windows.append(window)
            
            # Convert label to class index (T1=0 for left, T2=1 for right)
            label = 0 if event_label == 'T1' else 1
            labels.append(label)
        
        if len(windows) == 0:
            logger.warning("No valid windows created!")
            return np.array([]), np.array([])
        
        windows = np.array(windows, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        logger.info(f"Created {len(windows)} windows of shape {windows.shape[1:]} with labels: {np.bincount(labels)}")
        return windows, labels
    
    def load_subject_data(self, 
                         subject_id: int,
                         preprocess: bool = True,
                         create_windows_flag: bool = True,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process all data for a single subject across specified runs
        
        Args:
            subject_id: Subject ID to load
            preprocess: Whether to apply preprocessing
            create_windows_flag: Whether to create time windows
            **kwargs: Additional arguments for preprocessing and windowing
            
        Returns:
            Tuple of (windows, labels) for the subject
        """
        all_windows = []
        all_labels = []
        
        for run in self.runs:
            try:
                # Load raw data
                eeg_data, events = self.load_csv_data(subject_id, run)
                
                # Preprocess if requested
                if preprocess:
                    eeg_data = self.preprocess_data(eeg_data, **kwargs)
                
                # Create windows if requested
                if create_windows_flag:
                    windows, labels = self.create_windows(eeg_data, events, **kwargs)
                    if len(windows) > 0:
                        all_windows.append(windows)
                        all_labels.append(labels)
                        
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}, run {run}: {e}")
                continue
        
        if len(all_windows) == 0:
            logger.warning(f"No data loaded for subject {subject_id}")
            return np.array([]), np.array([])
        
        # Concatenate all runs
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Subject {subject_id}: {all_windows.shape[0]} total windows")
        return all_windows, all_labels
    
    def load_all_subjects(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for all specified subjects
        
        Args:
            **kwargs: Arguments passed to load_subject_data
            
        Returns:
            Tuple of (windows, labels, subject_ids) where subject_ids indicates
            which subject each window belongs to
        """
        if self.subjects is None:
            self.subjects = self.get_available_subjects()
        
        all_windows = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id in self.subjects:
            windows, labels = self.load_subject_data(subject_id, **kwargs)
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(labels)
                all_subject_ids.append(np.full(len(windows), subject_id))
        
        if len(all_windows) == 0:
            logger.error("No data loaded for any subjects!")
            return np.array([]), np.array([]), np.array([])
        
        # Concatenate all subjects
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.concatenate(all_subject_ids, axis=0)
        
        logger.info(f"Loaded {all_windows.shape[0]} total windows from {len(self.subjects)} subjects")
        logger.info(f"Class distribution: {np.bincount(all_labels)}")
        
        return all_windows, all_labels, all_subject_ids


class BCIDataset(Dataset):
    """
    PyTorch Dataset class for BCI EEG data
    
    Handles windowed EEG data for CNN training with optional data augmentation
    """
    
    def __init__(self, 
                 windows: np.ndarray, 
                 labels: np.ndarray,
                 transform: Optional[callable] = None,
                 augment: bool = False):
        """
        Initialize BCI dataset
        
        Args:
            windows: EEG windows of shape (n_windows, n_channels, n_timepoints)
            labels: Labels of shape (n_windows,)
            transform: Optional transform to apply to windows
            augment: Whether to apply data augmentation
        """
        assert len(windows) == len(labels), "Windows and labels must have same length"
        assert len(windows.shape) == 3, "Windows must have shape (n_windows, n_channels, n_timepoints)"
        
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        self.augment = augment
        
        logger.info(f"Created dataset with {len(self)} samples")
        logger.info(f"Window shape: {self.windows.shape[1:]}")
        logger.info(f"Class distribution: {torch.bincount(self.labels)}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset"""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (window, label)
        """
        window = self.windows[idx]
        label = self.labels[idx]
        
        # Apply data augmentation if enabled
        if self.augment:
            window = self._augment_window(window)
        
        # Apply transform if provided
        if self.transform:
            window = self.transform(window)
        
        return window, label
    
    def _augment_window(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to EEG window
        
        Args:
            window: EEG window tensor of shape (n_channels, n_timepoints)
            
        Returns:
            Augmented window
        """
        # Add small amount of noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(window) * 0.01
            window = window + noise
        
        # Time shifting
        if torch.rand(1) < 0.3:
            shift = torch.randint(-10, 11, (1,)).item()
            if shift != 0:
                window = torch.roll(window, shift, dims=1)
        
        # Amplitude scaling
        if torch.rand(1) < 0.3:
            scale = torch.uniform(0.9, 1.1, (1,)).item()
            window = window * scale
        
        return window
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = torch.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights


def create_data_loaders(data_path: str,
                       subjects: Optional[List[int]] = None,
                       batch_size: int = 32,
                       validation_split: float = 0.2,
                       test_split: float = 0.1,
                       num_workers: int = 4,
                       **loader_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders
    
    Args:
        data_path: Path to EEG data directory
        subjects: List of subject IDs to include
        batch_size: Batch size for data loaders
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of worker processes for data loading
        **loader_kwargs: Additional arguments for BCIDataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Initialize data loader
    bci_loader = BCIDataLoader(data_path, subjects=subjects, **loader_kwargs)
    
    # Load all data
    windows, labels, subject_ids = bci_loader.load_all_subjects()
    
    if len(windows) == 0:
        raise ValueError("No data loaded!")
    
    # Split data by subjects to avoid data leakage
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    # Calculate splits
    n_test = max(1, int(n_subjects * test_split))
    n_val = max(1, int(n_subjects * validation_split))
    n_train = n_subjects - n_test - n_val
    
    if n_train <= 0:
        raise ValueError("Not enough subjects for train/val/test split")
    
    # Randomly assign subjects to splits
    np.random.shuffle(unique_subjects)
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]
    
    # Create masks for data splits
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Create datasets
    train_dataset = BCIDataset(windows[train_mask], labels[train_mask], augment=True)
    val_dataset = BCIDataset(windows[val_mask], labels[val_mask], augment=False)
    test_dataset = BCIDataset(windows[test_mask], labels[test_mask], augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples from subjects {train_subjects}")
    logger.info(f"  Validation: {len(val_dataset)} samples from subjects {val_subjects}")
    logger.info(f"  Test: {len(test_dataset)} samples from subjects {test_subjects}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_path = r"c:\Users\Chari\OneDrive\Documentos\GitHub\projetoBCI\eeg_data"
    
    # Test data loading
    loader = BCIDataLoader(data_path, subjects=[1, 2, 3])
    
    # Load data for one subject
    windows, labels = loader.load_subject_data(1)
    print(f"Loaded {len(windows)} windows with shape {windows.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path, 
        subjects=list(range(1, 11))  # First 10 subjects
    )
    
    # Test a batch
    for batch_windows, batch_labels in train_loader:
        print(f"Batch shape: {batch_windows.shape}")
        print(f"Labels: {batch_labels}")
        break
