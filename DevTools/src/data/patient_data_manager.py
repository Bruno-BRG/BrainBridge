"""
Class:   PatientDataManager
Purpose: Manages patient-specific EEG data for fine-tuning and validation.
Author:  Bruno Rocha
Created: 2025-06-01
License: BSD (3-clause)
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Task 1.2: Patient Data Management
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
import logging
from sklearn.model_selection import train_test_split
from src.data.data_loader import BCIDataLoader, BCIDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientDataManager:
    """
    Manages patient-specific EEG data for fine-tuning and validation.
    
    This class provides functionality to load, organize, and prepare patient
    recordings for fine-tuning pre-trained models.
    
    Features:
    - Multiple recording session management per patient
    - Automatic train/validation splitting
    - Data quality assessment
    - Session metadata management
    - Patient data organization
    
    Args:
        patient_id (str): Unique identifier for the patient
        data_root (str, optional): Root directory for patient data storage
        verbose (bool, optional): Enable verbose logging. Defaults to True.
    """
    
    def __init__(self, 
                 patient_id: str,
                 data_root: str = "patient_data",
                 verbose: bool = True):
        """
        Initialize PatientDataManager for a specific patient.
        
        Args:
            patient_id: Unique identifier for the patient
            data_root: Root directory for patient data storage
            verbose: Enable verbose logging
        """
        self.patient_id = patient_id
        self.data_root = data_root
        self.verbose = verbose
        
        # Create patient data directory structure
        self.patient_dir = os.path.join(data_root, f"patient_{patient_id}")
        self.recordings_dir = os.path.join(self.patient_dir, "recordings")
        self.metadata_dir = os.path.join(self.patient_dir, "metadata")
        self.processed_dir = os.path.join(self.patient_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize session tracking
        self.sessions = {}
        self.metadata = self._load_patient_metadata()
        
        if self.verbose:
            print(f"PatientDataManager initialized for patient {patient_id}")
            print(f"Data directory: {self.patient_dir}")
    
    def _load_patient_metadata(self) -> Dict:
        """Load patient metadata from file or create new metadata."""
        metadata_file = os.path.join(self.metadata_dir, "patient_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            if self.verbose:
                print(f"Loaded existing metadata for patient {self.patient_id}")
        else:
            metadata = {
                'patient_id': self.patient_id,
                'created_date': datetime.now().isoformat(),
                'sessions': {},
                'data_quality': {},
                'preprocessing_params': {}
            }
            self._save_patient_metadata(metadata)
            if self.verbose:
                print(f"Created new metadata for patient {self.patient_id}")
        
        return metadata
    
    def _save_patient_metadata(self, metadata: Dict) -> None:
        """Save patient metadata to file."""
        metadata_file = os.path.join(self.metadata_dir, "patient_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_recording_session(self,
                            session_id: str,
                            recording_files: List[str],
                            session_metadata: Optional[Dict] = None) -> bool:
        """
        Add a new recording session for the patient.
        
        Args:
            session_id: Unique identifier for the recording session
            recording_files: List of paths to CSV recording files
            session_metadata: Optional metadata for the session
            
        Returns:
            bool: True if session added successfully
        """
        try:
            if session_metadata is None:
                session_metadata = {}
            
            # Validate recording files exist
            for file_path in recording_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Recording file not found: {file_path}")
            
            # Create session directory
            session_dir = os.path.join(self.recordings_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Copy or link recording files to session directory
            session_files = []
            for i, file_path in enumerate(recording_files):
                filename = f"{session_id}_recording_{i+1}.csv"
                dest_path = os.path.join(session_dir, filename)
                
                # For now, just store the original path reference
                # In production, you might want to copy the files
                session_files.append(file_path)
            
            # Store session information
            session_info = {
                'session_id': session_id,
                'recording_files': session_files,
                'created_date': datetime.now().isoformat(),
                'metadata': session_metadata,
                'processed': False,
                'quality_checked': False
            }
            
            self.sessions[session_id] = session_info
            self.metadata['sessions'][session_id] = session_info
            self._save_patient_metadata(self.metadata)
            
            if self.verbose:
                print(f"Added recording session '{session_id}' with {len(recording_files)} files")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding recording session: {e}")
            return False
    
    def load_patient_recordings(self,
                              session_ids: Optional[List[str]] = None,
                              preprocessing_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and concatenate EEG recordings from specified sessions.
        
        Args:
            session_ids: List of session IDs to load. If None, loads all sessions.
            preprocessing_params: Parameters for data preprocessing
            
        Returns:
            Tuple of (data, labels) arrays
        """
        if session_ids is None:
            session_ids = list(self.sessions.keys())
        
        if not session_ids:
            raise ValueError("No recording sessions available")
        
        all_data = []
        all_labels = []
        
        # Default preprocessing parameters
        default_params = {
            'sample_rate': 125,
            'n_channels': 16,
            'window_length': 3.2,
            'baseline_length': 1.0,
            'overlap': 0.5,
            'lowcut': 0.5,
            'highcut': 50.0,
            'notch_freq': 50.0
        }
        
        if preprocessing_params is not None:
            default_params.update(preprocessing_params)
        
        for session_id in session_ids:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found, skipping")
                continue
            
            session_info = self.sessions[session_id]
            recording_files = session_info['recording_files']
            
            if self.verbose:
                print(f"Loading session {session_id} with {len(recording_files)} files")              # Load data using direct CSV loading (patient files are not in PhysioNet structure)
            for file_path in recording_files:
                try:
                    # Parse filename to extract subject_id and run
                    # Expected format: S001R04_csv_openbci.csv
                    filename = os.path.basename(file_path)
                    
                    # Extract subject and run from filename
                    if 'S' in filename and 'R' in filename:
                        # Extract subject_id (e.g., S001 -> 1)
                        subject_part = filename.split('R')[0]
                        subject_id = int(subject_part[1:])  # Remove 'S' and convert to int
                        
                        # Extract run (e.g., R04 -> 4)
                        run_part = filename.split('R')[1].split('_')[0]
                        run = int(run_part)
                    else:
                        # Fallback for non-standard filenames
                        logger.warning(f"Cannot parse filename {filename}, using defaults")
                        subject_id = 1
                        run = 4
                    
                    # Load CSV directly since patient files are not in PhysioNet directory structure
                    eeg_data, events = self._load_patient_csv_direct(
                        file_path, 
                        default_params['sample_rate'],
                        default_params['n_channels']
                    )
                      # Create windows from the loaded data
                    if len(eeg_data) > 0 and len(events) > 0:
                        windows, labels = self._create_windows_from_data(
                            eeg_data, events,
                            window_length=default_params['window_length'],
                            baseline_length=default_params['baseline_length'],
                            overlap=default_params['overlap'],
                            sample_rate=default_params['sample_rate']
                        )
                        
                        if len(windows) > 0:
                            all_data.append(windows)
                            all_labels.append(labels)
                            
                            if self.verbose:
                                print(f"  - Loaded {len(windows)} windows from {filename}")
                        else:
                            logger.warning(f"No windows created from {filename}")
                    else:
                        logger.warning(f"No data or events loaded from {filename}")
                        
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data could be loaded from any session")
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        if self.verbose:
            print(f"Total loaded data: {combined_data.shape[0]} samples")
            print(f"Data shape: {combined_data.shape}")
            print(f"Label distribution: {np.bincount(combined_labels)}")
        
        return combined_data, combined_labels
    
    def split_training_validation(self,
                                data: np.ndarray,
                                labels: np.ndarray,
                                validation_split: float = 0.2,
                                random_state: int = 42,
                                stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split patient data into training and validation sets.
        
        Args:
            data: EEG data array
            labels: Label array
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducible splits
            stratify: Whether to stratify split by labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if self.verbose:
            print(f"Splitting data: {len(data)} samples, {validation_split*100}% for validation")
        
        stratify_labels = labels if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels,
            test_size=validation_split,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        if self.verbose:
            print(f"Training set: {len(X_train)} samples")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Training label distribution: {np.bincount(y_train)}")
            print(f"Validation label distribution: {np.bincount(y_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_fine_tuning_datasets(self,
                                   session_ids: Optional[List[str]] = None,
                                   validation_split: float = 0.2,
                                   batch_size: int = 32,
                                   preprocessing_params: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare PyTorch DataLoaders for fine-tuning.
        
        Args:
            session_ids: List of session IDs to include
            validation_split: Fraction for validation split
            batch_size: Batch size for DataLoaders
            preprocessing_params: Preprocessing parameters
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load patient recordings
        data, labels = self.load_patient_recordings(session_ids, preprocessing_params)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = self.split_training_validation(
            data, labels, validation_split
        )
        
        # Create PyTorch datasets
        train_dataset = BCIDataset(
            data=torch.FloatTensor(X_train),
            labels=torch.LongTensor(y_train)
        )
        
        val_dataset = BCIDataset(
            data=torch.FloatTensor(X_val),
            labels=torch.LongTensor(y_val)
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        if self.verbose:
            print(f"Created DataLoaders:")
            print(f"  - Training: {len(train_loader)} batches of size {batch_size}")
            print(f"  - Validation: {len(val_loader)} batches of size {batch_size}")
        
        return train_loader, val_loader
    
    def calculate_data_quality_metrics(self,
                                     session_ids: Optional[List[str]] = None) -> Dict:
        """
        Calculate data quality metrics for patient recordings.
        
        Args:
            session_ids: List of session IDs to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if session_ids is None:
            session_ids = list(self.sessions.keys())
        
        quality_metrics = {
            'session_metrics': {},
            'overall_metrics': {}
        }
        
        all_signal_quality = []
        all_artifact_ratios = []
        total_samples = 0
        
        for session_id in session_ids:
            if session_id not in self.sessions:
                continue
            
            try:
                # Load session data
                session_data, session_labels = self.load_patient_recordings([session_id])
                
                # Calculate signal quality metrics
                signal_variance = np.var(session_data, axis=-1)  # Variance across time
                signal_mean = np.mean(signal_variance, axis=1)   # Mean across channels
                
                # Artifact detection (simple threshold-based)
                artifact_threshold = np.percentile(signal_mean, 95)
                artifact_ratio = np.mean(signal_mean > artifact_threshold)
                
                # Signal-to-noise ratio estimation
                signal_power = np.mean(signal_variance)
                noise_power = np.var(signal_mean)
                snr_estimate = signal_power / (noise_power + 1e-8)
                
                session_metrics = {
                    'total_samples': len(session_data),
                    'signal_quality_score': float(np.mean(signal_mean)),
                    'artifact_ratio': float(artifact_ratio),
                    'snr_estimate': float(snr_estimate),
                    'channel_quality': signal_mean.tolist(),
                    'label_distribution': np.bincount(session_labels).tolist()
                }
                
                quality_metrics['session_metrics'][session_id] = session_metrics
                
                # Aggregate for overall metrics
                all_signal_quality.extend(signal_mean)
                all_artifact_ratios.append(artifact_ratio)
                total_samples += len(session_data)
                
                if self.verbose:
                    print(f"Session {session_id} quality:")
                    print(f"  - Samples: {len(session_data)}")
                    print(f"  - Signal quality: {np.mean(signal_mean):.3f}")
                    print(f"  - Artifact ratio: {artifact_ratio:.3f}")
                    print(f"  - SNR estimate: {snr_estimate:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating quality for session {session_id}: {e}")
                continue
        
        # Calculate overall metrics
        if all_signal_quality:
            quality_metrics['overall_metrics'] = {
                'total_samples': total_samples,
                'average_signal_quality': float(np.mean(all_signal_quality)),
                'signal_quality_std': float(np.std(all_signal_quality)),
                'average_artifact_ratio': float(np.mean(all_artifact_ratios)),
                'data_completeness': len(session_ids) / len(self.sessions) if self.sessions else 0,
                'quality_score': self._calculate_overall_quality_score(
                    np.mean(all_signal_quality),
                    np.mean(all_artifact_ratios),
                    total_samples
                )
            }
        
        # Save quality metrics
        self.metadata['data_quality'] = quality_metrics
        self._save_patient_metadata(self.metadata)
        
        return quality_metrics
    
    def _calculate_overall_quality_score(self,
                                       signal_quality: float,
                                       artifact_ratio: float,
                                       total_samples: int) -> float:
        """
        Calculate an overall quality score for the patient data.
        
        Args:
            signal_quality: Average signal quality
            artifact_ratio: Ratio of artifacts
            total_samples: Total number of samples
            
        Returns:
            Quality score between 0 and 1
        """
        # Normalize signal quality (assuming reasonable range)
        signal_score = min(1.0, signal_quality / 100.0)
        
        # Artifact penalty (lower artifacts = higher score)
        artifact_score = max(0.0, 1.0 - artifact_ratio * 2.0)
        
        # Sample count bonus (more data = higher score, up to a point)
        sample_score = min(1.0, total_samples / 1000.0)
        
        # Weighted combination
        overall_score = (signal_score * 0.5 + artifact_score * 0.3 + sample_score * 0.2)
        
        return float(overall_score)
    
    def get_session_summary(self) -> Dict:
        """
        Get a summary of all recording sessions for this patient.
        
        Returns:
            Dictionary with session summary information
        """
        summary = {
            'patient_id': self.patient_id,
            'total_sessions': len(self.sessions),
            'sessions': []
        }
        
        for session_id, session_info in self.sessions.items():
            session_summary = {
                'session_id': session_id,
                'recording_files': len(session_info['recording_files']),
                'created_date': session_info['created_date'],
                'processed': session_info.get('processed', False),
                'quality_checked': session_info.get('quality_checked', False)
            }
            summary['sessions'].append(session_summary)
        
        return summary
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove a recording session from patient data.
        
        Args:
            session_id: ID of session to remove
            
        Returns:
            bool: True if session removed successfully
        """
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Remove from memory
            del self.sessions[session_id]
            
            # Remove from metadata
            if session_id in self.metadata['sessions']:
                del self.metadata['sessions'][session_id]
            
            # Save updated metadata
            self._save_patient_metadata(self.metadata)
            
            # Remove session directory if it exists
            session_dir = os.path.join(self.recordings_dir, session_id)
            if os.path.exists(session_dir):
                import shutil
                shutil.rmtree(session_dir)
            
            if self.verbose:
                print(f"Removed session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing session {session_id}: {e}")
            return False
    
    def _load_patient_csv_direct(self, file_path: str, sample_rate: int = 125, n_channels: int = 16) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        """
        Load EEG data directly from patient CSV file without using BCIDataLoader's path structure.
        
        Args:
            file_path: Direct path to the CSV file
            sample_rate: Sampling rate in Hz
            n_channels: Number of EEG channels
            
        Returns:
            Tuple of (eeg_data, events)
        """
        import pandas as pd
        
        # Event mapping for motor imagery
        event_mapping = {
            'T0': 0,  # Rest/baseline
            'T1': 1,  # Left hand imagery  
            'T2': 2   # Right hand imagery
        }
        
        # EEG channel names
        channel_names = [
            'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
            'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7',
            'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11',
            'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15'
        ]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.verbose:
            print(f"Loading data from: {file_path}")
        
        # Read CSV file, skip header comments
        df = pd.read_csv(file_path, comment='%')
        
        # Extract EEG channels (first 16 columns after Sample Index)
        eeg_data = df[channel_names].values.astype(np.float32)
        
        # Extract events from annotations column
        events = []
        annotations = df['Annotations'].fillna('')
        
        for idx, annotation in enumerate(annotations):
            if annotation in event_mapping:
                events.append((idx, annotation))
        
        if self.verbose:
            print(f"Loaded {eeg_data.shape[0]} samples, {len(events)} events")
        
        return eeg_data, events
    
    def _create_windows_from_data(self, 
                                 eeg_data: np.ndarray, 
                                 events: List[Tuple[int, str]],
                                 window_length: float = 3.2,
                                 baseline_length: float = 1.0,
                                 overlap: float = 0.0,
                                 sample_rate: int = 125) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time windows around events for CNN training.
        
        Args:
            eeg_data: Preprocessed EEG data of shape (n_samples, n_channels)
            events: List of (sample_index, event_label) tuples
            window_length: Length of each window in seconds
            baseline_length: Baseline period before event onset in seconds  
            overlap: Overlap between windows (not used in current implementation)
            sample_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (windows, labels) where:
            - windows: array of shape (n_windows, n_channels, n_timepoints)
            - labels: array of shape (n_windows,) with class labels
        """
        if self.verbose:
            print("Creating time windows...")
        
        # Convert time to samples
        window_samples = int(window_length * sample_rate)
        baseline_samples = int(baseline_length * sample_rate)
        
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
                if self.verbose:
                    print(f"Skipping event at sample {event_sample} - out of bounds")
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
        
        if self.verbose:
            print(f"Created {len(windows)} windows of shape {windows.shape[1:]} with labels: {np.bincount(labels)}")
        
        return windows, labels

    def prepare_patient_data(self, patient_data_path: str,
                         validation_split: float = 0.2,
                         preprocessing_params: Optional[Dict[str, any]] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, any]]:
        """
        Prepare patient data for fine-tuning: load recordings, split, and create DataLoaders.
        
        Args:
            patient_data_path: Path to the patient data directory
            validation_split: Fraction of data to use for validation
            preprocessing_params: Optional preprocessing parameters
            
        Returns:
            Tuple of (train_loader, val_loader, metadata)
        """
        from src.data.data_normalizer import create_universal_normalizer
        normalizer = create_universal_normalizer(method='zscore', mode='training', stats_path="global_stats.json")
        
        # Load patient recordings
        all_data, all_labels = self.load_patient_recordings(None, preprocessing_params)
        
        # Split data (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            all_data, all_labels, 
            test_size=validation_split, 
            stratify=all_labels, 
            random_state=42
        )
        
        combined = np.concatenate([X_train, X_val])
        combined = normalizer.fit_transform(combined)
        
        # Debug: Save normalized fine tuning data as CSV for inspection
        import os, pandas as pd
        debug_folder = os.path.join("debug", "normalized_finetuning")
        os.makedirs(debug_folder, exist_ok=True)
        combined_flat = combined.reshape(combined.shape[0], -1)
        pd.DataFrame(combined_flat).to_csv(os.path.join(debug_folder, "normalized_combined_finetuning.csv"), index=False)
        
        train_length = len(X_train)
        X_train_norm = combined[:train_length]
        X_val_norm = combined[train_length:]
        
        # Create PyTorch datasets
        train_dataset = BCIDataset(
            data=torch.FloatTensor(X_train_norm),
            labels=torch.LongTensor(y_train)
        )
        
        val_dataset = BCIDataset(
            data=torch.FloatTensor(X_val_norm),
            labels=torch.LongTensor(y_val)
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )
        
        # Metadata summary
        metadata = {
            'total_samples': len(all_data),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_channels': preprocessing_params.get('n_channels', 16),
            'sample_rate': preprocessing_params.get('sample_rate', 125),
            'window_length': preprocessing_params.get('window_length', 3.2),
            'baseline_length': preprocessing_params.get('baseline_length', 1.0),
            'overlap': preprocessing_params.get('overlap', 0.5),
            'label_distribution': {
                'train': np.bincount(y_train).tolist(),
                'val': np.bincount(y_val).tolist()
            }
        }
        
        if self.verbose:
            print(f"Prepared patient data: {metadata['total_samples']} total samples")
            print(f"  - Training: {metadata['train_samples']} samples")
            print(f"  - Validation: {metadata['val_samples']} samples")
            print(f"  - Channels: {metadata['n_channels']}")
            print(f"  - Sample rate: {metadata['sample_rate']} Hz")
            print(f"  - Window length: {metadata['window_length']} s")
            print(f"  - Baseline length: {metadata['baseline_length']} s")
            print(f"  - Overlap: {metadata['overlap']}")
            print(f"  - Training label distribution: {metadata['label_distribution']['train']}")
            print(f"  - Validation label distribution: {metadata['label_distribution']['val']}")
        
        return train_loader, val_loader, metadata
