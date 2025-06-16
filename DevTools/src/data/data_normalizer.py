"""
Universal Data Normalization Module

This module provides universal data normalization for EEG data from any equipment.
It ensures all data is normalized consistently for training, fine-tuning, and inference.

Key Features:
1. Equipment-agnostic normalization
2. Global statistics for consistent normalization across all data
3. Training vs inference modes
4. Statistics persistence for model deployment
5. Automatic data shape handling
"""

import numpy as np
import logging
from typing import Union, Tuple, Optional, Dict, Any, List
import pickle
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalEEGNormalizer:
    """
    Universal EEG Data Normalizer for Training and Inference
    
    This class provides methods to normalize EEG data consistently across:
    - Initial model training (fits global statistics)
    - Fine-tuning (uses global statistics, optionally updates)
    - Inference (uses frozen global statistics)
    """
    
    def __init__(self, 
                 method: str = 'zscore',
                 mode: str = 'training',
                 stats_path: Optional[str] = None):
        """
        Initialize the universal normalizer
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            mode: Operation mode ('training', 'finetuning', 'inference')
            stats_path: Path to save/load global statistics
        """
        self.method = method
        self.mode = mode
        self.stats_path = stats_path
        self.global_stats = {}
        self.is_fitted = False
        self.data_shape_info = {}
        
        # Load existing stats if in finetuning or inference mode
        if self.mode in ['finetuning', 'inference'] and self.stats_path:
            self.load_global_stats(self.stats_path)
        
        logger.info(f"Initialized UniversalEEGNormalizer - Method: {self.method}, Mode: {self.mode}")
    
    def fit_global_stats(self, data_batches: List[np.ndarray]) -> 'UniversalEEGNormalizer':
        """
        Fit global statistics across multiple data batches/subjects
        Used for initial training to establish universal normalization
        
        Args:
            data_batches: List of data arrays from different sources/subjects
        
        Returns:
            self
        """
        if self.mode not in ['training', 'finetuning']:
            logger.warning(f"fit_global_stats called in {self.mode} mode")
        
        logger.info(f"Fitting global statistics across {len(data_batches)} data batches")
        
        # Collect all data for global statistics
        all_data = []
        shape_info = {}
        
        for i, data in enumerate(data_batches):
            data = self._ensure_3d(data)
            all_data.append(data)
            
            # Track shape information
            shape_key = f"batch_{i}"
            shape_info[shape_key] = {
                'original_shape': data.shape,
                'n_samples': data.shape[0],
                'n_channels': data.shape[1],
                'n_timepoints': data.shape[2]
            }
        
        # Concatenate all data for global statistics
        global_data = np.concatenate(all_data, axis=0)
        logger.info(f"Global data shape: {global_data.shape}")
        
        # Fit normalization method
        if self.method == 'zscore':
            self._fit_global_zscore(global_data)
        elif self.method == 'minmax':
            self._fit_global_minmax(global_data)
        elif self.method == 'robust':
            self._fit_global_robust(global_data)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.data_shape_info = shape_info
        self.is_fitted = True
        
        # Save global stats automatically in training mode
        if self.mode == 'training' and self.stats_path:
            self.save_global_stats(self.stats_path)
        
        logger.info("Global statistics fitted successfully")
        return self
    
    def fit(self, data: np.ndarray) -> 'UniversalEEGNormalizer':
        """
        Fit normalizer to single dataset (wrapper for compatibility)
        
        Args:
            data: Training data
            
        Returns:
            self
        """
        return self.fit_global_stats([data])
    
    def transform(self, data: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """
        Transform data using global statistics
        
        Args:
            data: Data to normalize
            update_stats: Whether to update global stats (only in finetuning mode)
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        original_shape = data.shape
        data_3d = self._ensure_3d(data)
        
        # Apply normalization
        if self.method == 'zscore':
            normalized = self._transform_zscore(data_3d)
        elif self.method == 'minmax':
            normalized = self._transform_minmax(data_3d)
        elif self.method == 'robust':
            normalized = self._transform_robust(data_3d)
        
        # Update stats if in finetuning mode and requested
        if update_stats and self.mode == 'finetuning':
            self._update_global_stats(data_3d)
        
        # Restore original shape
        return self._restore_shape(normalized, original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data in one step
        
        Args:
            data: Training data
            
        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)
    
    def _ensure_3d(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in 3D format (n_samples, n_channels, n_timepoints)"""
        if len(data.shape) == 2:
            # Assume (n_samples, n_features) - reshape to (n_samples, n_channels, n_timepoints)
            # This is a heuristic - might need adjustment based on actual data
            n_samples, n_features = data.shape
            if n_features % 16 == 0:  # Assume 16 channels
                n_channels = 16
                n_timepoints = n_features // n_channels
                data = data.reshape(n_samples, n_channels, n_timepoints)
            else:
                # Add channel dimension
                data = data[:, np.newaxis, :]
        elif len(data.shape) == 1:
            # Single sample - add sample and channel dimensions
            data = data[np.newaxis, np.newaxis, :]
            
        return data
    
    def _restore_shape(self, data: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Restore data to original shape"""
        if len(original_shape) == 2:
            # Flatten back to 2D
            return data.reshape(original_shape[0], -1)
        elif len(original_shape) == 1:
            # Return as 1D
            return data.flatten()
        else:
            return data
    
    def _fit_global_zscore(self, data: np.ndarray):
        """Fit global Z-score normalization"""
        # Calculate global statistics across all samples and time, per channel
        self.global_stats['mean'] = np.mean(data, axis=(0, 2), keepdims=True)  # Shape: (1, n_channels, 1)
        self.global_stats['std'] = np.std(data, axis=(0, 2), keepdims=True)
        
        # Avoid division by zero
        self.global_stats['std'] = np.where(self.global_stats['std'] == 0, 1.0, self.global_stats['std'])
        
        logger.info(f"Global Z-score stats:")
        logger.info(f"  Mean range: [{np.min(self.global_stats['mean']):.6f}, {np.max(self.global_stats['mean']):.6f}]")
        logger.info(f"  Std range: [{np.min(self.global_stats['std']):.6f}, {np.max(self.global_stats['std']):.6f}]")
    
    def _fit_global_minmax(self, data: np.ndarray):
        """Fit global Min-max normalization"""
        self.global_stats['min'] = np.min(data, axis=(0, 2), keepdims=True)
        self.global_stats['max'] = np.max(data, axis=(0, 2), keepdims=True)
        
        # Calculate range and avoid division by zero
        range_val = self.global_stats['max'] - self.global_stats['min']
        self.global_stats['range'] = np.where(range_val == 0, 1.0, range_val)
        
        logger.info(f"Global MinMax stats:")
        logger.info(f"  Min range: [{np.min(self.global_stats['min']):.6f}, {np.max(self.global_stats['min']):.6f}]")
        logger.info(f"  Max range: [{np.min(self.global_stats['max']):.6f}, {np.max(self.global_stats['max']):.6f}]")
    
    def _fit_global_robust(self, data: np.ndarray):
        """Fit global Robust scaling"""
        self.global_stats['median'] = np.median(data, axis=(0, 2), keepdims=True)
        self.global_stats['q25'] = np.percentile(data, 25, axis=(0, 2), keepdims=True)
        self.global_stats['q75'] = np.percentile(data, 75, axis=(0, 2), keepdims=True)
        
        # Calculate IQR and avoid division by zero
        iqr = self.global_stats['q75'] - self.global_stats['q25']
        self.global_stats['iqr'] = np.where(iqr == 0, 1.0, iqr)
        
        logger.info(f"Global Robust stats:")
        logger.info(f"  Median range: [{np.min(self.global_stats['median']):.6f}, {np.max(self.global_stats['median']):.6f}]")
        logger.info(f"  IQR range: [{np.min(self.global_stats['iqr']):.6f}, {np.max(self.global_stats['iqr']):.6f}]")
    
    def _transform_zscore(self, data: np.ndarray) -> np.ndarray:
        """Apply global Z-score normalization"""
        return (data - self.global_stats['mean']) / self.global_stats['std']
    
    def _transform_minmax(self, data: np.ndarray) -> np.ndarray:
        """Apply global Min-max normalization"""
        return (data - self.global_stats['min']) / self.global_stats['range']
    
    def _transform_robust(self, data: np.ndarray) -> np.ndarray:
        """Apply global Robust scaling"""
        return (data - self.global_stats['median']) / self.global_stats['iqr']
    
    def _update_global_stats(self, new_data: np.ndarray, alpha: float = 0.1):
        """
        Update global statistics with new data (for fine-tuning)
        
        Args:
            new_data: New data to incorporate
            alpha: Learning rate for exponential moving average
        """
        if self.method == 'zscore':
            new_mean = np.mean(new_data, axis=(0, 2), keepdims=True)
            new_std = np.std(new_data, axis=(0, 2), keepdims=True)
            
            self.global_stats['mean'] = (1 - alpha) * self.global_stats['mean'] + alpha * new_mean
            self.global_stats['std'] = (1 - alpha) * self.global_stats['std'] + alpha * new_std
            self.global_stats['std'] = np.where(self.global_stats['std'] == 0, 1.0, self.global_stats['std'])
            
        # Similar updates for other methods...
        logger.info(f"Updated global statistics with alpha={alpha}")
    
    def save_global_stats(self, filepath: str):
        """Save global normalization statistics"""
        stats_data = {
            'method': self.method,
            'mode': self.mode,
            'global_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in self.global_stats.items()},
            'is_fitted': self.is_fitted,
            'data_shape_info': self.data_shape_info,
            'version': '2.0'  # Version for compatibility
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for human readability
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # Also save binary version for numpy arrays
        binary_path = filepath.replace('.json', '.pkl')
        with open(binary_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'global_stats': self.global_stats,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Global normalization stats saved to: {filepath}")
    
    def load_global_stats(self, filepath: str):
        """Load global normalization statistics"""
        # Try JSON first, then pickle
        if filepath.endswith('.json') and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                stats_data = json.load(f)
            
            # Convert lists back to numpy arrays
            self.global_stats = {k: np.array(v) if isinstance(v, list) else v 
                               for k, v in stats_data['global_stats'].items()}
            
        else:
            # Try pickle version
            pickle_path = filepath.replace('.json', '.pkl')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    stats_data = pickle.load(f)
                self.global_stats = stats_data['global_stats']
            else:
                raise FileNotFoundError(f"Stats file not found: {filepath}")
        
        self.method = stats_data['method']
        self.is_fitted = stats_data['is_fitted']
        
        logger.info(f"Global normalization stats loaded from: {filepath}")
        logger.info(f"Loaded method: {self.method}, fitted: {self.is_fitted}")


# Factory function for universal use
def create_universal_normalizer(method: str = 'zscore', mode: str = 'training', stats_path: Optional[str] = None) -> UniversalEEGNormalizer:
    return UniversalEEGNormalizer(method=method, mode=mode, stats_path=stats_path)
