"""
File:    EEGFilter.py
Class:   EEGFilter
Purpose: Provides filtering capabilities for EEG data including bandpass filtering,
         notch filtering, and other signal processing utilities.
Author:  Bruno Rocha
Created: 2025-05-29
Notes:   Used by the real-time inference system for preprocessing EEG signals.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class EEGFilter:
    """
    A class for filtering EEG signals with various filter types.
    
    Provides methods for bandpass filtering, notch filtering, and other
    signal processing operations commonly used in EEG analysis.
    """
    
    def __init__(self, sample_rate: float = 125.0):
        """
        Initialize the EEG filter.
        
        Args:
            sample_rate: The sampling rate of the EEG data in Hz
        """
        self.sample_rate = sample_rate
    
    def bandpass_filter(self, data: np.ndarray, l_freq: float, h_freq: float, 
                       order: int = 4) -> np.ndarray:
        """
        Apply a bandpass filter to the EEG data.
        
        Args:
            data: Input EEG data (channels x samples)
            l_freq: Low frequency cutoff in Hz
            h_freq: High frequency cutoff in Hz
            order: Filter order (default: 4)
            
        Returns:
            Filtered EEG data with the same shape as input
        """
        nyquist = self.sample_rate / 2.0
        low = l_freq / nyquist
        high = h_freq / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        if low >= high:
            raise ValueError(f"Low frequency ({l_freq}) must be less than high frequency ({h_freq})")
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter
        if data.ndim == 1:
            # Single channel
            filtered_data = signal.filtfilt(b, a, data)
        else:
            # Multiple channels - filter each channel separately
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, freq: float = 50.0, 
                    quality: float = 30.0) -> np.ndarray:
        """
        Apply a notch filter to remove specific frequency (e.g., powerline noise).
        
        Args:
            data: Input EEG data (channels x samples)
            freq: Frequency to remove in Hz (default: 50Hz for EU powerline)
            quality: Quality factor of the notch filter
            
        Returns:
            Filtered EEG data with the same shape as input
        """
        nyquist = self.sample_rate / 2.0
        freq_norm = freq / nyquist
        
        # Design notch filter
        b, a = signal.iirnotch(freq_norm, quality)
        
        # Apply filter
        if data.ndim == 1:
            # Single channel
            filtered_data = signal.filtfilt(b, a, data)
        else:
            # Multiple channels - filter each channel separately
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered_data
    
    def highpass_filter(self, data: np.ndarray, cutoff: float, 
                       order: int = 4) -> np.ndarray:
        """
        Apply a highpass filter to the EEG data.
        
        Args:
            data: Input EEG data (channels x samples)
            cutoff: Cutoff frequency in Hz
            order: Filter order (default: 4)
            
        Returns:
            Filtered EEG data with the same shape as input
        """
        nyquist = self.sample_rate / 2.0
        cutoff_norm = cutoff / nyquist
        cutoff_norm = max(0.001, min(0.999, cutoff_norm))
        
        # Design Butterworth highpass filter
        b, a = signal.butter(order, cutoff_norm, btype='high')
        
        # Apply filter
        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered_data
    
    def lowpass_filter(self, data: np.ndarray, cutoff: float, 
                      order: int = 4) -> np.ndarray:
        """
        Apply a lowpass filter to the EEG data.
        
        Args:
            data: Input EEG data (channels x samples)
            cutoff: Cutoff frequency in Hz
            order: Filter order (default: 4)
            
        Returns:
            Filtered EEG data with the same shape as input
        """
        nyquist = self.sample_rate / 2.0
        cutoff_norm = cutoff / nyquist
        cutoff_norm = max(0.001, min(0.999, cutoff_norm))
        
        # Design Butterworth lowpass filter
        b, a = signal.butter(order, cutoff_norm, btype='low')
        
        # Apply filter
        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered_data
    
    def z_score_normalize(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply z-score normalization to the data.
        
        Args:
            data: Input EEG data
            axis: Axis along which to normalize (default: -1, last axis)
            
        Returns:
            Z-score normalized data
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        
        # Prevent division by zero
        std = np.where(std == 0, 1, std)
        
        return (data - mean) / std
    
    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) to the EEG data.
        
        Args:
            data: Input EEG data (channels x samples)
            
        Returns:
            CAR-filtered EEG data
        """
        if data.ndim == 1:
            # Single channel - cannot apply CAR
            return data
        
        # Calculate common average across channels
        common_avg = np.mean(data, axis=0, keepdims=True)
        
        # Subtract common average from each channel
        return data - common_avg
    
    def preprocess_epoch(self, data: np.ndarray, l_freq: float = 0.5, 
                        h_freq: float = 50.0, apply_car: bool = True,
                        apply_notch: bool = True, normalize: bool = True) -> np.ndarray:
        """
        Apply a complete preprocessing pipeline to an EEG epoch.
        
        Args:
            data: Input EEG data (channels x samples)
            l_freq: Low frequency for bandpass filter
            h_freq: High frequency for bandpass filter
            apply_car: Whether to apply Common Average Reference
            apply_notch: Whether to apply notch filter (50Hz)
            normalize: Whether to apply z-score normalization
            
        Returns:
            Preprocessed EEG data
        """
        processed_data = data.copy()
        
        # Apply bandpass filter
        processed_data = self.bandpass_filter(processed_data, l_freq, h_freq)
        
        # Apply notch filter for powerline noise
        if apply_notch:
            processed_data = self.notch_filter(processed_data, freq=50.0)
        
        # Apply Common Average Reference
        if apply_car and processed_data.ndim > 1:
            processed_data = self.apply_car(processed_data)
        
        # Apply normalization
        if normalize:
            processed_data = self.z_score_normalize(processed_data)
        
        return processed_data
