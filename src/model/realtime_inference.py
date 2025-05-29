"""
Class:   RealTimeInferenceProcessor
Purpose: Handles real-time EEG inference by maintaining a sliding window buffer
         of EEG samples, applying preprocessing, and running model inference.
Author:  Bruno Rocha
Created: 2025-01-21
Notes:   Follows Task Management & Coding Guide for Copilot v2.0.
         Designed for use with PyLSL streams for real-time BCI applications.
"""

import numpy as np
import torch
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
import traceback
from .EEGFilter import EEGFilter


class RealTimeInferenceProcessor:
    """
    Processes real-time EEG data for inference using a sliding window approach.
    
    This class maintains a buffer of EEG samples, applies filtering and preprocessing,
    and runs inference on 400-sample windows every 30ms for real-time BCI applications.
    """
    
    def __init__(
        self, 
        model,
        n_channels: int = 16,
        sample_rate: float = 125.0,
        window_size: int = 400,
        filter_enabled: bool = True,
        l_freq: float = 0.5,
        h_freq: float = 50.0
    ):
        """
        Initialize the real-time inference processor.
        
        Args:
            model: The trained EEG model for inference (EEGInceptionERPModel or EEGITNetModel)
            n_channels (int): Number of EEG channels expected
            sample_rate (float): Sample rate in Hz
            window_size (int): Number of samples in inference window (default 400 for 3.2s at 125Hz)
            filter_enabled (bool): Whether to apply bandpass filtering
            l_freq (float): Low frequency cutoff for bandpass filter
            h_freq (float): High frequency cutoff for bandpass filter
        """
        self.model = model
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.filter_enabled = filter_enabled
        
        # Initialize EEG filter
        if self.filter_enabled:
            self.eeg_filter = EEGFilter(
                sfreq=sample_rate,
                l_freq=l_freq,
                h_freq=h_freq,
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'}
            )
        else:
            self.eeg_filter = None
        
        # Initialize sample buffer using deque for efficient append/pop
        # Store samples as (n_channels,) arrays
        self.sample_buffer = deque(maxlen=window_size)
        
        # Initialize buffer with zeros
        self._initialize_buffer()
        
        # Statistics tracking
        self.inference_count = 0
        self.last_inference_result = None
        
        # Set model to evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def _initialize_buffer(self):
        """Initialize the sample buffer with zeros."""
        for _ in range(self.window_size):
            self.sample_buffer.append(np.zeros(self.n_channels, dtype=np.float32))
    
    def add_samples(self, samples: np.ndarray) -> None:
        """
        Add new EEG samples to the buffer.
        
        Args:
            samples (np.ndarray): New samples with shape (n_samples, n_channels) or (n_channels, n_samples)
        """
        if len(samples.shape) != 2:
            raise ValueError(f"Expected 2D samples array, got shape {samples.shape}")
        
        # Ensure samples are in (n_samples, n_channels) format
        if samples.shape[1] == self.n_channels:
            # samples is (n_samples, n_channels)
            pass
        elif samples.shape[0] == self.n_channels:
            # samples is (n_channels, n_samples) - transpose
            samples = samples.T
        else:
            raise ValueError(f"Sample dimensions don't match expected channels {self.n_channels}, got shape {samples.shape}")
        
        # Add each sample to the buffer
        for sample in samples:
            self.sample_buffer.append(sample.astype(np.float32))
    
    def get_current_window(self) -> np.ndarray:
        """
        Get the current window of samples for inference.
        
        Returns:
            np.ndarray: Window of shape (n_channels, window_size)
        """
        if len(self.sample_buffer) < self.window_size:
            raise RuntimeError(f"Buffer not full enough for inference. Has {len(self.sample_buffer)}, needs {self.window_size}")
        
        # Convert deque to numpy array and transpose to (n_channels, n_samples)
        window = np.array(list(self.sample_buffer)).T  # Shape: (n_channels, window_size)
        return window
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to the window.
        
        Args:
            window (np.ndarray): Raw window of shape (n_channels, window_size)
            
        Returns:
            np.ndarray: Preprocessed window
        """
        processed_window = window.copy()
        
        # Apply bandpass filtering if enabled
        if self.filter_enabled and self.eeg_filter is not None:
            try:
                processed_window = self.eeg_filter.filter_stream(processed_window)
            except Exception as e:
                print(f"Warning: Filtering failed: {e}")
                # Continue with unfiltered data
        
        # Apply standardization (z-score normalization)
        # Calculate mean and std across time dimension for each channel
        mean = np.mean(processed_window, axis=1, keepdims=True)
        std = np.std(processed_window, axis=1, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        processed_window = (processed_window - mean) / std
        
        return processed_window
    
    def predict(self, new_samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run inference on the current window.
        
        Args:
            new_samples (Optional[np.ndarray]): New samples to add before inference
            
        Returns:
            Dict[str, Any]: Inference results containing predictions and confidence
        """
        try:
            # Add new samples if provided
            if new_samples is not None:
                self.add_samples(new_samples)
            
            # Check if buffer is ready
            if len(self.sample_buffer) < self.window_size:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need {self.window_size - len(self.sample_buffer)} more samples',
                    'confidence': 0.0,
                    'prediction': 'unknown'
                }
            
            # Get current window and preprocess
            window = self.get_current_window()  # Shape: (n_channels, window_size)
            processed_window = self.preprocess_window(window)
            
            # Convert to tensor for model input
            # Model expects (batch_size, n_channels, n_times)
            tensor_input = torch.from_numpy(processed_window).float().unsqueeze(0)  # Add batch dimension
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    output = self.model.forward(tensor_input)
                else:
                    output = self.model(tensor_input)
            
            # Process model output
            if output.dim() == 2 and output.shape[1] == 2:
                # Classification output with 2 classes
                probabilities = torch.softmax(output, dim=1)
                class_probs = probabilities[0].cpu().numpy()
                predicted_class = int(torch.argmax(output, dim=1)[0])
                confidence = float(class_probs[predicted_class])
                
                class_names = ['left_hand', 'right_hand']
                prediction = class_names[predicted_class]
                
            elif output.dim() == 2 and output.shape[1] == 1:
                # Binary classification output
                prob = torch.sigmoid(output)[0, 0].cpu().numpy()
                predicted_class = int(prob > 0.5)
                confidence = float(prob if predicted_class == 1 else 1 - prob)
                
                class_names = ['left_hand', 'right_hand']
                prediction = class_names[predicted_class]
                class_probs = [1 - prob, prob]
                
            else:
                # Unknown output format
                return {
                    'status': 'error',
                    'message': f'Unexpected model output shape: {output.shape}',
                    'confidence': 0.0,
                    'prediction': 'unknown'
                }
            
            # Update statistics
            self.inference_count += 1
            
            result = {
                'status': 'success',
                'prediction': prediction,
                'confidence': confidence,
                'class_probabilities': {
                    'left_hand': float(class_probs[0]),
                    'right_hand': float(class_probs[1])
                },
                'inference_count': self.inference_count,
                'buffer_size': len(self.sample_buffer),
                'window_size': self.window_size
            }
            
            self.last_inference_result = result
            return result
            
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            print(f"RealTimeInferenceProcessor error: {error_msg}")
            print(traceback.format_exc())
            
            return {
                'status': 'error',
                'message': error_msg,
                'confidence': 0.0,
                'prediction': 'error'
            }
    
    def reset_buffer(self):
        """Reset the sample buffer to initial state."""
        self.sample_buffer.clear()
        self._initialize_buffer()
        self.inference_count = 0
        self.last_inference_result = None
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Get information about the current buffer state.
        
        Returns:
            Dict containing buffer statistics
        """
        return {
            'current_size': len(self.sample_buffer),
            'max_size': self.window_size,
            'fill_percentage': (len(self.sample_buffer) / self.window_size) * 100,
            'ready_for_inference': len(self.sample_buffer) >= self.window_size,
            'sample_rate': self.sample_rate,
            'n_channels': self.n_channels,
            'inference_count': self.inference_count,
            'filter_enabled': self.filter_enabled
        }
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure processor parameters.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
        """
        if 'filter_enabled' in config:
            self.filter_enabled = config['filter_enabled']
        
        if 'l_freq' in config or 'h_freq' in config:
            l_freq = config.get('l_freq', 0.5)
            h_freq = config.get('h_freq', 50.0)
            if self.filter_enabled:
                self.eeg_filter = EEGFilter(
                    sfreq=self.sample_rate,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method='iir',
                    iir_params={'order': 4, 'ftype': 'butter'}
                )
