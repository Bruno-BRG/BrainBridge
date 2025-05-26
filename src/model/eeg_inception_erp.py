"""
EEGInceptionERP model wrapper using braindecode library
Author: GitHub Copilot  
License: BSD (3-clause)

This module implements an EEG classification model using the EEGInceptionERP
architecture from the braindecode library. The model is designed for EEG signal
classification tasks, particularly motor imagery.
"""

import torch
import torch.nn as nn
try:
    from braindecode.models import EEGInceptionERP as BraindecodeEEGInceptionERP
except ImportError:
    # Fallback to EEGNetv4 if EEGInceptionERP is not available
    try:
        from braindecode.models import EEGNetv4 as BraindecodeEEGInceptionERP
    except ImportError:
        # If both fail, create a simple CNN as fallback
        BraindecodeEEGInceptionERP = None


class EEGModel(torch.nn.Module):
    """
    A wrapper class for the braindecode EEG model with additional functionality
    for BCI applications.
    
    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_outputs : int
        Number of classes to predict (2 for binary classification).
    n_times : int
        Number of time points in the input.
    sfreq : float
        Sampling frequency of the input signals.
    drop_prob : float, optional
        Dropout probability (default=0.5).
    n_filters : int, optional
        Number of temporal filters (default=8).
    """
    
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        sfreq: float,
        drop_prob: float = 0.5,
        n_filters: int = 8
    ):
        super().__init__()
        
        # Store configuration
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        
        # Initialize the model
        if BraindecodeEEGInceptionERP is not None:
            try:
                self.model = BraindecodeEEGInceptionERP(
                    n_chans=n_chans,
                    n_outputs=n_outputs,  # Use n_outputs instead of n_classes
                    n_times=n_times,      # Use n_times instead of input_window_samples
                    sfreq=sfreq,
                    drop_prob=drop_prob,
                    n_filters=n_filters,
                )
            except Exception:
                # If parameters are incompatible, try simplified version
                self.model = BraindecodeEEGInceptionERP(
                    n_chans=n_chans,
                    n_outputs=n_outputs,
                    n_times=n_times,
                )
        else:
            # Fallback simple CNN if braindecode models are not available
            self.model = self._create_simple_cnn(n_chans, n_outputs, n_times, drop_prob)

    def _create_simple_cnn(self, n_chans, n_outputs, n_times, drop_prob):
        """Create a simple CNN as fallback"""
        return nn.Sequential(
            # Temporal convolution
            nn.Conv1d(n_chans, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            
            # Spatial-temporal convolution
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_prob),
            
            # Additional layers
            nn.Conv1d(64, 128, kernel_size=10, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_prob),
            
            # Global average pooling and classification
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs)
        """
        # Handle input shape for simple CNN fallback
        if hasattr(self.model, '__len__'):  # It's a Sequential model (fallback)
            # Transpose for 1D conv: (batch, channels, time)
            if x.dim() == 3:
                return self.model(x)
            else:
                return self.model(x.squeeze())
        else:
            # Braindecode model
            return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the second-to-last layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times)
            
        Returns
        -------
        torch.Tensor
            Feature tensor from the penultimate layer
        """
        # Get all intermediate activations
        activations = []
        
        def hook(module, input, output):
            activations.append(output)
            
        # Register forward hook on the second-to-last layer
        modules = list(self.model.modules())
        if len(modules) > 2:
            handle = modules[-2].register_forward_hook(hook)
        else:
            handle = modules[-1].register_forward_hook(hook)
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove the hook
        handle.remove()
        
        return activations[0] if activations else torch.tensor([])
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device


# Alternative constructor for compatibility
def EEGInceptionERP(n_chans: int, n_outputs: int, n_times: int, sfreq: float, **kwargs):
    """
    Create an EEGInceptionERP model instance
    
    Parameters
    ----------
    n_chans : int
        Number of EEG channels
    n_outputs : int
        Number of output classes
    n_times : int
        Number of time points
    sfreq : float
        Sampling frequency
    **kwargs
        Additional arguments
    
    Returns
    -------
    EEGModel
        Model instance
    """
    return EEGModel(n_chans, n_outputs, n_times, sfreq, **kwargs)


# Re-export the model
__all__ = ['EEGModel', 'EEGInceptionERP']
