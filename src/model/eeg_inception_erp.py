"""
EEGInceptionERP model wrapper using braindecode library
Author: GitHub Copilot  
License: BSD (3-clause)

This module implements an EEG classification model using the EEGInceptionERP
architecture from the braindecode library. The model is designed for EEG signal
classification tasks, particularly motor imagery.
"""

import torch
from braindecode.models import EEGNetv4

class EEGModel(torch.nn.Module):
    """
    A wrapper class for the braindecode EEGNetv4 model with additional functionality
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
    scaling_n_filters : float, optional
        Scaling factor for number of filters (default=1.0).
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
        self.model = EEGNetv4(
            in_chans=n_chans,
            n_classes=n_outputs,
            input_window_samples=n_times,
            final_conv_length='auto',
            pool_mode='max',
            F1=8*n_filters,
            drop_prob=drop_prob,
            sfreq=sfreq,
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
        handle = list(self.model.modules())[-2].register_forward_hook(hook)
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove the hook
        handle.remove()
        
        return activations[0]
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

# Re-export the model from braindecode
__all__ = ['EEGInceptionERP']
