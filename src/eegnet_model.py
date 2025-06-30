"""
EEGNet model implementation for BCI system
Based on the original EEGNet paper by Lawhern et al. (2018)
Adapted for motor imagery classification (left vs right hand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EEGNet(nn.Module):
    """
    EEGNet implementation for motor imagery classification
    
    Paper: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    Authors: Lawhern et al. (2018)
    
    Adapted for:
    - 16 EEG channels
    - 400 time points (3.2 seconds at 125 Hz)
    - 2 classes (left vs right hand)
    """
    
    def __init__(
        self,
        n_chans: int = 16,
        n_outputs: int = 2,
        n_times: int = 400,
        sfreq: float = 125.0,
        dropout_rate: float = 0.25,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
        **kwargs
    ):
        """
        Initialize EEGNet
        
        Args:
            n_chans: Number of EEG channels (default: 16)
            n_outputs: Number of output classes (default: 2)
            n_times: Number of time points (default: 400)
            sfreq: Sampling frequency (default: 125.0)
            dropout_rate: Dropout rate (default: 0.25)
            kernel_length: Length of temporal convolution (default: 64)
            F1: Number of temporal filters (default: 8)
            D: Depth multiplier for depthwise convolution (default: 2)
            F2: Number of pointwise filters (default: 16)
            norm_rate: Max norm constraint for classification layer (default: 0.25)
        """
        super(EEGNet, self).__init__()
        
        # Store parameters
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_chans, 1),
            groups=F1,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable convolution
        self.conv3 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate size after convolutions
        self._calculate_feature_size()
        
        # Classification layer
        self.classifier = nn.Linear(self.feature_size, n_outputs)
        
        # Initialize weights
        self._initialize_weights()
        
    def _calculate_feature_size(self):
        """Calculate the size of features after all convolutions"""
        # Simulate forward pass to get feature size
        x = torch.zeros(1, 1, self.n_chans, self.n_times)
        
        # Block 1
        x = self.conv1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.avgpool1(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool2(x)
        
        self.feature_size = x.view(x.size(0), -1).size(1)
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, n_chans, n_times)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        # Ensure input has correct shape: (batch_size, 1, n_chans, n_times)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: Depthwise spatial convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable convolution
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_type': 'EEGNet',
            'n_chans': self.n_chans,
            'n_outputs': self.n_outputs,
            'n_times': self.n_times,
            'sfreq': self.sfreq,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class EEGNetWrapper(nn.Module):
    """
    Wrapper for EEGNet to maintain compatibility with existing inference engine
    """
    
    def __init__(self, n_chans: int, n_outputs: int, n_times: int, sfreq: float = 125.0, **kwargs):
        super().__init__()
        
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.model_type = "EEGNet"
        
        # Create EEGNet model
        self.model = EEGNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            **kwargs
        )
        
    def forward(self, x):
        return self.model(x)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with compatibility handling"""
        try:
            return self.model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠️ Size mismatch detected, trying strict=False: {e}")
                return self.model.load_state_dict(state_dict, strict=False)
            else:
                raise e
    
    def state_dict(self):
        return self.model.state_dict()
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode=True):
        self.model.train(mode)
        return self
    
    def get_model_info(self):
        return self.model.get_model_info()


def create_eegnet_model(n_chans: int = 16, n_outputs: int = 2, n_times: int = 400, sfreq: float = 125.0, **kwargs):
    """
    Factory function to create EEGNet model
    
    Args:
        n_chans: Number of EEG channels
        n_outputs: Number of output classes
        n_times: Number of time points
        sfreq: Sampling frequency
        **kwargs: Additional arguments for EEGNet
        
    Returns:
        EEGNetWrapper instance
    """
    return EEGNetWrapper(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    print("🧠 Testando modelo EEGNet personalizado...")
    
    # Create model
    model = create_eegnet_model(n_chans=16, n_outputs=2, n_times=400)
    
    # Print model info
    info = model.get_model_info()
    print(f"📊 Informações do modelo:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test forward pass
    print(f"\n🔮 Testando forward pass...")
    batch_size = 4
    test_input = torch.randn(batch_size, 16, 400)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output example: {output[0].numpy()}")
    
    # Test with different input shapes
    print(f"\n🧪 Testando diferentes formatos de entrada...")
    
    # Single sample
    single_sample = torch.randn(16, 400)
    output_single = model(single_sample)
    print(f"✅ Single sample: {single_sample.shape} → {output_single.shape}")
    
    # With batch dimension
    batch_sample = torch.randn(1, 16, 400)
    output_batch = model(batch_sample)
    print(f"✅ Batch sample: {batch_sample.shape} → {output_batch.shape}")
    
    print(f"\n🎉 Modelo EEGNet testado com sucesso!")
