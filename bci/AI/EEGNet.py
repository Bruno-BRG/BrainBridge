import torch.nn as nn

class EEGNet(nn.Module):
    """Modelo EEGNet que coincide exatamente com o arquivo salvo"""
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, dropout_rate=0.5):
        super().__init__()
        
        # Architecture parameters
        self.F1 = 8  # Number of temporal filters
        self.F2 = 16  # Number of pointwise filters
        self.D = 2   # Depth multiplier
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        # Block 2: Spatial Filter
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (n_channels, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 8), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F2 * (n_samples // 32), n_classes)
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

