from torch import nn

class EEGNet(nn.Module):
    """Modelo EEGNet para classificação de sinais EEG"""
    def __init__(self):
        super().__init__()
        # Camada temporal
        self.conv1 = nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        self.depthwise = nn.Conv2d(16, 32, (16, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)
        
        # Camada separável
        self.separable = nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, False)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)
        
        # Classificação
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(736, 2)  # 2 classes: esquerda/direita
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Camada temporal
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Camada separável
        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Classificação
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x
