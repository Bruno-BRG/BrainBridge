import torch
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn

# Definindo a arquitetura do modelo EEGNet
class EEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
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
        self.n_samples = n_samples
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

# Funções de pré-processamento
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=1)
    return y

def preprocess_data(data, fs=125):
    # 1. Filtragem passa-banda
    filtered_data = bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=fs)
    
    # 2. Normalização (StandardScaler)
    scaler = StandardScaler()
    # A normalização é aplicada em cada canal (linha) ao longo do tempo (colunas)
    # Transpomos para que o scaler atue nas features (canais)
    scaled_data = scaler.fit_transform(filtered_data.T).T
    
    return scaled_data

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, skiprows=4)
    
    eeg_columns = [f'EXG Channel {i}' for i in range(16)]
    eeg_data = df[eeg_columns].values.astype(float)
    
    # Encontrar os índices e os tipos dos eventos T1 e T2
    events = df[['Annotations']].dropna()
    events = events[events['Annotations'].isin(['T1', 'T2'])]
    
    event_mapping = {'T1': 0, 'T2': 1}
    
    all_trials = []
    all_labels = []
    for idx, row in events.iterrows():
        start_idx = idx + 1
        end_idx = start_idx + 400
        
        if end_idx <= len(eeg_data):
            trial = eeg_data[start_idx:end_idx, :].T
            all_trials.append(trial)
            all_labels.append(event_mapping[row['Annotations']])
            
    return np.array(all_trials), np.array(all_labels)

def main():
    # Carregar o modelo
    model_path = 'models/best_model.pth'
    model = EEGNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Carregar e preparar os dados
    file_path = 'S002R04_T0T1T0T2T0_full15_sequences.csv'
    trials, true_labels = load_and_prepare_data(file_path)
    
    if trials.shape[0] == 0:
        print("Nenhum trial completo de 400 amostras encontrado após T1/T2.")
        return

    print(f"Encontrados {trials.shape[0]} trials para avaliação.")

    # Pré-processar cada trial
    processed_trials = np.array([preprocess_data(trial) for trial in trials])
    
    # Converter para tensor do PyTorch
    trials_tensor = torch.from_numpy(processed_trials).float()
    trials_tensor = trials_tensor.unsqueeze(1)  # Adicionar dimensão de canal (batch, 1, 16, 400)
    
    # Fazer previsões
    with torch.no_grad():
        outputs = model(trials_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
    # Converter predições para numpy
    predictions_np = predictions.cpu().numpy()

    # Calcular e exibir métricas
    accuracy = accuracy_score(true_labels, predictions_np)
    report = classification_report(true_labels, predictions_np, target_names=['Mão Esquerda (T1)', 'Mão Direita (T2)'], zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions_np)

    print("\n--- Métricas de Avaliação ---")
    print(f"Acurácia: {accuracy:.2%}")
    print("\nRelatório de Classificação:")
    print(report)
    print("\nMatriz de Confusão:")
    print(conf_matrix)

    # Exibir resultados detalhados
    print("\n--- Resultados Detalhados da Avaliação ---")
    for i, (pred, true) in enumerate(zip(predictions_np, true_labels)):
        pred_label = "Mão Direita (T2)" if pred == 1 else "Mão Esquerda (T1)"
        true_label = "Mão Direita (T2)" if true == 1 else "Mão Esquerda (T1)"
        is_correct = "CORRETO" if pred == true else "INCORRETO"
        print(f"Trial {i+1}: Previsão = {pred_label} | Real = {true_label} -> {is_correct} (Confiança: {probabilities[i].max().item():.2f})")

if __name__ == '__main__':
    main()
