"""
Sistema BCI em Tempo Real - Versão Integrada
Recebe dados UDP do OpenBCI GUI, processa em tempo real com EEGNet, e produz saídas de classificação
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import logging
import json
import csv
from pathlib import Path
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from udp_receiver import UDPReceiver

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGNet(nn.Module):
    """Implementação compacta do EEGNet para BCI - MODELO PRINCIPAL"""
    
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Bloco 1: Temporal Convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        
        # Bloco 2: Depthwise Convolution (Spatial filtering)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Bloco 3: Separable Convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classificador
        self.feature_size = self._get_conv_output_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, n_classes)
        )
        
        self.apply(self._init_weights)
        
    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.numel()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, channels, samples) -> (batch, 1, channels, samples)
        
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x

class RobustEEGNormalizer:
    """Normalizador robusto para dados EEG"""
    
    def __init__(self, outlier_threshold=3.0):
        self.outlier_threshold = outlier_threshold
        self.stats = {}
        self.is_fitted = False
    
    def fit(self, X):
        # Garantir formato 3D (trials, channels, time)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 16, -1)
        
        # Tratar outliers usando IQR
        Q1 = np.percentile(X, 25, axis=(0, 2), keepdims=True)
        Q3 = np.percentile(X, 75, axis=(0, 2), keepdims=True)
        IQR = Q3 - Q1
        lower = Q1 - self.outlier_threshold * IQR
        upper = Q3 + self.outlier_threshold * IQR
        X = np.clip(X, lower, upper)
        
        # Calcular estatísticas por canal
        self.stats['median'] = np.median(X, axis=(0, 2), keepdims=True)
        self.stats['iqr'] = IQR + 1e-8
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Deve ajustar antes de transformar")
        
        original_shape = X.shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 16, -1)
        
        X_norm = (X - self.stats['median']) / self.stats['iqr']
        
        if len(original_shape) == 2:
            X_norm = X_norm.reshape(original_shape)
        return X_norm
    
    def transform_single(self, X):
        """Transformar uma única amostra"""
        if not self.is_fitted:
            raise ValueError("Deve ajustar antes de transformar")
        
        # Garantir formato (channels, samples)
        if len(X.shape) == 1:
            X = X.reshape(16, -1)
        
        # Expandir dimensões para compatibilidade
        X_expanded = X[np.newaxis, ...]  # (1, channels, samples)
        
        # Aplicar normalização
        X_norm = (X_expanded - self.stats['median']) / self.stats['iqr']
        
        # Retornar sem a dimensão extra
        return X_norm[0]  # (channels, samples)
    
    def load_stats(self, stats_dict):
        """Carregar estatísticas salvas"""
        self.stats = stats_dict
        self.is_fitted = True

class RealtimeBCIPredictor:
    """Sistema de inferência BCI em tempo real"""
    
    def __init__(self, 
                 model_path: str,
                 window_size: int = 400,
                 n_channels: int = 16,
                 sample_rate: float = 125.0,
                 prediction_callback: Optional[Callable] = None,
                 sliding_step: int = 50):
        
        self.model_path = Path(model_path)
        self.window_size = window_size
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.prediction_callback = prediction_callback
        self.sliding_step = sliding_step
        
        # Configurar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando device: {self.device}")
        
        # Carregar modelo
        self.model, self.normalizer = self._load_model()
        
        # Buffer para dados EEG
        self.buffer = deque(maxlen=window_size * 2)  # Buffer para 2 janelas
        self.buffer_lock = threading.Lock()
        
        # Estatísticas
        self.prediction_count = 0
        self.last_prediction_time = time.time()
        
        # Filtro passa-banda 1-40 Hz
        self.filter_b, self.filter_a = self._setup_filter()
        
        # Histórico de predições
        self.prediction_history = deque(maxlen=10)
        
        logger.info(f"BCIPredictor inicializado: {window_size} samples, {n_channels} canais")
    
    def _load_model(self):
        """Carregar modelo treinado"""
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            # Carregar checkpoint (PyTorch 2.6 compatibility)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extrair parâmetros do modelo
            if 'model_params' in checkpoint:
                model_params = checkpoint['model_params']
            else:
                # Parâmetros padrão
                model_params = {
                    'n_channels': self.n_channels,
                    'n_classes': 2,
                    'n_samples': self.window_size,
                    'dropout_rate': 0.5,
                    'F1': 4,
                    'D': 2,
                    'F2': 8
                }
            
            # Criar modelo
            model = EEGNet(**model_params)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Modelo carregado: {sum(p.numel() for p in model.parameters()):,} parâmetros")
            
            # Configurar normalizador
            normalizer = RobustEEGNormalizer()
            if 'normalization_stats' in checkpoint:
                normalizer.load_stats(checkpoint['normalization_stats'])
                logger.info("Estatísticas de normalização carregadas")
            else:
                logger.warning("Estatísticas de normalização não encontradas")
            
            return model, normalizer
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _setup_filter(self):
        """Configurar filtro passa-banda"""
        nyquist = self.sample_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 40.0 / nyquist
        b, a = butter(4, [low_freq, high_freq], btype='band')
        return b, a
    
    def _process_udp_data(self, data):
        """Processar dados UDP e adicionar ao buffer"""
        try:
            # Converter dados JSON para array numpy
            if isinstance(data, str):
                data = json.loads(data)
            
            # Extrair dados EEG
            if 'timeSeriesRaw' in data:
                time_series = data['timeSeriesRaw']
                if len(time_series) > 0:
                    # Converter para numpy array
                    sample_data = np.array(time_series[0])  # Primeira amostra
                    
                    # Garantir 16 canais
                    if len(sample_data) >= self.n_channels:
                        eeg_sample = sample_data[:self.n_channels]
                    else:
                        eeg_sample = np.zeros(self.n_channels)
                        eeg_sample[:len(sample_data)] = sample_data
                    
                    # Adicionar ao buffer
                    with self.buffer_lock:
                        self.buffer.append(eeg_sample)
                    
                    # Verificar se temos dados suficientes para predição
                    if len(self.buffer) >= self.window_size:
                        self._try_prediction()
                        
        except Exception as e:
            logger.error(f"Erro ao processar dados UDP: {e}")
    
    def _try_prediction(self):
        """Tentar fazer predição se temos dados suficientes"""
        try:
            with self.buffer_lock:
                if len(self.buffer) < self.window_size:
                    return
                
                # Extrair janela de dados
                window_data = np.array(list(self.buffer)[-self.window_size:])  # (samples, channels)
                
            # Transpor para formato (channels, samples)
            window_data = window_data.T
            
            # Aplicar filtro
            filtered_data = np.zeros_like(window_data)
            for ch in range(window_data.shape[0]):
                if np.any(np.isfinite(window_data[ch, :])):
                    filtered_data[ch, :] = filtfilt(self.filter_b, self.filter_a, window_data[ch, :])
            
            # Normalizar dados
            if self.normalizer.is_fitted:
                normalized_data = self.normalizer.transform_single(filtered_data)
            else:
                normalized_data = filtered_data
            
            # Converter para tensor
            input_tensor = torch.from_numpy(normalized_data).float().unsqueeze(0)  # (1, channels, samples)
            input_tensor = input_tensor.to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Criar resultado
            prediction_result = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': 'Left Hand' if predicted_class == 0 else 'Right Hand',
                'raw_probabilities': probabilities[0].cpu().numpy().tolist(),
                'prediction_count': self.prediction_count
            }
            
            # Atualizar histórico
            self.prediction_history.append(prediction_result)
            self.prediction_count += 1
            
            # Chamar callback se fornecido
            if self.prediction_callback:
                self.prediction_callback(prediction_result)
            
            # Log da predição
            logger.info(f"Predição {self.prediction_count}: {prediction_result['class_name']} "
                       f"(confiança: {confidence:.3f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return None
    
    def start_udp_processing(self, host='localhost', port=12345):
        """Iniciar processamento de dados UDP"""
        try:
            self.udp_receiver = UDPReceiver(host=host, port=port)
            self.udp_receiver.start(callback=self._process_udp_data)
            logger.info(f"Processamento UDP iniciado em {host}:{port}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar UDP: {e}")
            raise
    
    def stop_udp_processing(self):
        """Parar processamento de dados UDP"""
        if hasattr(self, 'udp_receiver'):
            self.udp_receiver.stop()
            logger.info("Processamento UDP parado")
    
    def get_prediction_stats(self):
        """Obter estatísticas de predição"""
        if not self.prediction_history:
            return None
        
        # Calcular estatísticas
        recent_predictions = list(self.prediction_history)[-10:]
        
        class_counts = {}
        confidences = []
        
        for pred in recent_predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(pred['confidence'])
        
        return {
            'total_predictions': self.prediction_count,
            'recent_predictions': len(recent_predictions),
            'class_distribution': class_counts,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'buffer_size': len(self.buffer),
            'last_prediction': recent_predictions[-1] if recent_predictions else None
        }

def default_prediction_callback(prediction_result):
    """Callback padrão para predições"""
    timestamp = prediction_result['timestamp']
    class_name = prediction_result['class_name']
    confidence = prediction_result['confidence']
    
    print(f"[{timestamp}] PREDIÇÃO: {class_name} (confiança: {confidence:.3f})")
    print(f"  Probabilidades: Esquerda={prediction_result['raw_probabilities'][0]:.3f}, "
          f"Direita={prediction_result['raw_probabilities'][1]:.3f}")

def main():
    """Função principal para teste"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema BCI em Tempo Real')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo .pt')
    parser.add_argument('--host', type=str, default='localhost', help='Host UDP')
    parser.add_argument('--port', type=int, default=12345, help='Porta UDP')
    parser.add_argument('--window-size', type=int, default=400, help='Tamanho da janela')
    parser.add_argument('--channels', type=int, default=16, help='Número de canais')
    parser.add_argument('--sample-rate', type=float, default=125.0, help='Taxa de amostragem')
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print(f"🚀 Iniciando Sistema BCI em Tempo Real")
    print(f"📂 Modelo: {model_path}")
    print(f"🌐 UDP: {args.host}:{args.port}")
    print(f"📊 Janela: {args.window_size} amostras, {args.channels} canais")
    print(f"⏱️ Taxa: {args.sample_rate} Hz")
    
    try:
        # Criar preditor
        predictor = RealtimeBCIPredictor(
            model_path=str(model_path),
            window_size=args.window_size,
            n_channels=args.channels,
            sample_rate=args.sample_rate,
            prediction_callback=default_prediction_callback
        )
        
        # Iniciar processamento
        predictor.start_udp_processing(host=args.host, port=args.port)
        
        print("✅ Sistema iniciado! Pressione Ctrl+C para parar.")
        print("📡 Aguardando dados UDP do OpenBCI GUI...")
        
        # Loop principal
        try:
            while True:
                time.sleep(1)
                
                # Mostrar estatísticas a cada 10 segundos
                if predictor.prediction_count % 10 == 0 and predictor.prediction_count > 0:
                    stats = predictor.get_prediction_stats()
                    if stats:
                        print(f"\n📊 Estatísticas: {stats['total_predictions']} predições")
                        print(f"   Distribuição: {stats['class_distribution']}")
                        print(f"   Confiança média: {stats['avg_confidence']:.3f}")
                        print(f"   Buffer: {stats['buffer_size']}/{predictor.window_size * 2}\n")
                
        except KeyboardInterrupt:
            print("\n⏹️ Parando sistema...")
            predictor.stop_udp_processing()
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return

if __name__ == "__main__":
    main()
