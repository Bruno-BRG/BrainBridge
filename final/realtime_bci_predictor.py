"""
Pipeline de Inferência em Tempo Real para BCI
Módulo que recebe dados UDP e aplica modelo CNN EEGNet para classificação de movimento motor
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
import pickle
from udp_receiver import UDPReceiver

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGNet(nn.Module):
    """Implementação compacta do EEGNet para BCI - MODELO PRINCIPAL"""
    
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Bloco 1: Temporal Convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
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
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Calcular tamanho após convoluções
        self.feature_size = self._get_conv_output_size()
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, n_classes),
        )
        
        logger.info(f"✅ EEGNet criado: {sum(p.numel() for p in self.parameters()):,} parâmetros")
        
    def _get_conv_output_size(self):
        """Calcula o tamanho da saída das convoluções"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.numel()
    
    def forward(self, x):
        # x shape: (batch, channels, samples) -> (batch, 1, channels, samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class RobustEEGNormalizer:
    """Normalizador robusto para dados EEG - Compatível com treinamento"""
    
    def __init__(self):
        self.channel_means = None
        self.channel_stds = None
        self.fitted = False
        
    def fit(self, X):
        """Fit normalizador nos dados de treinamento"""
        # X shape: (samples, channels, time)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], 16, -1)
        
        # Calcular estatísticas por canal
        self.channel_means = np.mean(X, axis=(0, 2))  # (channels,)
        self.channel_stds = np.std(X, axis=(0, 2))    # (channels,)
        
        # Evitar divisão por zero
        self.channel_stds = np.where(self.channel_stds < 1e-8, 1.0, self.channel_stds)
        
        self.fitted = True
        return self
        
    def transform(self, X):
        """Transform dados usando estatísticas calculadas"""
        if not self.fitted:
            raise ValueError("Normalizador deve ser fitted antes de transform")
            
        original_shape = X.shape
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], 16, -1)
            
        # Normalizar por canal
        X_norm = (X - self.channel_means[None, :, None]) / self.channel_stds[None, :, None]
        
        # Restaurar shape original
        X_norm = X_norm.reshape(original_shape)
        return X_norm
        
    def fit_transform(self, X):
        """Fit e transform em uma operação"""
        return self.fit(X).transform(X)
        
    def get_stats(self):
        """Retorna estatísticas para salvar"""
        return {
            'channel_means': self.channel_means,
            'channel_stds': self.channel_stds,
            'fitted': self.fitted
        }
        
    def set_stats(self, stats):
        """Define estatísticas carregadas"""
        self.channel_means = stats['channel_means']
        self.channel_stds = stats['channel_stds'] 
        self.fitted = stats['fitted']


class RealTimeBCIPredictor:
    """
    Preditor BCI em tempo real usando janela deslizante
    """
    
    def __init__(self, model_path: str, host: str = 'localhost', port: int = 12345):
        """
        Inicializa o preditor em tempo real
        
        Args:
            model_path: Caminho para o modelo treinado (.pt)
            host: Host UDP
            port: Porta UDP
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        
        # Configurações da janela deslizante
        self.window_size = 400  # 400 amostras por predição
        self.n_channels = 16
        self.sample_rate = 250  # Hz
        
        # Buffer para dados EEG (janela deslizante)
        self.eeg_buffer = deque(maxlen=self.window_size)
        self.buffer_lock = threading.Lock()
        
        # Modelo e normalizador
        self.model = None
        self.normalizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Controle de execução
        self.is_predicting = False
        self.prediction_callback = None
        
        # Estatísticas
        self.total_predictions = 0
        self.predictions_history = deque(maxlen=100)  # Últimas 100 predições
        self.processing_times = deque(maxlen=50)
        
        # Receptor UDP
        self.udp_receiver = UDPReceiver(host=host, port=port)
        
        # Carregar modelo
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo treinado e normalizador"""
        try:
            logger.info(f"Carregando modelo de: {self.model_path}")
            
            # Carregar checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Criar modelo
            self.model = EEGNet(
                n_channels=16,
                n_classes=2, 
                n_samples=400
            ).to(self.device)
            
            # Carregar pesos
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Carregar normalizador
            if 'normalization_stats' in checkpoint:
                self.normalizer = RobustEEGNormalizer()
                self.normalizer.set_stats(checkpoint['normalization_stats'])
                logger.info("✅ Normalizador carregado do checkpoint")
            else:
                logger.warning("⚠️ Normalizador não encontrado no checkpoint")
                self.normalizer = RobustEEGNormalizer()
                
            logger.info(f"✅ Modelo carregado com sucesso")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Parâmetros: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
            
    def set_prediction_callback(self, callback: Callable[[int, float, Dict], None]):
        """
        Define callback para receber predições
        
        Args:
            callback: Função que recebe (predição, confiança, metadados)
        """
        self.prediction_callback = callback
        
    def _process_udp_data(self, data: Any):
        """Processa dados UDP e adiciona ao buffer"""
        if not self.is_predicting:
            return
            
        try:
            # Extrair dados EEG do formato OpenBCI GUI
            eeg_samples = self._extract_eeg_samples(data)
            
            if eeg_samples is not None and len(eeg_samples) > 0:
                # Adicionar amostras ao buffer
                with self.buffer_lock:
                    for sample in eeg_samples:
                        self.eeg_buffer.append(sample)
                        
                        # Se buffer está cheio, fazer predição
                        if len(self.eeg_buffer) == self.window_size:
                            self._make_prediction()
                            
        except Exception as e:
            logger.error(f"Erro ao processar dados UDP: {e}")
            
    def _extract_eeg_samples(self, data: Any) -> Optional[List[List[float]]]:
        """
        Extrai amostras EEG dos dados UDP
        
        Args:
            data: Dados UDP recebidos
            
        Returns:
            Lista de amostras [amostra1, amostra2, ...] onde cada amostra é [ch1, ch2, ..., ch16]
        """
        try:
            if isinstance(data, dict):
                # Formato OpenBCI GUI: {'type': 'timeSeriesRaw', 'data': [[ch1_samples], [ch2_samples], ...]}
                if 'type' in data and data['type'] == 'timeSeriesRaw' and 'data' in data:
                    channel_arrays = data['data']
                    
                    if len(channel_arrays) >= self.n_channels:
                        # Transpor para formato [amostra][canal]
                        n_samples = len(channel_arrays[0])
                        samples = []
                        
                        for sample_idx in range(n_samples):
                            sample = []
                            for ch_idx in range(self.n_channels):
                                if ch_idx < len(channel_arrays):
                                    sample.append(float(channel_arrays[ch_idx][sample_idx]))
                                else:
                                    sample.append(0.0)
                            samples.append(sample)
                            
                        return samples
                        
        except Exception as e:
            logger.error(f"Erro ao extrair amostras EEG: {e}")
            
        return None
        
    def _make_prediction(self):
        """Faz predição com a janela atual"""
        start_time = time.time()
        
        try:
            with self.buffer_lock:
                # Converter buffer para numpy array
                window_data = np.array(list(self.eeg_buffer))  # (400, 16)
                
            # Transpor para formato do modelo: (1, 16, 400)
            window_data = window_data.T  # (16, 400)
            window_data = window_data[np.newaxis, :, :]  # (1, 16, 400)
            
            # Normalizar
            if self.normalizer and self.normalizer.fitted:
                window_data = self.normalizer.transform(window_data)
                
            # Converter para tensor
            input_tensor = torch.FloatTensor(window_data).to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            # Estatísticas
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.total_predictions += 1
            
            # Metadados da predição
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': processing_time * 1000,
                'confidence': confidence,
                'probabilities': {
                    'left_hand': probabilities[0][0].item(),
                    'right_hand': probabilities[0][1].item()
                },
                'prediction_count': self.total_predictions,
                'window_size': self.window_size
            }
            
            # Armazenar histórico
            self.predictions_history.append({
                'prediction': predicted_class,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Chamar callback
            if self.prediction_callback:
                self.prediction_callback(predicted_class, confidence, metadata)
                
            # Log da predição
            class_name = "Mão Esquerda (T1)" if predicted_class == 0 else "Mão Direita (T2)"
            logger.info(f"🧠 Predição: {class_name} | Confiança: {confidence:.3f} | "
                       f"Tempo: {processing_time*1000:.1f}ms")
                       
        except Exception as e:
            logger.error(f"Erro ao fazer predição: {e}")
            
    def start_prediction(self):
        """Inicia o sistema de predição em tempo real"""
        if self.is_predicting:
            logger.warning("Sistema já está fazendo predições")
            return
            
        try:
            # Configurar callback UDP
            self.udp_receiver.set_callback(self._process_udp_data)
            
            # Iniciar receptor UDP
            self.udp_receiver.start()
            
            self.is_predicting = True
            logger.info("🚀 Sistema de predição BCI iniciado")
            logger.info(f"   Modelo: {self.model_path}")
            logger.info(f"   Janela: {self.window_size} amostras ({self.window_size/self.sample_rate:.1f}s)")
            logger.info(f"   Canais: {self.n_channels}")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar predição: {e}")
            self.is_predicting = False
            
    def stop_prediction(self):
        """Para o sistema de predição"""
        if not self.is_predicting:
            return
            
        logger.info("🛑 Parando sistema de predição...")
        self.is_predicting = False
        
        # Parar receptor UDP
        self.udp_receiver.stop()
        
        # Limpar buffer
        with self.buffer_lock:
            self.eeg_buffer.clear()
            
        logger.info("Sistema de predição parado")
        
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        with self.buffer_lock:
            buffer_fill = len(self.eeg_buffer)
            
        # Calcular estatísticas das predições
        recent_predictions = list(self.predictions_history)
        if recent_predictions:
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
            left_count = sum(1 for p in recent_predictions if p['prediction'] == 0)
            right_count = sum(1 for p in recent_predictions if p['prediction'] == 1)
        else:
            avg_confidence = 0.0
            left_count = 0
            right_count = 0
            
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            'is_predicting': self.is_predicting,
            'total_predictions': self.total_predictions,
            'buffer_fill': buffer_fill,
            'buffer_progress': buffer_fill / self.window_size * 100,
            'udp_received': self.udp_receiver.get_data_count(),
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'recent_predictions': {
                'left_hand_count': left_count,
                'right_hand_count': right_count,
                'total': len(recent_predictions)
            },
            'model_info': {
                'path': self.model_path,
                'device': str(self.device),
                'window_size': self.window_size,
                'channels': self.n_channels
            }
        }


def prediction_handler(prediction: int, confidence: float, metadata: Dict):
    """Handler exemplo para predições"""
    class_name = "🤚 Mão Esquerda" if prediction == 0 else "✋ Mão Direita"
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    
    print(f"🧠 {class_name} | {confidence_bar} {confidence:.1%} | "
          f"{metadata['processing_time_ms']:.1f}ms | "
          f"#{metadata['prediction_count']}")


def main():
    """Teste do sistema de predição em tempo real"""
    print("🧠 SISTEMA BCI - PREDIÇÃO EM TEMPO REAL")
    print("=" * 60)
    
    # Caminho do modelo (ajustar conforme necessário)
    model_path = "../models/custom_eegnet_1751389051.pt"
    
    try:
        # Criar preditor
        predictor = RealTimeBCIPredictor(model_path)
        
        # Definir callback para predições
        predictor.set_prediction_callback(prediction_handler)
        
        # Iniciar predição
        predictor.start_prediction()
        
        print("🎯 Sistema aguardando dados EEG...")
        print("🛑 Pressione Ctrl+C para parar")
        print("=" * 60)
        
        # Loop de monitoramento
        counter = 0
        while True:
            time.sleep(3)
            counter += 3
            
            stats = predictor.get_stats()
            
            print(f"[{counter:03d}s] 📊 UDP: {stats['udp_received']:4d} | "
                  f"Buffer: {stats['buffer_progress']:5.1f}% | "
                  f"Predições: {stats['total_predictions']:3d} | "
                  f"Conf.Média: {stats['avg_confidence']:.2f} | "
                  f"Tempo: {stats['avg_processing_time_ms']:.1f}ms")
            
    except KeyboardInterrupt:
        print("\n🛑 Parando sistema...")
        predictor.stop_prediction()
        
        stats = predictor.get_stats()
        print("📊 ESTATÍSTICAS FINAIS:")
        print(f"   • Total de predições: {stats['total_predictions']}")
        print(f"   • Mão esquerda: {stats['recent_predictions']['left_hand_count']}")
        print(f"   • Mão direita: {stats['recent_predictions']['right_hand_count']}")
        print(f"   • Confiança média: {stats['avg_confidence']:.2f}")
        print("✅ Sistema finalizado!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()
