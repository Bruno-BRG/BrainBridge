"""
Sistema BCI Integrado V2 - Captura UDP → Processamento → CNN → Predição
Arquitetura: UDP → OpenBCI CSV → CNN (400 amostras) → Output (0=esquerda, 1=direita)
Versão atualizada para o novo modelo EEGNet
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
import time
import threading
import logging
from datetime import datetime
from typing import Optional, Callable
import os
import sys

# Importar módulos do sistema UDP
from udp_receiver import UDPReceiver
from realtime_udp_converter import RealTimeUDPConverter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGNet(nn.Module):
    """
    Modelo EEGNet atualizado para classificação de motor imagery
    """
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

class RealTimeBCISystem:
    """
    Sistema BCI integrado para classificação em tempo real
    """
    
    def __init__(self, model_path: str, window_size: int = 400, sampling_rate: int = 125):
        """
        Inicializa o sistema BCI
        
        Args:
            model_path: Caminho para o modelo treinado (.pth)
            window_size: Tamanho da janela de análise (400 amostras = 3.2s @ 125Hz)
            sampling_rate: Taxa de amostragem (125Hz)
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.model_path = model_path
        
        # Buffer circular para armazenar dados EEG
        self.data_buffer = deque(maxlen=window_size * 2)  # Buffer duplo para sobreposição
        self.buffer_lock = threading.Lock()
        
        # Estatísticas
        self.total_predictions = 0
        self.predictions_history = deque(maxlen=100)  # Últimas 100 predições
        
        # Modelo e dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_running = False
        
        # Callback para predições
        self.prediction_callback = None
        
        # Sistema UDP
        self.udp_receiver = UDPReceiver()
        self.udp_converter = RealTimeUDPConverter()
        
        # Última predição
        self.last_prediction = None
        self.last_prediction_time = None
        
        # Normalização
        self.mean = None
        self.std = None
        
        logger.info(f"Sistema BCI inicializado - Janela: {window_size} amostras, Device: {self.device}")
        
    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
                
            # Criar modelo
            self.model = EEGNet(n_channels=16, n_classes=2, n_samples=self.window_size)
            
            # Carregar pesos
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Modelo carregado: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def set_prediction_callback(self, callback: Callable):
        """Define callback para receber predições"""
        self.prediction_callback = callback
        
    def _process_udp_data(self, data):
        """Processa dados UDP recebidos"""
        try:
            # Converter dados para formato OpenBCI
            converted_samples = self.udp_converter._convert_to_openbci_format(data)
            
            if converted_samples:
                # Extrair apenas os canais EEG (0-15)
                for sample in converted_samples:
                    eeg_channels = []
                    for ch in range(16):
                        eeg_channels.append(sample[f'EXG Channel {ch}'])
                    
                    # Adicionar ao buffer
                    with self.buffer_lock:
                        self.data_buffer.append(eeg_channels)
                    
                    # Verificar se temos dados suficientes para predição
                    if len(self.data_buffer) >= self.window_size:
                        self._trigger_prediction()
                        
        except Exception as e:
            logger.error(f"Erro ao processar dados UDP: {e}")
    
    def _trigger_prediction(self):
        """Dispara predição em thread separada"""
        threading.Thread(target=self._make_prediction, daemon=True).start()
    
    def _normalize_data(self, data):
        """Normaliza os dados EEG"""
        if self.mean is None or self.std is None:
            # Usar normalização online se não tiver estatísticas salvas
            self.mean = np.mean(data)
            self.std = np.std(data)
        return (data - self.mean) / (self.std + 1e-8)
    
    def _make_prediction(self):
        """Faz predição usando os dados do buffer"""
        try:
            # Copiar dados do buffer
            with self.buffer_lock:
                if len(self.data_buffer) < self.window_size:
                    return
                    
                # Pegar últimas window_size amostras
                window_data = list(self.data_buffer)[-self.window_size:]
            
            # Preparar dados para o modelo
            eeg_data = np.array(window_data)  # Shape: (400, 16)
            eeg_data = eeg_data.T  # Shape: (16, 400) - canais x amostras
            
            # Normalizar dados
            eeg_data = self._normalize_data(eeg_data)
            
            # Adicionar dimensões para batch e canal
            eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)  # (1, 16, 400)
            eeg_tensor = eeg_tensor.to(self.device)
            
            # Predição
            with torch.no_grad():
                output = self.model(eeg_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Interpretar resultado
            class_names = ['Mão Esquerda', 'Mão Direita']
            predicted_class = class_names[prediction]
            
            # Atualizar estatísticas
            self.total_predictions += 1
            self.predictions_history.append(prediction)
            self.last_prediction = prediction
            self.last_prediction_time = datetime.now()
            
            # Criar resultado
            result = {
                'prediction': prediction,
                'class': predicted_class,
                'confidence': confidence,
                'timestamp': self.last_prediction_time,
                'sample_count': len(window_data)
            }
            
            # Chamar callback se definido
            if self.prediction_callback:
                self.prediction_callback(result)
                
            logger.info(f"🧠 Predição: {predicted_class} (Confiança: {confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
    
    def start(self):
        """Inicia o sistema BCI"""
        if not self.load_model():
            return False
            
        try:
            # Configurar callback UDP
            self.udp_receiver.set_callback(self._process_udp_data)
            
            # Iniciar receptor UDP
            self.udp_receiver.start()
            
            self.is_running = True
            logger.info("🚀 Sistema BCI iniciado e aguardando dados...")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar sistema: {e}")
            return False
    
    def stop(self):
        """Para o sistema BCI"""
        self.is_running = False
        self.udp_receiver.stop()
        logger.info("🛑 Sistema BCI parado")
    
    def get_stats(self):
        """Retorna estatísticas do sistema"""
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
            
        # Calcular distribuição das predições
        if self.predictions_history:
            left_count = sum(1 for p in self.predictions_history if p == 0)
            right_count = sum(1 for p in self.predictions_history if p == 1)
        else:
            left_count = right_count = 0
            
        return {
            'is_running': self.is_running,
            'total_predictions': self.total_predictions,
            'buffer_size': buffer_size,
            'window_size': self.window_size,
            'last_prediction': self.last_prediction,
            'last_prediction_time': self.last_prediction_time,
            'recent_predictions': {
                'left_hand': left_count,
                'right_hand': right_count,
                'total': len(self.predictions_history)
            },
            'udp_received': self.udp_receiver.get_data_count()
        }

def prediction_callback(result):
    """Callback para processar predições"""
    timestamp = result['timestamp'].strftime('%H:%M:%S')
    print(f"[{timestamp}] 🎯 {result['class']} - {result['confidence']:.1%} confiança")

def main():
    """Função principal do sistema BCI"""
    print("🧠 SISTEMA BCI INTEGRADO V2 - TEMPO REAL")
    print("=" * 50)
    
    # Determinar caminho do modelo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(script_dir), "models", "best_model.pth")
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em: {model_path}")
        print("💡 Certifique-se de que o modelo treinado está na pasta 'models' com o nome 'best_model.pth'")
        return
    
    print(f"📁 Usando modelo: {model_path}")
    
    # Criar sistema
    bci_system = RealTimeBCISystem(model_path=model_path)
    bci_system.set_prediction_callback(prediction_callback)
    
    try:
        # Iniciar sistema
        if not bci_system.start():
            print("❌ Falha ao iniciar sistema")
            return
            
        print("📡 Aguardando dados UDP em localhost:12345")
        print("🧠 Processando janelas de 400 amostras (3.2s)")
        print("🎯 Predições: 0=Mão Esquerda, 1=Mão Direita")
        print("🛑 Pressione Ctrl+C para parar")
        print("=" * 50)
        
        # Loop de monitoramento
        counter = 0
        while True:
            time.sleep(3)
            counter += 3
            
            stats = bci_system.get_stats()
            
            print(f"[{counter:03d}s] 📊 Buffer: {stats['buffer_size']:3d}/{stats['window_size']} | "
                  f"Predições: {stats['total_predictions']:3d} | "
                  f"UDP: {stats['udp_received']:4d} | "
                  f"Última: {stats['last_prediction']}")
            
            # Estatísticas detalhadas a cada 15s
            if counter % 15 == 0:
                recent = stats['recent_predictions']
                print(f"   📈 Últimas {recent['total']} predições: "
                      f"Esquerda={recent['left_hand']}, Direita={recent['right_hand']}")
                
    except KeyboardInterrupt:
        print("\n🛑 Parando sistema...")
        bci_system.stop()
        
        # Estatísticas finais
        stats = bci_system.get_stats()
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   • Total de predições: {stats['total_predictions']}")
        print(f"   • Dados UDP recebidos: {stats['udp_received']}")
        print(f"   • Última predição: {stats['last_prediction']}")
        print("✅ Sistema parado com sucesso!")

if __name__ == "__main__":
    main()
