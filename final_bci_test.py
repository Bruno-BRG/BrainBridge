#!/usr/bin/env python3
"""
🎯 Teste LSL Final - Sistema BCI Completo

Script definitivo para testar seu sistema BCI:
✅ Detecta automaticamente streams LSL
✅ Coleta buffers de 400 frames (3.2s @ 125Hz)
✅ Aplica normalização idêntica ao treinamento
✅ Faz predições: 0 = Mão Esquerda, 1 = Mão Direita
✅ Estatísticas em tempo real
✅ Funciona com dados REAIS de EEG

Para testar sem hardware EEG, rode primeiro:
    python test_lsl_realtime_bci.py --simulate

Depois:
    python final_bci_test.py
"""

import numpy as np
import torch
import torch.nn as nn
import time
import logging
from pathlib import Path
from collections import deque
import threading
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# VERIFICAR LSL
# ============================================================================
try:
    from pylsl import StreamInlet, resolve_streams, resolve_byprop
    logger.info("✅ pylsl disponível")
except ImportError:
    logger.error("❌ Instale: pip install pylsl")
    sys.exit(1)

# ============================================================================
# MODELO (DO SEU NOTEBOOK)
# ============================================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.25, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Bloco 1: Temporal Convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Bloco 2: Depthwise Convolution
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
        
    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.numel()
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x

class AdvancedEEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.25, kernel_length=64, F1=8, D=2, F2=16):
        super(AdvancedEEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        self.temporal_conv1 = nn.Conv2d(1, F1//2, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.temporal_conv2 = nn.Conv2d(1, F1//2, (1, kernel_length//2), padding=(0, kernel_length // 4), bias=False)
        self.temporal_bn = nn.BatchNorm2d(F1)
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(F1, F1//4, 1),
            nn.ReLU(),
            nn.Conv2d(F1//4, F1, 1),
            nn.Sigmoid()
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(F2, F2*2, 1, bias=False),
            nn.BatchNorm2d(F2*2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2*2, F2),
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(F2, n_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        temp1 = self.temporal_conv1(x)
        temp2 = self.temporal_conv2(x)
        x = torch.cat([temp1, temp2], dim=1)
        x = self.temporal_bn(x)
        
        attention = self.spatial_attention(x)
        x = x * attention
        
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.feature_enhancement(x)
        x = self.classifier(x)
        
        return x

class CustomEEGModel(nn.Module):
    def __init__(self, n_chans=16, n_outputs=2, n_times=400, sfreq=125.0, 
                 model_type='advanced', **kwargs):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.model_type = model_type
        
        if model_type == 'advanced':
            self.model = AdvancedEEGNet(
                n_channels=n_chans, n_classes=n_outputs, n_samples=n_times, **kwargs
            )
        else:
            self.model = EEGNet(
                n_channels=n_chans, n_classes=n_outputs, n_samples=n_times, **kwargs
            )
        
        self.is_trained = False
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# CLASSE PRINCIPAL BCI
# ============================================================================

class FinalBCITest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalizer_stats = None
        self.buffer = deque(maxlen=1000)  # Buffer para 8 segundos
        self.predictions = []
        self.confidences = []
        self.processing_times = []
        
        logger.info(f"🖥️ Device: {self.device}")
    
    def load_model(self):
        """Carregar modelo mais recente"""
        models_dir = Path("models")
        
        if not models_dir.exists() or not list(models_dir.glob("*.pt")):
            logger.error("❌ Nenhum modelo encontrado! Execute o notebook de treinamento primeiro.")
            return False
        
        latest = max(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        logger.info(f"📁 Carregando: {latest.name}")
        
        try:
            checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
            
            # Criar modelo
            if 'constructor_args' in checkpoint:
                args = checkpoint['constructor_args']
                self.model = CustomEEGModel(**args)
            else:
                self.model = CustomEEGModel()
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Carregar estatísticas de normalização
            if 'normalization_stats' in checkpoint:
                self.normalizer_stats = checkpoint['normalization_stats']
                logger.info("✅ Normalizador do treinamento carregado")
            else:
                logger.info("⚠️ Usando normalização z-score padrão")
            
            # Info do modelo
            if 'test_accuracy' in checkpoint:
                logger.info(f"📊 Acurácia: {checkpoint['test_accuracy']:.3f}")
            
            logger.info("✅ Modelo carregado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False
    
    def find_stream(self):
        """Encontrar e conectar ao stream LSL"""
        logger.info("🔍 Procurando streams LSL...")
        
        # Tentar diferentes tipos
        streams = []
        for stream_type in ['EEG', 'ExG', 'EMG']:
            try:
                found = resolve_byprop('type', stream_type, timeout=2)
                streams.extend(found)
                if found:
                    logger.info(f"  📡 {len(found)} stream(s) {stream_type}")
            except:
                pass
        
        # Streams gerais
        try:
            general = resolve_streams(wait_time=3)
            streams.extend(general)
        except:
            pass
        
        if not streams:
            logger.error("❌ Nenhum stream encontrado!")
            logger.info("💡 Para testar sem hardware:")
            logger.info("   Terminal 1: python test_lsl_realtime_bci.py --simulate")
            logger.info("   Terminal 2: python final_bci_test.py")
            return None
        
        # Usar primeiro stream
        stream = streams[0]
        logger.info(f"🔗 Conectando: {stream.name()} ({stream.channel_count()} canais)")
        
        inlet = StreamInlet(stream, max_chunklen=1)
        return inlet
    
    def normalize_data(self, data):
        """Normalizar dados usando estatísticas do treinamento"""
        if self.normalizer_stats and 'median' in self.normalizer_stats and 'iqr' in self.normalizer_stats:
            try:
                median = self.normalizer_stats['median']
                iqr = self.normalizer_stats['iqr']
                
                # Converter para numpy se necessário
                if hasattr(median, 'numpy'):
                    median = median.numpy()
                if hasattr(iqr, 'numpy'):
                    iqr = iqr.numpy()
                
                # Garantir shapes corretos
                if median.ndim > 2:
                    median = median.squeeze()
                if iqr.ndim > 2:
                    iqr = iqr.squeeze()
                
                # Shape para (16, 1) para broadcasting
                if median.shape == (16,):
                    median = median.reshape(16, 1)
                if iqr.shape == (16,):
                    iqr = iqr.reshape(16, 1)
                
                return (data - median) / (iqr + 1e-8)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro na normalização salva: {e}")
        
        # Fallback: z-score simples
        return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    
    def predict(self, eeg_data):
        """Fazer predição"""
        start_time = time.time()
        
        # Normalizar
        normalized = self.normalize_data(eeg_data)
        
        # Para tensor
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
        
        processing_time = time.time() - start_time
        return prediction, confidence, processing_time
    
    def run_test(self, duration=300):  # 5 minutos por padrão
        """Executar teste principal"""
        logger.info("🧠 SISTEMA BCI EM TEMPO REAL")
        logger.info("=" * 40)
        
        # 1. Carregar modelo
        if not self.load_model():
            return
        
        # 2. Conectar LSL
        inlet = self.find_stream()
        if not inlet:
            return
        
        # 3. Aguardar dados iniciais
        logger.info("⏳ Coletando dados iniciais...")
        while len(self.buffer) < 400:
            samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=50)
            if samples:
                for sample in samples:
                    # Usar primeiros 16 canais ou preencher com zeros
                    if len(sample) >= 16:
                        self.buffer.append(sample[:16])
                    else:
                        padded = sample + [0.0] * (16 - len(sample))
                        self.buffer.append(padded[:16])
            
            logger.info(f"   📊 Buffer: {len(self.buffer)}/400 ({len(self.buffer)/400*100:.1f}%)")
        
        logger.info("✅ Buffer inicial pronto!")
        logger.info(f"🎯 Teste por {duration}s (Ctrl+C para parar)")
        logger.info("🧠 Predições: 0=Mão Esquerda, 1=Mão Direita")
        logger.info("-" * 50)
        
        # 4. Loop principal
        start_time = time.time()
        prediction_count = 0
        last_prediction_time = 0
        
        try:
            while (time.time() - start_time) < duration:
                # Coletar novos dados
                samples, _ = inlet.pull_chunk(timeout=0.1, max_samples=20)
                if samples:
                    for sample in samples:
                        if len(sample) >= 16:
                            self.buffer.append(sample[:16])
                        else:
                            padded = sample + [0.0] * (16 - len(sample))
                            self.buffer.append(padded[:16])
                
                # Fazer predição a cada 2 segundos
                current_time = time.time()
                if (current_time - last_prediction_time) >= 2.0 and len(self.buffer) >= 400:
                    # Pegar últimas 400 amostras
                    data = np.array(list(self.buffer)[-400:]).T  # (16, 400)
                    
                    # Predição
                    prediction, confidence, proc_time = self.predict(data)
                    
                    # Armazenar
                    self.predictions.append(prediction)
                    self.confidences.append(confidence)
                    self.processing_times.append(proc_time)
                    
                    prediction_count += 1
                    last_prediction_time = current_time
                    
                    # Log
                    class_name = "Mão Direita" if prediction == 1 else "Mão Esquerda"
                    elapsed = current_time - start_time
                    
                    logger.info(f"🧠 #{prediction_count:2d} [{elapsed:5.1f}s]: {class_name:12s} "
                              f"(conf: {confidence:.3f}, proc: {proc_time*1000:.1f}ms)")
                    
                    # Estatísticas a cada 10 predições
                    if prediction_count % 10 == 0:
                        self.print_stats()
                
                time.sleep(0.05)  # 20Hz de polling
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Teste interrompido pelo usuário")
        
        # Resultados finais
        total_time = time.time() - start_time
        self.print_final_results(total_time)
    
    def print_stats(self):
        """Imprimir estatísticas atuais"""
        if not self.predictions:
            return
        
        recent = self.predictions[-10:]
        left = recent.count(0)
        right = recent.count(1)
        avg_conf = np.mean(self.confidences[-10:])
        avg_time = np.mean(self.processing_times[-10:]) * 1000
        
        logger.info(f"📊 Últimas 10: {left}L/{right}R | Conf: {avg_conf:.3f} | {avg_time:.1f}ms")
    
    def print_final_results(self, total_time):
        """Resultados finais"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 RESULTADOS FINAIS")
        logger.info("=" * 50)
        
        if not self.predictions:
            logger.info("❌ Nenhuma predição realizada")
            return
        
        left_total = self.predictions.count(0)
        right_total = self.predictions.count(1)
        total_preds = len(self.predictions)
        
        logger.info(f"⏱️ Duração: {total_time:.1f}s")
        logger.info(f"🔢 Predições: {total_preds}")
        logger.info(f"📈 Taxa: {total_preds/total_time:.1f} pred/s")
        logger.info(f"👈 Mão Esquerda: {left_total} ({left_total/total_preds*100:.1f}%)")
        logger.info(f"👉 Mão Direita: {right_total} ({right_total/total_preds*100:.1f}%)")
        
        if self.confidences:
            avg_conf = np.mean(self.confidences)
            min_conf = np.min(self.confidences)
            max_conf = np.max(self.confidences)
            logger.info(f"🎯 Confiança: {avg_conf:.3f} (min: {min_conf:.3f}, max: {max_conf:.3f})")
        
        if self.processing_times:
            avg_proc = np.mean(self.processing_times) * 1000
            min_proc = np.min(self.processing_times) * 1000
            max_proc = np.max(self.processing_times) * 1000
            logger.info(f"⚡ Processamento: {avg_proc:.1f}ms (min: {min_proc:.1f}ms, max: {max_proc:.1f}ms)")
        
        # Verificar variação
        if len(set(self.predictions)) == 1:
            logger.info("⚠️ ATENÇÃO: Todas as predições são da mesma classe!")
            logger.info("   Possíveis causas:")
            logger.info("   - Modelo não foi treinado adequadamente")
            logger.info("   - Dados de entrada não têm padrões discriminativos")
            logger.info("   - Problema na normalização")
        else:
            logger.info("✅ Modelo produz predições variadas")
        
        logger.info("=" * 50)

def main():
    print("""
🎯 Teste LSL Final - Sistema BCI Completo
========================================

Este é o teste definitivo do seu sistema BCI:
✅ Carrega modelo treinado automaticamente
✅ Conecta ao primeiro stream LSL encontrado  
✅ Coleta buffers de 400 frames (3.2s @ 125Hz)
✅ Aplica normalização idêntica ao treinamento
✅ Faz predições a cada 2 segundos
✅ Mostra estatísticas em tempo real

Para testar SEM hardware EEG:
  Terminal 1: python test_lsl_realtime_bci.py --simulate
  Terminal 2: python final_bci_test.py

Com OpenBCI ou outro hardware:
  1. Configure stream LSL no dispositivo
  2. python final_bci_test.py

Pressione Ctrl+C para parar a qualquer momento.
""")
    
    test = FinalBCITest()
    test.run_test(duration=300)  # 5 minutos

if __name__ == "__main__":
    main()
