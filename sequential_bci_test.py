#!/usr/bin/env python3
"""
🎯 BCI Sequencial - 400 em 400 amostras SEM sobreposição

Este script funciona EXATAMENTE como você pediu:
✅ Coleta 400 amostras (índices 1-400)
✅ Faz predição
✅ Coleta próximas 400 amostras (índices 401-800)
✅ Faz predição
✅ E assim por diante...

SEM sobreposição entre os buffers!
"""

import numpy as np
import torch
import torch.nn as nn
import time
import logging
from pathlib import Path
from collections import deque
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Verificar LSL
try:
    from pylsl import StreamInlet, resolve_streams, resolve_byprop
    logger.info("✅ pylsl disponível")
except ImportError:
    logger.error("❌ Instale: pip install pylsl")
    sys.exit(1)

# ============================================================================
# MODELOS (SIMPLIFICADO)
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
# CLASSE BCI SEQUENCIAL (SEM SOBREPOSIÇÃO)
# ============================================================================

class SequentialBCITest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalizer_stats = None
        
        # Buffer para coleta sequencial
        self.current_buffer = []  # Buffer atual sendo preenchido
        
        # Estatísticas
        self.predictions = []
        self.confidences = []
        self.processing_times = []
        
        # Contadores
        self.total_samples_collected = 0
        self.prediction_count = 0
        
        logger.info(f"🖥️ Device: {self.device}")
        logger.info("🔄 Modo: SEQUENCIAL (400 em 400, SEM sobreposição)")
    
    def load_model(self):
        """Carregar modelo mais recente"""
        models_dir = Path("models")
        
        if not models_dir.exists() or not list(models_dir.glob("*.pt")):
            logger.error("❌ Nenhum modelo encontrado!")
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
                logger.info("✅ Normalizador carregado")
            
            # Info do modelo
            if 'test_accuracy' in checkpoint:
                logger.info(f"📊 Acurácia: {checkpoint['test_accuracy']:.3f}")
            
            logger.info("✅ Modelo carregado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False
    
    def find_stream(self):
        """Encontrar stream LSL"""
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
            logger.info("💡 Para testar: python test_lsl_realtime_bci.py --simulate")
            return None
        
        # Usar primeiro stream
        stream = streams[0]
        logger.info(f"🔗 Conectando: {stream.name()} ({stream.channel_count()} canais)")
        
        inlet = StreamInlet(stream, max_chunklen=1)
        return inlet
    
    def normalize_data(self, data):
        """Normalizar dados"""
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
    
    def collect_next_400_samples(self, inlet):
        """
        Coleta EXATAMENTE 400 amostras sequenciais
        SEM sobreposição com o buffer anterior
        """
        logger.info(f"📥 Coletando amostras {self.total_samples_collected + 1} até {self.total_samples_collected + 400}...")
        
        self.current_buffer = []  # Limpar buffer atual
        
        while len(self.current_buffer) < 400:
            # Puxar dados do stream
            samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=50)
            
            if samples:
                for sample in samples:
                    # Verificar se já temos 400 amostras
                    if len(self.current_buffer) >= 400:
                        break
                    
                    # Usar primeiros 16 canais ou preencher com zeros
                    if len(sample) >= 16:
                        self.current_buffer.append(sample[:16])
                    else:
                        padded = sample + [0.0] * (16 - len(sample))
                        self.current_buffer.append(padded[:16])
                    
                    self.total_samples_collected += 1
            
            # Status de progresso
            current = len(self.current_buffer)
            percentage = (current / 400) * 100
            if current % 50 == 0 or current >= 400:  # Log a cada 50 samples
                logger.info(f"   📊 Progresso: {current}/400 ({percentage:.1f}%) - Total coletado: {self.total_samples_collected}")
            
            # Se não recebeu dados, aguardar
            if not samples:
                logger.warning("⚠️ Nenhum dado recebido - aguardando...")
                time.sleep(0.1)
        
        logger.info(f"✅ Buffer completo! Amostras {self.total_samples_collected - 399} a {self.total_samples_collected}")
        return True
    
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
    
    def run_sequential_test(self, n_predictions=10):
        """
        Executar teste sequencial
        n_predictions: número de predições a fazer (cada uma usa 400 amostras sequenciais)
        """
        logger.info("🎯 SISTEMA BCI SEQUENCIAL")
        logger.info("=" * 50)
        logger.info(f"🔢 Fazendo {n_predictions} predições")
        logger.info(f"📊 Cada predição usa 400 amostras SEQUENCIAIS (sem sobreposição)")
        logger.info(f"⏱️ @ 125Hz = 3.2s por buffer")
        logger.info("=" * 50)
        
        # 1. Carregar modelo
        if not self.load_model():
            return
        
        # 2. Conectar LSL
        inlet = self.find_stream()
        if not inlet:
            return
        
        # 3. Fazer predições sequenciais
        try:
            for i in range(n_predictions):
                logger.info(f"\n🧠 PREDIÇÃO #{i+1}/{n_predictions}")
                logger.info("-" * 30)
                
                # Coletar próximas 400 amostras
                if not self.collect_next_400_samples(inlet):
                    logger.error("❌ Falha na coleta de dados")
                    break
                
                # Converter para formato do modelo (16, 400)
                data = np.array(self.current_buffer).T  # (16, 400)
                
                # Fazer predição
                prediction, confidence, proc_time = self.predict(data)
                
                # Armazenar estatísticas
                self.predictions.append(prediction)
                self.confidences.append(confidence)
                self.processing_times.append(proc_time)
                
                # Log da predição
                class_name = "Mão Direita" if prediction == 1 else "Mão Esquerda"
                start_idx = self.total_samples_collected - 399
                end_idx = self.total_samples_collected
                
                logger.info(f"🎯 Resultado: {class_name}")
                logger.info(f"📊 Confiança: {confidence:.3f}")
                logger.info(f"⚡ Processamento: {proc_time*1000:.1f}ms")
                logger.info(f"📍 Amostras usadas: {start_idx} a {end_idx}")
                
                # Aguardar um pouco antes da próxima coleta (opcional)
                if i < n_predictions - 1:
                    logger.info("⏳ Aguardando próxima coleta...")
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Teste interrompido pelo usuário")
        
        # Resultados finais
        self.print_final_results()
    
    def print_final_results(self):
        """Resultados finais"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESULTADOS FINAIS - MODO SEQUENCIAL")
        logger.info("=" * 60)
        
        if not self.predictions:
            logger.info("❌ Nenhuma predição realizada")
            return
        
        left_total = self.predictions.count(0)
        right_total = self.predictions.count(1)
        total_preds = len(self.predictions)
        
        logger.info(f"🔢 Total de predições: {total_preds}")
        logger.info(f"📥 Total de amostras coletadas: {self.total_samples_collected}")
        logger.info(f"⏱️ Tempo total de dados: {self.total_samples_collected/125:.1f}s @ 125Hz")
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
        else:
            logger.info("✅ Modelo produz predições variadas")
        
        # Mostrar sequência de predições
        logger.info(f"\n📈 Sequência de predições:")
        pred_str = ""
        for i, pred in enumerate(self.predictions):
            symbol = "L" if pred == 0 else "R"
            pred_str += f"{symbol}"
            if (i + 1) % 10 == 0:
                pred_str += " "
        logger.info(f"   {pred_str}")
        logger.info(f"   (L=Esquerda, R=Direita)")
        
        logger.info("=" * 60)

def main():
    print("""
🎯 BCI Sequencial - 400 em 400 amostras
======================================

Este script funciona EXATAMENTE como você pediu:

✅ Coleta amostras 1-400 → Predição #1
✅ Coleta amostras 401-800 → Predição #2  
✅ Coleta amostras 801-1200 → Predição #3
✅ E assim por diante...

SEM SOBREPOSIÇÃO entre buffers!
Cada buffer é completamente independente.

""")
    
    # Perguntar quantas predições fazer
    try:
        n_preds = int(input("Quantas predições fazer? (padrão: 5): ") or "5")
    except:
        n_preds = 5
    
    print(f"\nFazendo {n_preds} predições sequenciais...")
    print(f"Total de amostras necessárias: {n_preds * 400}")
    print(f"Tempo estimado: {n_preds * 3.2:.1f}s @ 125Hz")
    print("\nPressione Ctrl+C para parar a qualquer momento.\n")
    
    test = SequentialBCITest()
    test.run_sequential_test(n_predictions=n_preds)

if __name__ == "__main__":
    main()
