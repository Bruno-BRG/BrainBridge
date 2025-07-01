#!/usr/bin/env python3
"""
🎯 BCI com Aprendizado por Reforço - Sequencial T0/T1/T2

PROTOCOLO DE REFORÇO:
✅ Streaming sequencial 400 em 400 (sem sobreposição)
✅ Usa sequência conhecida T0→T1→T0→T2→T0 (padrão das gravações)
✅ REFORÇA apenas quando:
   - Predição é T1 ou T2 (ignora T0/repouso)
   - Confiança > 60%
✅ Usa o label correto baseado na posição na sequência conhecida
✅ Atualiza modelo online com gradiente descendente

SEQUÊNCIA PADRÃO (15 ciclos):
T0(repouso) → T1(mão_esq) → T0(repouso) → T2(mão_dir) → T0(repouso) → repeat...
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
from pathlib import Path
from collections import deque
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verificar LSL
try:
    from pylsl import StreamInlet, resolve_streams, resolve_byprop
    logger.info("✅ pylsl disponível")
except ImportError:
    logger.error("❌ Instale: pip install pylsl")
    sys.exit(1)

# ============================================================================
# MODELOS (COPIADO DO SEQUENTIAL_BCI_TEST.PY)
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
# CLASSE BCI COM APRENDIZADO POR REFORÇO
# ============================================================================

class ReinforcementBCISystem:
    def __init__(self, confidence_threshold=0.6, learning_rate=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.normalizer_stats = None
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        
        # Buffer para coleta sequencial
        self.current_buffer = []
        
        # PROTOCOLO DE SEQUÊNCIA T0/T1/T2 (padrão das gravações)
        # Cada ciclo: T0 → T1 → T0 → T2 → T0 (5 buffers por ciclo)
        self.t_sequence = [0, 1, 0, 2, 0]  # T0=repouso, T1=mão_esq, T2=mão_dir
        self.current_t_index = 0  # Posição atual na sequência
        self.cycle_count = 0  # Quantos ciclos completos
        
        # Estatísticas de reforço
        self.total_predictions = 0
        self.reinforcement_attempts = 0
        self.successful_reinforcements = 0
        self.t1_reinforcements = 0
        self.t2_reinforcements = 0
        
        # Armazenar dados para análise
        self.predictions_log = []
        self.confidences_log = []
        self.true_labels_log = []
        self.reinforced_log = []
        
        # Contadores gerais
        self.total_samples_collected = 0
        
        logger.info(f"🖥️ Device: {self.device}")
        logger.info(f"🎯 Threshold de confiança: {confidence_threshold*100}%")
        logger.info(f"📚 Learning rate: {learning_rate}")
        logger.info("🔄 Modo: REFORÇO com sequência T0→T1→T0→T2→T0")
    
    def get_current_true_label(self):
        """
        Retorna o label verdadeiro baseado na posição atual na sequência
        T0=repouso (não usado para reforço), T1=0 (mão esq), T2=1 (mão dir)
        """
        current_t = self.t_sequence[self.current_t_index]
        
        if current_t == 0:  # T0 - repouso
            return None  # Não fazemos reforço para repouso
        elif current_t == 1:  # T1 - mão esquerda
            return 0
        elif current_t == 2:  # T2 - mão direita  
            return 1
        
        return None
    
    def advance_sequence(self):
        """Avança para próxima posição na sequência T"""
        self.current_t_index += 1
        
        # Se completou um ciclo (5 posições), resetar
        if self.current_t_index >= len(self.t_sequence):
            self.current_t_index = 0
            self.cycle_count += 1
            logger.info(f"🔄 Ciclo {self.cycle_count} completado! Reiniciando sequência T0→T1→T0→T2→T0")
    
    def get_current_t_info(self):
        """Retorna informações sobre o T atual"""
        current_t = self.t_sequence[self.current_t_index]
        t_names = {0: "T0(repouso)", 1: "T1(mão_esq)", 2: "T2(mão_dir)"}
        
        position_in_cycle = self.current_t_index + 1
        return current_t, t_names[current_t], position_in_cycle
    
    def load_model(self):
        """Carregar modelo e configurar optimizer para reforço"""
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
            
            # IMPORTANTE: Modelo deve estar em modo eval para predições (evita erro BatchNorm)
            # Só colocamos em train mode durante o gradient update
            self.model.eval()
            
            # Configurar optimizer para aprendizado online
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Carregar estatísticas de normalização
            if 'normalization_stats' in checkpoint:
                self.normalizer_stats = checkpoint['normalization_stats']
                logger.info("✅ Normalizador carregado")
            
            # Info do modelo
            if 'test_accuracy' in checkpoint:
                logger.info(f"📊 Acurácia base: {checkpoint['test_accuracy']:.3f}")
            
            logger.info("✅ Modelo carregado e pronto para REFORÇO!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False
    
    def find_stream(self):
        """Encontrar stream LSL (mesmo código do sequential)"""
        logger.info("🔍 Procurando streams LSL...")
        
        streams = []
        for stream_type in ['EEG', 'ExG', 'EMG']:
            try:
                found = resolve_byprop('type', stream_type, timeout=2)
                streams.extend(found)
                if found:
                    logger.info(f"  📡 {len(found)} stream(s) {stream_type}")
            except:
                pass
        
        try:
            general = resolve_streams(wait_time=3)
            streams.extend(general)
        except:
            pass
        
        if not streams:
            logger.error("❌ Nenhum stream encontrado!")
            return None
        
        stream = streams[0]
        logger.info(f"🔗 Conectando: {stream.name()} ({stream.channel_count()} canais)")
        
        inlet = StreamInlet(stream, max_chunklen=1)
        return inlet
    
    def normalize_data(self, data):
        """Normalizar dados (mesmo código do sequential)"""
        if self.normalizer_stats and 'median' in self.normalizer_stats and 'iqr' in self.normalizer_stats:
            try:
                median = self.normalizer_stats['median']
                iqr = self.normalizer_stats['iqr']
                
                if hasattr(median, 'numpy'):
                    median = median.numpy()
                if hasattr(iqr, 'numpy'):
                    iqr = iqr.numpy()
                
                if median.ndim > 2:
                    median = median.squeeze()
                if iqr.ndim > 2:
                    iqr = iqr.squeeze()
                
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
        """Coletar próximas 400 amostras sequenciais (mesmo código do sequential)"""
        logger.info(f"📥 Coletando amostras {self.total_samples_collected + 1} até {self.total_samples_collected + 400}...")
        
        self.current_buffer = []
        
        while len(self.current_buffer) < 400:
            samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=50)
            
            if samples:
                for sample in samples:
                    if len(self.current_buffer) >= 400:
                        break
                    
                    if len(sample) >= 16:
                        self.current_buffer.append(sample[:16])
                    else:
                        padded = sample + [0.0] * (16 - len(sample))
                        self.current_buffer.append(padded[:16])
                    
                    self.total_samples_collected += 1
            
            current = len(self.current_buffer)
            if current % 50 == 0 or current >= 400:
                percentage = (current / 400) * 100
                logger.info(f"   📊 Progresso: {current}/400 ({percentage:.1f}%)")
            
            if not samples:
                logger.warning("⚠️ Nenhum dado recebido - aguardando...")
                time.sleep(0.1)
        
        logger.info(f"✅ Buffer completo!")
        return True
    
    def predict_and_reinforce(self, eeg_data):
        """
        Fazer predição E aplicar reforço se critérios forem atendidos
        """
        # 1. PREDIÇÃO
        start_time = time.time()
        
        # Normalizar
        normalized = self.normalize_data(eeg_data)
        logger.info(f"🔍 Debug - EEG data shape: {eeg_data.shape}")
        logger.info(f"🔍 Debug - Normalized shape: {normalized.shape}")
        
        # Para tensor - garantir formato correto (batch, channels, height, width)
        # normalized tem shape (16, 400), precisamos (1, 1, 16, 400)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        logger.info(f"🔍 Debug - Tensor shape: {tensor.shape}")
        
        # Forward pass em modo eval para BatchNorm (mas mantendo gradientes)
        self.model.eval()  # Coloca BatchNorm em eval mode
        with torch.no_grad():  # Desabilita gradientes apenas para predição
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
        
        processing_time = time.time() - start_time
        
        # 2. VERIFICAR SE DEVE FAZER REFORÇO
        true_label = self.get_current_true_label()
        reinforced = False
        
        # Critérios para reforço:
        # - True label não é None (ou seja, não é T0/repouso)
        # - Confiança > threshold
        if true_label is not None and confidence > self.confidence_threshold:
            # 3. APLICAR REFORÇO - criar novo tensor sem no_grad para gradientes
            train_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(self.device)
            reinforced = self.apply_reinforcement(train_tensor, true_label)
        
        # 4. LOG E ESTATÍSTICAS
        self.total_predictions += 1
        self.predictions_log.append(prediction)
        self.confidences_log.append(confidence)
        self.true_labels_log.append(true_label)
        self.reinforced_log.append(reinforced)
        
        if reinforced:
            self.successful_reinforcements += 1
            if true_label == 0:  # T1 - mão esquerda
                self.t1_reinforcements += 1
            elif true_label == 1:  # T2 - mão direita
                self.t2_reinforcements += 1
        
        return prediction, confidence, processing_time, reinforced, true_label
    
    def apply_reinforcement(self, tensor, true_label):
        """
        Aplicar reforço usando backpropagation
        """
        try:
            self.reinforcement_attempts += 1
            
            logger.info(f"🔍 Debug - Tensor shape no reforço: {tensor.shape}")
            
            # SOLUÇÃO PARA BATCH SIZE: usar eval mode para BatchNorm durante forward pass
            # e apenas fazer update dos gradientes
            self.model.eval()  # Manter BatchNorm em eval mode
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass com gradientes habilitados mas BatchNorm em eval mode
            tensor.requires_grad_(True)
            output = self.model(tensor)
            
            # Calcular loss
            true_tensor = torch.tensor([true_label], dtype=torch.long, device=self.device)
            loss = nn.CrossEntropyLoss()(output, true_tensor)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            logger.info(f"🎯 REFORÇO aplicado! Loss: {loss.item():.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no reforço: {e}")
            return False
    
    def run_reinforcement_session(self, n_cycles=3):
        """
        Executar sessão com aprendizado por reforço
        n_cycles: número de ciclos completos T0→T1→T0→T2→T0 (5 predições por ciclo)
        """
        total_predictions = n_cycles * 5
        
        logger.info("🧠 SISTEMA BCI COM APRENDIZADO POR REFORÇO")
        logger.info("=" * 70)
        logger.info(f"🔢 Executando {n_cycles} ciclos ({total_predictions} predições)")
        logger.info(f"📊 Sequência: T0→T1→T0→T2→T0 (repouso→esq→repouso→dir→repouso)")
        logger.info(f"🎯 Reforço apenas em T1/T2 com confiança > {self.confidence_threshold*100}%")
        logger.info(f"📚 Learning rate: {self.learning_rate}")
        logger.info("=" * 70)
        
        # 1. Carregar modelo
        if not self.load_model():
            return
        
        # 2. Conectar LSL
        inlet = self.find_stream()
        if not inlet:
            return
        
        # 3. Executar predições com reforço
        try:
            prediction_count = 0
            
            for cycle in range(n_cycles):
                logger.info(f"\n🔄 CICLO {cycle + 1}/{n_cycles}")
                logger.info("=" * 50)
                
                # 5 predições por ciclo (T0→T1→T0→T2→T0)
                for step in range(5):
                    prediction_count += 1
                    current_t, t_name, position = self.get_current_t_info()
                    
                    logger.info(f"\n🧠 PREDIÇÃO #{prediction_count}/{total_predictions}")
                    logger.info(f"📍 Posição: {position}/5 no ciclo {cycle + 1}")
                    logger.info(f"🏷️ Esperado: {t_name}")
                    logger.info("-" * 40)
                    
                    # Coletar dados
                    if not self.collect_next_400_samples(inlet):
                        logger.error("❌ Falha na coleta")
                        return
                    
                    # Converter para formato do modelo
                    data = np.array(self.current_buffer).T  # (16, 400)
                    
                    # Predição + Reforço
                    pred, conf, proc_time, reinforced, true_label = self.predict_and_reinforce(data)
                    
                    # Log da predição
                    pred_name = "Mão Esquerda" if pred == 0 else "Mão Direita"
                    logger.info(f"🎯 Predição: {pred_name}")
                    logger.info(f"📊 Confiança: {conf:.3f}")
                    logger.info(f"⚡ Processamento: {proc_time*1000:.1f}ms")
                    
                    # Log do reforço
                    if current_t == 0:  # T0 - repouso
                        logger.info(f"😴 T0 (repouso) - SEM reforço")
                    else:
                        if reinforced:
                            logger.info(f"✅ REFORÇO aplicado! (confiança {conf:.3f} > {self.confidence_threshold})")
                        else:
                            logger.info(f"❌ Sem reforço (confiança {conf:.3f} < {self.confidence_threshold})")
                    
                    # Avançar na sequência
                    self.advance_sequence()
                    
                    # Pequena pausa
                    if prediction_count < total_predictions:
                        time.sleep(0.5)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Sessão interrompida pelo usuário")
        
        # Resultados finais
        self.print_reinforcement_results()
    
    def print_reinforcement_results(self):
        """Imprimir resultados da sessão de reforço"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 RESULTADOS DA SESSÃO DE APRENDIZADO POR REFORÇO")
        logger.info("=" * 80)
        
        if not self.predictions_log:
            logger.info("❌ Nenhuma predição realizada")
            return
        
        # Estatísticas gerais
        total_preds = len(self.predictions_log)
        left_preds = self.predictions_log.count(0)
        right_preds = self.predictions_log.count(1)
        
        logger.info(f"🔢 Total de predições: {total_preds}")
        logger.info(f"📥 Amostras coletadas: {self.total_samples_collected}")
        logger.info(f"⏱️ Tempo de dados: {self.total_samples_collected/125:.1f}s @ 125Hz")
        logger.info(f"🔄 Ciclos completados: {self.cycle_count}")
        
        # Distribuição de predições
        logger.info(f"\n📈 Distribuição de predições:")
        logger.info(f"👈 Mão Esquerda: {left_preds} ({left_preds/total_preds*100:.1f}%)")
        logger.info(f"👉 Mão Direita: {right_preds} ({right_preds/total_preds*100:.1f}%)")
        
        # Estatísticas de reforço
        logger.info(f"\n🎯 Estatísticas de reforço:")
        logger.info(f"🔄 Tentativas de reforço: {self.reinforcement_attempts}")
        logger.info(f"✅ Reforços bem-sucedidos: {self.successful_reinforcements}")
        logger.info(f"👈 Reforços T1 (mão esq): {self.t1_reinforcements}")
        logger.info(f"👉 Reforços T2 (mão dir): {self.t2_reinforcements}")
        
        if self.reinforcement_attempts > 0:
            success_rate = (self.successful_reinforcements / self.reinforcement_attempts) * 100
            logger.info(f"📊 Taxa de sucesso do reforço: {success_rate:.1f}%")
        
        # Confiança média
        if self.confidences_log:
            avg_conf = np.mean(self.confidences_log)
            min_conf = np.min(self.confidences_log)
            max_conf = np.max(self.confidences_log)
            logger.info(f"\n🎯 Confiança: {avg_conf:.3f} (min: {min_conf:.3f}, max: {max_conf:.3f})")
            
            # Confiança por tipo de predição
            above_threshold = sum(1 for c in self.confidences_log if c > self.confidence_threshold)
            logger.info(f"📈 Predições > threshold ({self.confidence_threshold*100}%): {above_threshold}/{total_preds} ({above_threshold/total_preds*100:.1f}%)")
        
        # Análise por tipo de T
        logger.info(f"\n📊 Análise por tipo de evento:")
        t0_count = t1_count = t2_count = 0
        t0_reinforced = t1_reinforced = t2_reinforced = 0
        
        for i, (true_label, reinforced) in enumerate(zip(self.true_labels_log, self.reinforced_log)):
            if true_label is None:  # T0
                t0_count += 1
            elif true_label == 0:  # T1
                t1_count += 1
                if reinforced:
                    t1_reinforced += 1
            elif true_label == 1:  # T2
                t2_count += 1
                if reinforced:
                    t2_reinforced += 1
        
        logger.info(f"😴 T0 (repouso): {t0_count} eventos (reforço não aplicado)")
        if t1_count > 0:
            logger.info(f"👈 T1 (mão esq): {t1_count} eventos, {t1_reinforced} reforçados ({t1_reinforced/t1_count*100:.1f}%)")
        if t2_count > 0:
            logger.info(f"👉 T2 (mão dir): {t2_count} eventos, {t2_reinforced} reforçados ({t2_reinforced/t2_count*100:.1f}%)")

def main():
    print("""
🧠 BCI com Aprendizado por Reforço
==================================

PROTOCOLO:
✅ Streaming sequencial 400 em 400 (SEM sobreposição)  
✅ Sequência conhecida: T0→T1→T0→T2→T0 (padrão das gravações)
✅ Reforço APENAS quando:
   - Evento é T1 ou T2 (ignora T0/repouso)
   - Confiança > 60%
✅ Usa label correto baseado na posição na sequência
✅ Atualiza modelo online com gradiente descendente

EXEMPLO DE USO:
- 3 ciclos = 15 predições
- Cada ciclo: T0→T1→T0→T2→T0
- Reforço aplicado apenas em T1/T2 com alta confiança
""")
    
    # Criar sistema
    bci = ReinforcementBCISystem(confidence_threshold=0.6, learning_rate=1e-5)
    
    # Executar sessão (3 ciclos = 15 predições)
    bci.run_reinforcement_session(n_cycles=3)

if __name__ == "__main__":
    main()
