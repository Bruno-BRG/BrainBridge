"""
🧠 SISTEMA BCI COMPLETO E MINIMALISTA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UDP → Processamento → CNN → Predição
Tudo integrado em um único arquivo!
"""

import torch
import torch.nn as nn
import numpy as np
import socket
import json
import threading
import time
import os
from collections import deque
from datetime import datetime
import logging

# Configurar logging minimalista
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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

class MinimalBCI:
    """Sistema BCI ultra-minimalista"""
    
    def __init__(self, model_path="models/best_model.pth"):
        # Configurações
        self.model_path = model_path
        self.window_size = 400  # 3.2s @ 125Hz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Buffers
        self.eeg_buffer = deque(maxlen=1000)  # Buffer maior
        self.predictions = deque(maxlen=50)  # Últimas predições
        
        # Controle de predições - só prediz a cada 400 amostras
        self.samples_since_last_prediction = 0
        
        # Salvar dados das janelas
        self.save_windows = True
        self.window_counter = 0
        
        # Estado
        self.model = None
        self.socket = None
        self.running = False
        self.stats = {'predictions': 0, 'udp_packets': 0}
        
        # Thread locks
        self.buffer_lock = threading.Lock()
        
        # Estatísticas dos dados brutos
        self.samples_received = 0
        self.n_channels = 16  # Número de canais esperados
        self.max_buffer_size = 1000  # Tamanho máximo do buffer de dados brutos
        
    def load_model(self):
        """Carrega modelo CNN"""
        try:
            # Tentar vários caminhos possíveis
            possible_paths = [
                self.model_path,
                f"models/{os.path.basename(self.model_path)}",
                f"../models/{os.path.basename(self.model_path)}",
                os.path.join(os.getcwd(), "models", os.path.basename(self.model_path))
            ]
            
            model_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    model_found = True
                    break
            
            if not model_found:
                logger.error(f"❌ Modelo não encontrado em nenhum dos caminhos:")
                for path in possible_paths:
                    logger.error(f"   • {path}")
                return False
            
            self.model = EEGNet(n_channels=16, n_classes=2, n_samples=self.window_size)
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Carregar pesos, seja direto ou de state_dict
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
                
            self.model.to(self.device).eval()
            logger.info(f"✅ Modelo carregado: {self.model_path}")
            logger.info(f"✅ Device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def start_udp(self):
        """Inicia receptor UDP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('localhost', 12345))  # Porta original
            self.socket.settimeout(1.0)
            self.running = True
            
            # Thread para receber dados
            threading.Thread(target=self._udp_loop, daemon=True).start()
            logger.info("📡 UDP receptor iniciado na porta 12345")
            return True
        except Exception as e:
            logger.error(f"❌ Erro UDP: {e}")
            return False
    
    def _udp_loop(self):
        """Loop principal UDP"""
        while self.running:
            try:
                data, _ = self.socket.recvfrom(4096)
                self._process_udp(data.decode('utf-8'))
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Erro UDP: {e}")
    
    def _process_udp(self, data_str):
        """Processa dados UDP"""
        try:
            data = json.loads(data_str)
            self.stats['udp_packets'] += 1
            
            # Formato OpenBCI GUI: {'type': 'timeSeriesRaw', 'data': [[ch1...], [ch2...], ...]}
            if data.get('type') == 'timeSeriesRaw' and 'data' in data:
                channel_arrays = data['data']
                
                if len(channel_arrays) >= 16:  # Precisa de pelo menos 16 canais
                    num_samples = len(channel_arrays[0])
                    
                    # Converter para formato (samples, channels)
                    for sample_idx in range(num_samples):
                        eeg_sample = []
                        for ch_idx in range(16):  # 16 canais EEG
                            eeg_sample.append(float(channel_arrays[ch_idx][sample_idx]))
                        
                        # Adicionar ao buffer
                        with self.buffer_lock:
                            self.eeg_buffer.append(eeg_sample)
                            self.samples_since_last_prediction += 1
                        
                        # Verificar se deve fazer predição (apenas a cada 400 amostras)
                        if (len(self.eeg_buffer) >= self.window_size and 
                            self.samples_since_last_prediction >= self.window_size):
                            self._predict()
                            self.samples_since_last_prediction = 0
                            
        except Exception as e:
            logger.error(f"Erro processamento: {e}")
    
    def _process_raw_data(self, data):
        """Processa e armazena dados EEG brutos."""
        if len(data) != self.n_channels:
            logger.warning(f"⚠️ Dados recebidos com {len(data)} canais, esperados {self.n_channels}")
            return
            
        # Verificar valores extremos
        if np.any(np.abs(data) > 1e6):  # Detectar artefatos grandes
            logger.warning("⚠️ Valores extremos detectados nos dados")
            return
            
        # Calcular estatísticas dos dados brutos
        ch_means = np.mean(data)
        ch_stds = np.std(data)
        if self.samples_received % 100 == 0:  # Log a cada 100 amostras
            logger.debug(f"📊 Stats: mean={ch_means:.2f}, std={ch_stds:.2f}")
            
        # Armazenar dados no buffer
        with self.buffer_lock:
            self.eeg_buffer.append(data)
            if len(self.eeg_buffer) > self.max_buffer_size:
                self.eeg_buffer.popleft()
                
        self.samples_received += 1
        self.samples_since_last_prediction += 1
    
    def _predict(self):
        """Faz predição CNN"""
        if self.model is None:
            return
            
        try:
            # Copiar dados do buffer
            with self.buffer_lock:
                if len(self.eeg_buffer) < self.window_size:
                    return
                window_data = list(self.eeg_buffer)[-self.window_size:]
            
            # Converter para numpy array (400, 16)
            eeg_array = np.array(window_data)
            
            # Normalização por canal (mesmo método do treinamento)
            for ch in range(16):
                channel_data = eeg_array[:, ch]
                q75, q25 = np.percentile(channel_data, [75, 25])
                iqr = q75 - q25
                if iqr == 0:  # Evitar divisão por zero
                    iqr = 1.0
                channel_mean = np.mean(channel_data)
                eeg_array[:, ch] = (channel_data - channel_mean) / iqr
            
            # Transpor para (16, 400) e criar tensor
            eeg_array = eeg_array.T
            eeg_tensor = torch.FloatTensor(eeg_array).unsqueeze(0).unsqueeze(0)
            eeg_tensor = eeg_tensor.to(self.device)
            
            # Predição
            with torch.no_grad():
                output = self.model(eeg_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                
                # Log detalhado das probabilidades
                left_prob = probs[0][0].item()
                right_prob = probs[0][1].item()
            
            # Resultado com timestamp preciso
            timestamp = datetime.now()
            classes = ['🤚 Mão Esquerda', '✋ Mão Direita']
            result = {
                'class': pred,
                'name': classes[pred],
                'confidence': conf,
                'time': timestamp.strftime('%H:%M:%S.%f')[:-3]  # Incluir milissegundos
            }
            
            # Atualizar estatísticas
            self.predictions.append((timestamp, pred, conf))
            self.stats['predictions'] += 1
            
            # Log resultado detalhado
            logger.info(f"🧠 [{result['time']}] {result['name']} "
                      f"(L: {left_prob:.2%} | R: {right_prob:.2%}) "
                      f"[#{self.stats['predictions']} | {self.samples_since_last_prediction} samples]")
            
        except Exception as e:
            logger.error(f"Erro predição: {e}")
    
    def _save_window_data(self, window_data):
        """Salva dados da janela de 400 amostras"""
        try:
            # Criar diretório se não existir
            windows_dir = "windows_data"
            if not os.path.exists(windows_dir):
                os.makedirs(windows_dir)
            
            # Nome do arquivo com timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"window_{self.window_counter:04d}_{timestamp}.csv"
            filepath = os.path.join(windows_dir, filename)
            
            # Converter para numpy array
            window_array = np.array(window_data)  # (400, 16)
            
            # Salvar CSV com header e formato decimal (sem notação científica)
            header = [f"Ch{i+1}" for i in range(16)]
            np.savetxt(filepath, window_array, delimiter=',', header=','.join(header), 
                      comments='', fmt='%.6f')  # 6 casas decimais, formato decimal
            
            # Incrementar contador
            self.window_counter += 1
            
            # Log (só a cada 5 janelas para não poluir)
            if self.window_counter % 5 == 0:
                logger.info(f"💾 Janela #{self.window_counter} salva: {filename}")
                
        except Exception as e:
            logger.error(f"Erro ao salvar janela: {e}")
    
    def get_stats(self):
        """Estatísticas do sistema"""
        with self.buffer_lock:
            buffer_size = len(self.eeg_buffer)
            
        # Contadores das últimas predições
        if self.predictions:
            left_count = sum(1 for p in self.predictions if p == 0)
            right_count = len(self.predictions) - left_count
        else:
            left_count = right_count = 0
            
        return {
            'buffer': buffer_size,
            'total_predictions': self.stats['predictions'],
            'udp_packets': self.stats['udp_packets'],
            'recent_left': left_count,
            'recent_right': right_count,
            'samples_since_last': self.samples_since_last_prediction,
            'windows_saved': self.window_counter
        }
    
    def print_prediction_stats(self):
        """Imprime estatísticas das predições."""
        if not self.predictions:
            logger.info("❌ Nenhuma predição feita ainda")
            return
            
        # Calcular estatísticas básicas
        total_preds = len(self.predictions)
        last_30s = [p for p in self.predictions if (datetime.now() - p[0]).total_seconds() <= 30]
        preds_30s = len(last_30s)
        
        # Contar classes
        left_count = sum(1 for _, pred, _ in self.predictions if pred == 0)
        right_count = sum(1 for _, pred, _ in self.predictions if pred == 1)
        
        # Contar transições
        transitions = 0
        for i in range(1, len(self.predictions)):
            if self.predictions[i][1] != self.predictions[i-1][1]:
                transitions += 1
                
        # Calcular confiança média
        avg_conf = np.mean([conf for _, _, conf in self.predictions])
        
        # Imprimir relatório
        logger.info("\n=== 📊 Estatísticas de Predição ===")
        logger.info(f"Total de predições: {total_preds}")
        logger.info(f"Predições nos últimos 30s: {preds_30s}")
        logger.info(f"Taxa de predição: {preds_30s/30:.1f} pred/s")
        logger.info(f"Distribuição: 🤚 {left_count} ({left_count/total_preds:.1%}) | "
                   f"✋ {right_count} ({right_count/total_preds:.1%})")
        logger.info(f"Transições entre classes: {transitions}")
        logger.info(f"Confiança média: {avg_conf:.1%}")
        logger.info("=================================")
    
    def stop(self):
        """Para o sistema"""
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info("🛑 Sistema parado")

def main():
    """Sistema BCI principal"""
    print("🧠 SISTEMA BCI MINIMALISTA")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📡 UDP → 🔄 Processamento → 🧠 CNN → 🎯 Predição")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Criar e iniciar sistema
    bci = MinimalBCI()
    
    if not bci.load_model():
        print("❌ Falha ao carregar modelo")
        return
        
    if not bci.start_udp():
        print("❌ Falha ao iniciar UDP")
        return
    
    print("✅ Sistema iniciado!")
    print("📊 Aguardando dados do OpenBCI GUI...")
    print("🎯 Janelas de 400 amostras (3.2s @ 125Hz)")
    print("🛑 Ctrl+C para parar")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        counter = 0
        while True:
            time.sleep(3)
            counter += 3
            
            stats = bci.get_stats()
            print(f"[{counter:03d}s] 📊 Buffer: {stats['buffer']:3d}/400 | "
                  f"Predições: {stats['total_predictions']:3d} | "
                  f"UDP: {stats['udp_packets']:4d} | "
                  f"Próx: {stats['samples_since_last']:3d}/400 | "
                  f"Salvos: {stats['windows_saved']:3d} | "
                  f"Esq: {stats['recent_left']:2d} | Dir: {stats['recent_right']:2d}")
            
            # Imprimir estatísticas de predição a cada 60 segundos
            if counter % 60 == 0:
                bci.print_prediction_stats()
            
    except KeyboardInterrupt:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        bci.stop()
        
        stats = bci.get_stats()
        print(f"📈 ESTATÍSTICAS FINAIS:")
        print(f"   • Predições totais: {stats['total_predictions']}")
        print(f"   • Pacotes UDP: {stats['udp_packets']}")
        print(f"   • Janelas salvas: {stats['windows_saved']}")
        print(f"   • Mão esquerda: {stats['recent_left']}")
        print(f"   • Mão direita: {stats['recent_right']}")
        print("✅ Sistema finalizado!")

if __name__ == "__main__":
    main()
