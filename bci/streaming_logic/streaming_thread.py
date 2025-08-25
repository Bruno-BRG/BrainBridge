import time
import socket
import json
import numpy as np
import torch
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal
from ..network.udp_receiver_BCI import UDPReceiver_BCI
from ..signal_processing.butter_filter import ButterworthFilter

class StreamingThread(QThread):
    """Thread para streaming de dados"""
    
    data_received = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self.udp_receiver = None
        self.data_queue = deque(maxlen=100)
        self.is_mock_mode = False
        
        # Inicialização do filtro Butterworth
        self.butter_filter = ButterworthFilter(
            lowcut=0.5,    # 0.5 Hz - remove artefatos de movimento
            highcut=50.0,  # 50 Hz - remove ruído elétrico
            fs=125.0,      # 125 Hz - frequência de amostragem padrão OpenBCI
            order=6        # Ordem 6 - filtros em cascata (3+3) para estabilidade
        )
        
        # Inicialização do modelo
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = 400  # 3.2s @ 125Hz
        self.samples_since_last_prediction = 0
        self.predictions = deque(maxlen=50)  # Últimas predições
        self.eeg_buffer = deque(maxlen=1000)  # Buffer para dados EEG
        self.game_mode = False  # Flag para modo jogo
        
    def start_streaming(self, host='localhost', port=12345):
        """Inicia o streaming"""
        self.host = host
        self.port = port
        self.is_running = True
        self.start()
        
    def stop_streaming(self):
        """Para o streaming"""
        self.is_running = False
        if self.udp_receiver:
            self.udp_receiver.stop()
        self.quit()
        self.wait()
        
    def run(self):
        """Executa o streaming"""
        try:
            # Tentar configurar receptor UDP
            self.udp_receiver = UDPReceiver_BCI(self.host, self.port)
            
            # Callback para dados recebidos
            def on_data_received(data):
                try:
                    # Processar dados UDP para extrair EEG
                    eeg_data = self.extract_eeg_from_udp(data)
                    if eeg_data is not None:
                        # Se é uma lista de amostras, processar cada uma
                        if isinstance(eeg_data, list):
                            for sample in eeg_data:
                                self.data_received.emit(sample)
                        else:
                            # Se é uma única amostra
                            self.data_received.emit(eeg_data)
                except Exception as e:
                    print(f"Erro ao processar dados: {e}")
            
            self.udp_receiver.set_callback(on_data_received)
            self.udp_receiver.start()
            
            self.connection_status.emit(True)
            self.is_mock_mode = False
            
            # Manter thread viva
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Erro no streaming UDP: {e}")
            print("Iniciando modo de simulação...")
            self.is_mock_mode = True
            self.connection_status.emit(True)
            
            # Modo simulação - gerar dados fake
            while self.is_running:
                try:
                    # Simular dados EEG (16 canais)
                    fake_data = np.random.randn(16) * 50 + np.sin(time.time() * 2 * np.pi * 0.5) * 20
                    self.data_received.emit(fake_data)
                    time.sleep(1/125)  # Simular 125 Hz
                except Exception as e:
                    print(f"Erro na simulação: {e}")
                    break
        
        finally:
            if self.udp_receiver:
                self.udp_receiver.stop()
            self.connection_status.emit(False)
    
    def extract_eeg_from_udp(self, data):
        """Extrai dados EEG do formato UDP e aplica filtro Butterworth"""
        try:
            raw_eeg_data = self._extract_raw_eeg_from_udp(data)
            
            if raw_eeg_data is None:
                return None
            
            # Aplicar filtro Butterworth aos dados extraídos
            try:
                if isinstance(raw_eeg_data, list):
                    # Lista de amostras - filtrar cada uma
                    filtered_samples = []
                    for sample in raw_eeg_data:
                        if len(sample) == 16:  # Verificar se tem 16 canais
                            filtered_sample = self.butter_filter.apply_realtime_filter(sample)
                            filtered_samples.append(filtered_sample)
                        else:
                            filtered_samples.append(sample)  # Manter original se não tem 16 canais
                    return filtered_samples
                else:
                    # Amostra única - filtrar diretamente
                    if len(raw_eeg_data) == 16:  # Verificar se tem 16 canais
                        return self.butter_filter.apply_realtime_filter(raw_eeg_data)
                    else:
                        return raw_eeg_data  # Manter original se não tem 16 canais
                        
            except Exception as filter_error:
                print(f"Erro ao aplicar filtro Butterworth: {filter_error}")
                # Em caso de erro no filtro, retornar dados sem filtrar
                return raw_eeg_data
            
        except Exception as e:
            print(f"Erro ao extrair EEG: {e}")
            return None
    
    def _extract_raw_eeg_from_udp(self, data):
        """Extrai dados EEG brutos do formato UDP (sem filtro)"""
        try:
            # Se os dados são string, tentar converter para JSON
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return None
            
            # Se é um dicionário
            if isinstance(data, dict):
                # Formato timeSeriesRaw
                if 'type' in data and data['type'] == 'timeSeriesRaw' and 'data' in data:
                    timeseries = data['data']
                    if len(timeseries) >= 16:
                        # Processar todas as amostras (5 por canal)
                        all_samples = []
                        
                        # Determinar o número de amostras (assumindo que todos os canais têm o mesmo)
                        num_samples = len(timeseries[0]) if len(timeseries[0]) > 0 else 0
                        
                        # Para cada amostra temporal
                        for sample_idx in range(num_samples):
                            eeg_sample = []
                            for ch in range(16):
                                if sample_idx < len(timeseries[ch]):
                                    eeg_sample.append(timeseries[ch][sample_idx])
                                else:
                                    eeg_sample.append(0.0)
                            all_samples.append(np.array(eeg_sample))
                        
                        return all_samples  # Retorna lista de arrays
                
                # Formato direto por canais
                elif 'Ch1' in data:
                    eeg_sample = []
                    for ch in range(1, 17):
                        ch_key = f'Ch{ch}'
                        if ch_key in data:
                            value = data[ch_key]
                            if isinstance(value, list) and len(value) > 0:
                                eeg_sample.append(value[-1])
                            else:
                                eeg_sample.append(float(value) if value is not None else 0.0)
                        else:
                            eeg_sample.append(0.0)
                    return np.array(eeg_sample)
                
                # Formato com channels
                elif 'channels' in data:
                    return self._extract_raw_eeg_from_udp(data['channels'])
            
            # Se é lista, assumir que são os 16 canais
            elif isinstance(data, list) and len(data) >= 16:
                return np.array(data[:16])
            
            return None
            
        except Exception as e:
            print(f"Erro ao extrair EEG bruto: {e}")
            return None

