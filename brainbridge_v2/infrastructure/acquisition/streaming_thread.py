import time
import json
import numpy as np
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal
from brainbridge_v2.infrastructure.acquisition.udp_receiver import UDPReceiver_BCI

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
        
        self.sample_rate = 125  # Source must be configured for Cyton + Daisy.
        self.last_error = None
        
        # Inicialização do modelo
        self.model = None
        self.window_size = 250  # 2s @ 125Hz
        self.samples_since_last_prediction = 0
        self.predictions = deque(maxlen=50)  # Últimas predições
        self.eeg_buffer = deque(maxlen=1000)  # Buffer para dados EEG
        self.game_mode = False  # Flag para modo jogo
        
    def start_streaming(self, host='localhost', port=12345):
        """Inicia o streaming"""
        self.host = host
        self.port = port
        self.last_error = None
        self.is_mock_mode = False
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
            # UDPReceiver_BCI.start logs errors instead of raising them.
            if not self.udp_receiver.is_running:
                raise RuntimeError(f"UDP receiver failed to start on {self.host}:{self.port}")
            
            self.connection_status.emit(True)
            self.is_mock_mode = False
            
            # Manter thread viva
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            self.last_error = str(e)
            print(f"Erro no streaming UDP: {e}")
        
        finally:
            self.is_running = False
            if self.udp_receiver:
                self.udp_receiver.stop()
                # Failed startup leaves a socket open because stop returns early.
                if self.udp_receiver.socket is not None:
                    self.udp_receiver.socket.close()
                    self.udp_receiver.socket = None
            self.connection_status.emit(False)
    
    def extract_eeg_from_udp(self, data):
        """Return all raw 16-channel samples; filtering belongs to consumers.

        UDP values are expected in microvolts at 125 Hz. Packet arrival timing
        is not a device sample clock, and runtime electrode montage is unknown.
        """
        return self._extract_raw_eeg_from_udp(data)
    
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
                    timeseries = np.asarray(data['data'], dtype=float)
                    if timeseries.ndim == 2 and timeseries.shape[0] == 16:
                        return [sample.copy() for sample in timeseries.T]
                
                # Formato direto por canais
                elif 'Ch1' in data:
                    values = [data[f'Ch{ch}'] for ch in range(1, 17)]
                    if any(isinstance(value, list) for value in values):
                        return self._extract_raw_eeg_from_udp(
                            {'type': 'timeSeriesRaw', 'data': values}
                        )
                    if all(value is not None for value in values):
                        return np.asarray(values, dtype=float)
                
                # Formato com channels
                elif 'channels' in data:
                    return self._extract_raw_eeg_from_udp(data['channels'])
            
            # Se é lista, assumir que são os 16 canais
            elif isinstance(data, list) and len(data) == 16:
                sample = np.asarray(data, dtype=float)
                if sample.shape == (16,):
                    return sample
            
            return None
            
        except Exception as e:
            print(f"Erro ao extrair EEG bruto: {e}")
            return None
