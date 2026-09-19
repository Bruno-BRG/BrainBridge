"""Thread de streaming EEG sem dependencias de GUI.

Substitui o antigo `QThread` do PyQt5 (a interface Qt foi removida; o
frontend agora e o app Tauri). A API publica foi preservada para o
`EEGStreamGatewayAdapter` e para os testes:

- `data_received` / `connection_status`: sinais com `connect`/`emit`.
- `start_streaming(host, port)` / `stop_streaming()` / `run()`.
- `isRunning()` / `is_running` / `is_mock_mode` / `sample_rate`.
"""

import json
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from brainbridge_v2.infrastructure.acquisition.udp_receiver import UDPReceiver_BCI


class Signal:
    """Sinal minimo compativel com o uso que o adapter fazia do pyqtSignal.

    `connect` aceita (e ignora) argumentos extras como o tipo de conexao
    do Qt, para nao quebrar chamadores legados.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []

    def connect(self, callback: Callable, *args, **kwargs) -> Callable:
        with self._lock:
            self._subscribers.append(callback)
        return callback

    def disconnect(self, callback: Callable) -> None:
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def emit(self, *args, **kwargs) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                print(f"Erro em callback de streaming: {exc}")


class StreamingThread:
    """Thread para streaming de dados (threading pura, sem Qt)."""

    def __init__(self, parent=None):
        self._parent = parent  # compatibilidade com a antiga assinatura QThread
        self.data_received = Signal()
        self.connection_status = Signal()
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
        self.host = "localhost"
        self.port = 12345
        self._worker: threading.Thread | None = None

    def isRunning(self) -> bool:
        worker = self._worker
        return bool(self.is_running and worker is not None and worker.is_alive())

    def start_streaming(self, host='localhost', port=12345):
        """Inicia o streaming"""
        self.host = host
        self.port = port
        self.last_error = None
        self.is_mock_mode = False
        self.is_running = True
        worker = threading.Thread(target=self.run, daemon=True)
        self._worker = worker
        worker.start()

    def stop_streaming(self):
        """Para o streaming"""
        self.is_running = False
        if self.udp_receiver:
            try:
                self.udp_receiver.stop()
            except Exception:
                pass
        worker, self._worker = self._worker, None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)

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
