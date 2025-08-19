"""
Serviço de Streaming de Dados - Seguindo DIP e SRP

Este serviço depende de abstrações e tem uma única responsabilidade:
gerenciar o streaming de dados EEG em tempo real.
"""

from typing import Dict, Any, Optional, Callable, List
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
import socket

from ..interfaces.service_interfaces import IDataStreamingService
from ..interfaces.communication_interfaces import IDataReceiver, INetworkCommunicator
from ..interfaces.data_interfaces import IDataLogger


class DataStreamingService(IDataStreamingService):
    """
    Serviço de streaming de dados seguindo DIP e SRP
    
    Responsabilidade única: Gerenciar streaming de dados EEG
    Depende de abstrações (IDataReceiver, IDataLogger)
    """
    
    def __init__(self, 
                 data_receiver: Optional[IDataReceiver] = None,
                 data_logger: Optional[IDataLogger] = None,
                 buffer_size: int = 1000):
        """
        Inicializa o serviço com dependências injetadas
        
        Args:
            data_receiver: Receptor de dados
            data_logger: Logger de dados
            buffer_size: Tamanho do buffer de dados
        """
        self._data_receiver = data_receiver
        self._data_logger = data_logger
        self._buffer_size = buffer_size
        
        # Estado do streaming
        self._is_streaming = False
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Buffer de dados
        self._data_buffer = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Callbacks para novos dados
        self._data_callbacks: List[Callable[[np.ndarray], None]] = []
        
        # Estatísticas
        self._samples_received = 0
        self._start_time: Optional[datetime] = None
        
        # Configuração de simulação
        self._simulation_mode = False
        self._sampling_rate = 250  # Hz
        self._num_channels = 16
    
    def start_streaming(self, config: Dict[str, Any]) -> bool:
        """
        Inicia o streaming de dados
        
        Args:
            config: Configuração do streaming
            
        Returns:
            True se iniciado com sucesso
        """
        if self._is_streaming:
            return False
        
        try:
            # Configurar parâmetros
            self._sampling_rate = config.get('sampling_rate', 250)
            self._num_channels = config.get('num_channels', 16)
            
            # Tentar iniciar receptor de dados
            if self._data_receiver:
                receiver_started = self._data_receiver.start_receiving()
                if receiver_started:
                    # Subscrever aos dados
                    self._data_receiver.subscribe_to_data(self._on_data_received)
                else:
                    # Falhar para modo simulação
                    self._simulation_mode = True
            else:
                # Usar modo simulação
                self._simulation_mode = True
            
            # Iniciar thread de streaming
            self._stop_event.clear()
            self._streaming_thread = threading.Thread(target=self._streaming_loop)
            self._streaming_thread.daemon = True
            self._streaming_thread.start()
            
            # Atualizar estado
            self._is_streaming = True
            self._start_time = datetime.now()
            self._samples_received = 0
            
            return True
            
        except Exception as e:
            print(f"Erro ao iniciar streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """
        Para o streaming de dados
        
        Returns:
            True se parado com sucesso
        """
        if not self._is_streaming:
            return False
        
        try:
            # Sinalizar parada
            self._stop_event.set()
            
            # Parar receptor se disponível
            if self._data_receiver:
                self._data_receiver.stop_receiving()
            
            # Esperar thread finalizar
            if self._streaming_thread:
                self._streaming_thread.join(timeout=2.0)
            
            # Atualizar estado
            self._is_streaming = False
            
            return True
            
        except Exception as e:
            print(f"Erro ao parar streaming: {e}")
            return False
    
    def is_streaming(self) -> bool:
        """Verifica se está fazendo streaming"""
        return self._is_streaming
    
    def get_latest_data(self) -> Optional[np.ndarray]:
        """
        Retorna os dados mais recentes
        
        Returns:
            Array com dados mais recentes ou None
        """
        with self._buffer_lock:
            if self._data_buffer:
                return np.array(list(self._data_buffer))
            return None
    
    def subscribe_to_data(self, callback: Callable[[np.ndarray], None]) -> bool:
        """
        Subscreve a callback para novos dados
        
        Args:
            callback: Função a ser chamada com novos dados
            
        Returns:
            True se subscrito com sucesso
        """
        try:
            if callback not in self._data_callbacks:
                self._data_callbacks.append(callback)
            return True
        except Exception:
            return False
    
    def unsubscribe_from_data(self, callback: Callable[[np.ndarray], None]) -> bool:
        """
        Remove subscrição de callback
        
        Args:
            callback: Função a ser removida
            
        Returns:
            True se removida com sucesso
        """
        try:
            if callback in self._data_callbacks:
                self._data_callbacks.remove(callback)
            return True
        except Exception:
            return False
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do streaming
        
        Returns:
            Dicionário com estatísticas
        """
        if not self._start_time:
            return {}
        
        duration = (datetime.now() - self._start_time).total_seconds()
        
        return {
            'is_streaming': self._is_streaming,
            'simulation_mode': self._simulation_mode,
            'duration_seconds': duration,
            'samples_received': self._samples_received,
            'sampling_rate': self._sampling_rate,
            'num_channels': self._num_channels,
            'buffer_size': len(self._data_buffer),
            'avg_samples_per_second': self._samples_received / duration if duration > 0 else 0
        }
    
    def _streaming_loop(self) -> None:
        """Loop principal do streaming"""
        while not self._stop_event.is_set():
            try:
                if self._simulation_mode:
                    # Gerar dados simulados
                    self._generate_simulated_data()
                
                # Pequena pausa para não sobrecarregar CPU
                time.sleep(1.0 / self._sampling_rate)
                
            except Exception as e:
                print(f"Erro no loop de streaming: {e}")
                break
    
    def _generate_simulated_data(self) -> None:
        """Gera dados simulados para teste"""
        # Gerar dados aleatórios simulando EEG
        data = np.random.normal(0, 10, self._num_channels)
        
        # Adicionar algumas variações realistas
        t = time.time()
        for i in range(self._num_channels):
            # Adicionar componentes de frequência típicas do EEG
            data[i] += 5 * np.sin(2 * np.pi * 10 * t + i)  # 10 Hz alpha
            data[i] += 2 * np.sin(2 * np.pi * 20 * t + i)  # 20 Hz beta
        
        # Processar dados
        self._on_data_received(data.tobytes())
    
    def _on_data_received(self, raw_data: bytes) -> None:
        """
        Callback chamado quando novos dados são recebidos
        
        Args:
            raw_data: Dados brutos recebidos
        """
        try:
            # Converter dados brutos para numpy array
            if self._simulation_mode:
                # Para simulação, assumir que os dados já estão processados
                data = np.frombuffer(raw_data, dtype=np.float64)
            else:
                # Para dados reais, implementar parsing específico
                data = self._parse_real_data(raw_data)
            
            # Adicionar ao buffer
            with self._buffer_lock:
                self._data_buffer.append(data)
            
            # Atualizar estatísticas
            self._samples_received += 1
            
            # Log se logger disponível
            if self._data_logger and self._data_logger.is_logging():
                self._data_logger.log_data({
                    'sample_index': self._samples_received,
                    'channels': data.tolist(),
                    'timestamp': datetime.now()
                })
            
            # Notificar callbacks
            for callback in self._data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Erro em callback de dados: {e}")
                    
        except Exception as e:
            print(f"Erro ao processar dados recebidos: {e}")
    
    def _parse_real_data(self, raw_data: bytes) -> np.ndarray:
        """
        Parse de dados reais do receptor
        
        Args:
            raw_data: Dados brutos
            
        Returns:
            Array numpy com dados processados
        """
        # Implementar parsing específico baseado no protocolo usado
        # Por exemplo, para dados OpenBCI:
        try:
            # Assumir formato simples: 16 floats de 4 bytes cada
            if len(raw_data) >= self._num_channels * 4:
                data = np.frombuffer(raw_data[:self._num_channels * 4], dtype=np.float32)
                return data
            else:
                # Preencher com zeros se dados insuficientes
                return np.zeros(self._num_channels)
        except Exception:
            return np.zeros(self._num_channels)
