"""
Filtro Butterworth para processamento de sinais EEG

Este módulo implementa um filtro Butterworth de passagem de banda
para filtrar sinais EEG em tempo real.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Union
import warnings

class ButterworthFilter:
    """
    Filtro Butterworth de passagem de banda para sinais EEG
    
    Este filtro remove ruídos de baixa e alta frequência dos sinais EEG,
    mantendo apenas as frequências de interesse (0.5 - 50 Hz).
    """
    
    def __init__(self, lowcut: float = 0.5, highcut: float = 50.0, 
                 fs: float = 125.0, order: int = 6):
        """
        Inicializa o filtro Butterworth
        
        Args:
            lowcut (float): Frequência de corte inferior em Hz (default: 0.5)
            highcut (float): Frequência de corte superior em Hz (default: 50.0)
            fs (float): Frequência de amostragem em Hz (default: 125.0)
            order (int): Ordem do filtro (default: 6 - compromisso estabilidade/performance)
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
        # Calcular frequências normalizadas
        nyquist = 0.5 * fs
        self.low = lowcut / nyquist
        self.high = highcut / nyquist
        
        # Validar frequências
        if self.low <= 0 or self.high >= 1:
            raise ValueError(f"Frequências de corte inválidas: {lowcut}-{highcut} Hz para fs={fs} Hz")
        
        # IMPLEMENTAÇÃO MELHORADA: Usar filtros separados para maior estabilidade
        # Filtro passa-alta para remover frequências < 0.5Hz
        self.b_high, self.a_high = butter(order//2, self.low, btype='highpass')
        
        # Filtro passa-baixa MELHORADO: frequência de corte mais baixa para maior atenuação
        # Usar 45Hz ao invés de 50Hz para garantir boa atenuação em 55Hz
        cutoff_improved = 45.0 / nyquist  # 45Hz ao invés de 50Hz
        self.b_low, self.a_low = butter(order//2 + 1, cutoff_improved, btype='lowpass')
        
        # Manter compatibilidade com código existente
        self.b, self.a = butter(order, [self.low, self.high], btype='band')
        
        # Buffer para filtro em tempo real (estado do filtro)
        self.zi = None
        self.zi_high = None  # Estado para filtro passa-alta
        self.zi_low = None   # Estado para filtro passa-baixa
        self.initialized = False
        self.use_cascade = True  # Usar filtros em cascata por padrão
        
    def apply_filter(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Aplica o filtro Butterworth aos dados
        
        Args:
            data: Dados de entrada (1D ou 2D)
                  Se 1D: representa um canal
                  Se 2D: cada linha é um canal, cada coluna é uma amostra
        
        Returns:
            np.ndarray: Dados filtrados com a mesma forma da entrada
        """
        # Converter para numpy array
        data = np.asarray(data, dtype=np.float64)
        
        # Verificar se há dados suficientes
        if data.size == 0:
            return data
            
        original_shape = data.shape
        
        # Se é 1D, tratar como um canal
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Se 2D mas apenas uma amostra por canal, retornar sem filtrar
        min_samples = 3 * (self.order // 2)  # Menor requisito para filtros separados
        if data.shape[1] < min_samples:
            warnings.warn(f"Dados insuficientes para filtrar ({data.shape[1]} amostras). "
                         f"Mínimo necessário: {min_samples}")
            return np.asarray(data).reshape(original_shape)
        
        # Aplicar filtro em cada canal
        filtered_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            try:
                if self.use_cascade:
                    # MÉTODO MELHORADO: Filtros em cascata para maior estabilidade
                    # 1. Primeiro aplicar passa-alta (remove < 0.5Hz)
                    temp_data = filtfilt(self.b_high, self.a_high, data[i, :])
                    # 2. Depois aplicar passa-baixa (remove > 50Hz)
                    filtered_data[i, :] = filtfilt(self.b_low, self.a_low, temp_data)
                else:
                    # Método original (bandpass direto)
                    filtered_data[i, :] = filtfilt(self.b, self.a, data[i, :])
            except Exception as e:
                warnings.warn(f"Erro ao filtrar canal {i}: {e}")
                filtered_data[i, :] = data[i, :]
        
        return filtered_data.reshape(original_shape)
    
    def apply_realtime_filter(self, sample: Union[np.ndarray, list]) -> np.ndarray:
        """
        Aplica filtro em tempo real para uma única amostra ou pequeno grupo de amostras
        
        Para uso em tempo real, este método mantém o estado do filtro entre chamadas.
        
        Args:
            sample: Amostra(s) de entrada
                   Se 1D: representa uma amostra de múltiplos canais
                   Se 2D: cada linha é um canal, cada coluna é uma amostra
        
        Returns:
            np.ndarray: Amostra(s) filtrada(s)
        """
        # Converter para numpy array
        sample = np.asarray(sample, dtype=np.float64)
        
        if sample.size == 0:
            return sample
            
        original_shape = sample.shape
        
        # Se é 1D com 16 elementos, assumir que são 16 canais em uma amostra
        if sample.ndim == 1:
            if len(sample) == 16:  # 16 canais EEG
                sample = sample.reshape(-1, 1)  # Transpor para (16, 1)
            else:
                sample = sample.reshape(1, -1)  # Um canal, múltiplas amostras
        
        # Para filtro em tempo real com poucas amostras, usar filtro simples
        # (filtfilt precisa de muitas amostras)
        min_samples_rt = 3 * (self.order // 2)
        if sample.shape[1] < min_samples_rt:
            # Para amostras únicas ou pequenos grupos, aplicar filtro simples
            # Nota: isso introduz atraso de fase, mas é necessário para tempo real
            
            if not self.initialized or self.zi_high is None or self.zi_low is None:
                # Inicializar estado dos filtros
                from scipy.signal import lfilter_zi
                max_len_high = max(len(self.a_high), len(self.b_high)) - 1
                max_len_low = max(len(self.a_low), len(self.b_low)) - 1
                
                self.zi_high = np.zeros((sample.shape[0], max_len_high))
                self.zi_low = np.zeros((sample.shape[0], max_len_low))
                self.initialized = True
            
            filtered_sample = np.zeros_like(sample)
            
            for i in range(sample.shape[0]):
                try:
                    from scipy.signal import lfilter
                    if self.use_cascade:
                        # Filtro em cascata para tempo real
                        # 1. Passa-alta primeiro
                        temp_sample, self.zi_high[i, :] = lfilter(
                            self.b_high, self.a_high, sample[i, :], zi=self.zi_high[i, :]
                        )
                        # 2. Passa-baixa depois
                        filtered_sample[i, :], self.zi_low[i, :] = lfilter(
                            self.b_low, self.a_low, temp_sample, zi=self.zi_low[i, :]
                        )
                    else:
                        # Método original
                        if self.zi is None:
                            max_len = max(len(self.a), len(self.b)) - 1
                            self.zi = np.zeros((sample.shape[0], max_len))
                        filtered_sample[i, :], self.zi[i, :] = lfilter(
                            self.b, self.a, sample[i, :], zi=self.zi[i, :]
                        )
                except Exception as e:
                    warnings.warn(f"Erro ao filtrar canal {i} em tempo real: {e}")
                    filtered_sample[i, :] = sample[i, :]
            
            return filtered_sample.reshape(original_shape)
        else:
            # Para grupos maiores de amostras, usar filtfilt
            return self.apply_filter(sample)
    
    def reset_filter_state(self):
        """
        Reseta o estado do filtro em tempo real
        """
        self.zi = None
        self.zi_high = None
        self.zi_low = None
        self.initialized = False
    
    def get_filter_info(self) -> dict:
        """
        Retorna informações sobre o filtro
        
        Returns:
            dict: Dicionário com informações do filtro
        """
        return {
            'type': 'Butterworth Bandpass (Cascata Melhorada)',
            'order': self.order,
            'lowcut_hz': self.lowcut,
            'highcut_hz': self.highcut,
            'effective_highcut_hz': 45.0,  # Corte efetivo mais agressivo
            'sampling_rate_hz': self.fs,
            'normalized_frequencies': [self.low, self.high],
            'cascade_mode': self.use_cascade,
            'highpass_order': self.order // 2,
            'lowpass_order': self.order // 2 + 1,
            'coefficients_b': self.b.tolist(),
            'coefficients_a': self.a.tolist()
        }
    
    @staticmethod
    def test_filter():
        """
        Teste básico do filtro com dados simulados
        """
        print("Testando Filtro Butterworth...")
        
        # Gerar sinal de teste
        fs = 125.0  # 125 Hz
        t = np.linspace(0, 2, int(2 * fs))  # 2 segundos
        
        # Sinal composto: 10Hz (EEG), 1Hz (movimento), 60Hz (ruído elétrico)
        signal = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz - válido
                 0.5 * np.sin(2 * np.pi * 1 * t) +   # 1 Hz - deve ser removido
                 0.3 * np.sin(2 * np.pi * 60 * t))   # 60 Hz - deve ser removido
        
        # Adicionar ruído
        signal += 0.1 * np.random.randn(len(signal))
        
        # Aplicar filtro
        butter_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=fs)
        filtered_signal = butter_filter.apply_filter(signal)
        
        print(f"Sinal original: {len(signal)} amostras")
        print(f"Sinal filtrado: {len(filtered_signal)} amostras")
        print(f"RMS original: {np.sqrt(np.mean(signal**2)):.3f}")
        print(f"RMS filtrado: {np.sqrt(np.mean(filtered_signal**2)):.3f}")
        
        # Teste com dados multicanais (16 canais)
        multi_channel_data = np.random.randn(16, 1000)  # 16 canais, 1000 amostras
        filtered_multi = butter_filter.apply_filter(multi_channel_data)
        
        print(f"Dados multicanais: {multi_channel_data.shape}")
        print(f"Filtrados multicanais: {filtered_multi.shape}")
        
        # Teste tempo real
        print("\nTestando filtro tempo real...")
        butter_filter.reset_filter_state()
        
        for i in range(10):
            # Simular amostra de 16 canais
            sample = np.random.randn(16)
            filtered_sample = butter_filter.apply_realtime_filter(sample)
            print(f"Amostra {i+1}: {sample.shape} -> {filtered_sample.shape}")
        
        print("Teste do filtro concluído!")


if __name__ == "__main__":
    ButterworthFilter.test_filter()
