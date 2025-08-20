"""
Testes para o filtro Butterworth bandpass (0.5-50Hz)

Este módulo testa a funcionalidade completa do filtro Butterworth,
incluindo filtragem em lote e tempo real.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


class TestButterworthFilter:
    """Testes para ButterworthFilter"""
    
    @pytest.fixture
    def filter_instance(self):
        """Cria uma instância padrão do filtro"""
        return ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    
    @pytest.fixture
    def test_signal(self):
        """Gera sinal de teste com múltiplas frequências"""
        fs = 125.0
        duration = 4.0  # 4 segundos
        t = np.linspace(0, duration, int(duration * fs))
        
        # Sinal composto:
        # - 10Hz: frequência alfa (deve passar)
        # - 0.2Hz: deriva/movimento (deve ser removido)
        # - 60Hz: ruído elétrico (deve ser removido)
        # - 30Hz: beta (deve passar)
        signal = (
            1.0 * np.sin(2 * np.pi * 10 * t) +    # 10 Hz - alfa
            0.8 * np.sin(2 * np.pi * 0.2 * t) +   # 0.2 Hz - deriva
            0.5 * np.sin(2 * np.pi * 60 * t) +    # 60 Hz - ruído
            0.7 * np.sin(2 * np.pi * 30 * t)      # 30 Hz - beta
        )
        
        # Adicionar ruído gaussiano
        signal += 0.1 * np.random.RandomState(42).randn(len(signal))
        
        return signal, fs, t
    
    def test_filter_initialization(self):
        """Testa inicialização do filtro"""
        # Teste com parâmetros padrão
        filt = ButterworthFilter()
        assert filt.lowcut == 0.5
        assert filt.highcut == 50.0
        assert filt.fs == 125.0
        assert filt.order == 5
        
        # Teste com parâmetros customizados
        filt_custom = ButterworthFilter(lowcut=1.0, highcut=50.0, fs=250.0, order=6)
        assert filt_custom.lowcut == 1.0
        assert filt_custom.highcut == 50.0
        assert filt_custom.fs == 250.0
        assert filt_custom.order == 6
    
    def test_invalid_frequencies(self):
        """Testa tratamento de frequências inválidas"""
        # Frequência muito baixa
        with pytest.raises(ValueError):
            ButterworthFilter(lowcut=0.0, highcut=50.0, fs=125.0)
        
        # Frequência muito alta
        with pytest.raises(ValueError):
            ButterworthFilter(lowcut=0.5, highcut=70.0, fs=125.0)
    
    def test_bandpass_filtering(self, filter_instance, test_signal):
        """Testa filtragem bandpass"""
        signal, fs, t = test_signal
        
        # Aplicar filtro
        filtered = filter_instance.apply_filter(signal)
        
        # Verificar que o sinal foi filtrado
        assert len(filtered) == len(signal)
        assert isinstance(filtered, np.ndarray)
        
        # Análise espectral
        freqs_orig, psd_orig = welch(signal, fs, nperseg=512)
        freqs_filt, psd_filt = welch(filtered, fs, nperseg=512)
        
        # Verificar atenuação de frequências baixas (< 0.5 Hz)
        low_freq_mask = freqs_filt < 0.5
        if np.any(low_freq_mask):
            low_power_orig = np.mean(psd_orig[low_freq_mask])
            low_power_filt = np.mean(psd_filt[low_freq_mask])
            assert low_power_filt < low_power_orig * 0.5  # Pelo menos 50% de atenuação
        
        # Verificar atenuação de frequências altas (> 50 Hz)
        high_freq_mask = freqs_filt > 50.0
        if np.any(high_freq_mask):
            high_power_orig = np.mean(psd_orig[high_freq_mask])
            high_power_filt = np.mean(psd_filt[high_freq_mask])
            assert high_power_filt < high_power_orig * 0.1  # Pelo menos 90% de atenuação
        
        # Verificar preservação de frequências válidas (5-45 Hz)
        valid_freq_mask = (freqs_filt >= 5.0) & (freqs_filt <= 45.0)
        if np.any(valid_freq_mask):
            valid_power_orig = np.mean(psd_orig[valid_freq_mask])
            valid_power_filt = np.mean(psd_filt[valid_freq_mask])
            assert valid_power_filt > valid_power_orig * 0.3  # Manter pelo menos 30%
    
    def test_multichannel_filtering(self, filter_instance):
        """Testa filtragem multicanal"""
        # Gerar dados de 16 canais
        fs = 125.0
        duration = 2.0
        n_samples = int(duration * fs)
        n_channels = 16
        
        # Cada canal com frequência diferente
        data = np.zeros((n_channels, n_samples))
        t = np.linspace(0, duration, n_samples)
        
        for ch in range(n_channels):
            freq = 10 + ch  # 10-25 Hz (dentro da banda passante)
            data[ch, :] = np.sin(2 * np.pi * freq * t)
        
        # Aplicar filtro
        filtered = filter_instance.apply_filter(data)
        
        # Verificar formato
        assert filtered.shape == data.shape
        assert isinstance(filtered, np.ndarray)
        
        # Verificar que cada canal foi processado
        for ch in range(n_channels):
            assert not np.array_equal(data[ch, :], filtered[ch, :])
    
    def test_realtime_filtering(self, filter_instance):
        """Testa filtragem em tempo real"""
        # Simular stream de dados
        fs = 125.0
        n_channels = 16
        n_samples_per_chunk = 10
        n_chunks = 50
        
        filter_instance.reset_filter_state()
        
        filtered_chunks = []
        
        for chunk in range(n_chunks):
            # Gerar chunk de dados
            t_start = chunk * n_samples_per_chunk / fs
            t_end = (chunk + 1) * n_samples_per_chunk / fs
            t = np.linspace(t_start, t_end, n_samples_per_chunk)
            
            # Sinal de teste: 10Hz + ruído
            data_chunk = np.zeros((n_channels, n_samples_per_chunk))
            for ch in range(n_channels):
                data_chunk[ch, :] = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples_per_chunk)
            
            # Aplicar filtro
            filtered_chunk = filter_instance.apply_realtime_filter(data_chunk)
            filtered_chunks.append(filtered_chunk)
            
            # Verificar formato
            assert filtered_chunk.shape == data_chunk.shape
        
        # Verificar continuidade do filtro
        assert len(filtered_chunks) == n_chunks
    
    def test_single_sample_realtime(self, filter_instance):
        """Testa filtragem de amostras únicas em tempo real"""
        n_channels = 16
        n_samples = 100
        
        filter_instance.reset_filter_state()
        
        for sample_idx in range(n_samples):
            # Gerar amostra única (16 canais)
            sample = np.random.randn(n_channels)
            
            # Aplicar filtro
            filtered_sample = filter_instance.apply_realtime_filter(sample)
            
            # Verificar formato
            assert filtered_sample.shape == sample.shape
            assert len(filtered_sample) == n_channels
    
    def test_filter_info(self, filter_instance):
        """Testa recuperação de informações do filtro"""
        info = filter_instance.get_filter_info()
        
        # Verificar estrutura do dicionário
        expected_keys = [
            'type', 'order', 'lowcut_hz', 'highcut_hz', 
            'sampling_rate_hz', 'normalized_frequencies',
            'coefficients_b', 'coefficients_a'
        ]
        
        for key in expected_keys:
            assert key in info
        
        # Verificar valores
        assert info['type'] == 'Butterworth Bandpass'
        assert info['order'] == 5
        assert info['lowcut_hz'] == 0.5
        assert info['highcut_hz'] == 50.0
        assert info['sampling_rate_hz'] == 125.0
    
    def test_insufficient_data_handling(self, filter_instance):
        """Testa tratamento de dados insuficientes"""
        # Dados muito pequenos para filtrar
        small_data = np.random.randn(16, 5)  # Apenas 5 amostras
        
        # Deve retornar dados sem filtrar (com warning)
        with pytest.warns(UserWarning):
            result = filter_instance.apply_filter(small_data)
        
        assert result.shape == small_data.shape
    
    def test_filter_state_reset(self, filter_instance):
        """Testa reset do estado do filtro"""
        # Aplicar algumas amostras
        for _ in range(10):
            sample = np.random.randn(16)
            filter_instance.apply_realtime_filter(sample)
        
        # Verificar que o estado foi inicializado
        assert filter_instance.initialized is True
        assert filter_instance.zi is not None
        
        # Reset
        filter_instance.reset_filter_state()
        
        # Verificar que o estado foi resetado
        assert filter_instance.initialized is False
        assert filter_instance.zi is None


def test_frequency_response_visualization():
    """Testa e visualiza a resposta em frequência do filtro"""
    from scipy.signal import freqz
    
    # Criar filtro
    butter_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    
    # Calcular resposta em frequência
    w, h = freqz(butter_filter.b, butter_filter.a, worN=2048, fs=125.0)
    
    # Verificar atenuação nas frequências corretas
    # Frequência de 0.1 Hz (deve ser atenuada)
    freq_01hz_idx = np.argmin(np.abs(w - 0.1))
    attenuation_01hz = 20 * np.log10(np.abs(h[freq_01hz_idx]))
    assert attenuation_01hz < -20  # Pelo menos 20dB de atenuação
    
    # Frequência de 10 Hz (deve passar)
    freq_10hz_idx = np.argmin(np.abs(w - 10.0))
    attenuation_10hz = 20 * np.log10(np.abs(h[freq_10hz_idx]))
    assert attenuation_10hz > -3  # Menos de 3dB de atenuação
    
    # Frequência de 60 Hz (deve ser atenuada)
    freq_60hz_idx = np.argmin(np.abs(w - 60.0))
    attenuation_60hz = 20 * np.log10(np.abs(h[freq_60hz_idx]))
    assert attenuation_60hz < -20  # Pelo menos 20dB de atenuação


if __name__ == "__main__":
    # Executar testes básicos
    print("Executando testes do filtro Butterworth...")
    
    # Teste manual rápido
    filter_test = ButterworthFilter()
    
    # Gerar sinal de teste
    fs = 125.0
    t = np.linspace(0, 2, int(2 * fs))
    signal = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz válido
             0.5 * np.sin(2 * np.pi * 0.1 * t) +  # 0.1 Hz inválido
             0.3 * np.sin(2 * np.pi * 60 * t))  # 60 Hz inválido
    
    # Aplicar filtro
    filtered = filter_test.apply_filter(signal)
    
    print(f"Sinal original - RMS: {np.sqrt(np.mean(signal**2)):.3f}")
    print(f"Sinal filtrado - RMS: {np.sqrt(np.mean(filtered**2)):.3f}")
    print(f"Redução de potência: {(1 - np.sqrt(np.mean(filtered**2))/np.sqrt(np.mean(signal**2))) * 100:.1f}%")
    
    # Teste tempo real
    print("\nTestando tempo real...")
    filter_test.reset_filter_state()
    
    for i in range(5):
        sample = np.random.randn(16)
        filtered_sample = filter_test.apply_realtime_filter(sample)
        print(f"Amostra {i+1}: {sample.shape} -> {filtered_sample.shape}")
    
    print("Testes concluídos com sucesso!")
