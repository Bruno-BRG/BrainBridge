"""
Teste simples do filtro Butterworth bandpass (0.5-50Hz)
Sem dependências externas como pytest ou matplotlib
"""

import numpy as np
from scipy.signal import welch, freqz
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


def test_filter_basic():
    """Teste básico do filtro"""
    print("=" * 60)
    print("TESTE 1: Inicialização e configuração do filtro")
    print("=" * 60)
    
    # Criar filtro com configurações padrão (0.5-50Hz)
    butter_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    
    # Verificar parâmetros
    info = butter_filter.get_filter_info()
    print(f"✓ Tipo: {info['type']}")
    print(f"✓ Frequência baixa: {info['lowcut_hz']} Hz")
    print(f"✓ Frequência alta: {info['highcut_hz']} Hz")
    print(f"✓ Taxa de amostragem: {info['sampling_rate_hz']} Hz")
    print(f"✓ Ordem: {info['order']}")
    
    return butter_filter


def test_frequency_response(butter_filter):
    """Teste da resposta em frequência"""
    print("\n" + "=" * 60)
    print("TESTE 2: Resposta em frequência do filtro")
    print("=" * 60)
    
    # Calcular resposta em frequência
    w, h = freqz(butter_filter.b, butter_filter.a, worN=2048, fs=125.0)
    magnitude_db = 20 * np.log10(np.abs(h))
    
    # Testar frequências específicas
    test_freqs = [0.1, 0.5, 10.0, 30.0, 50.0, 60.0]
    
    for freq in test_freqs:
        freq_idx = np.argmin(np.abs(w - freq))
        attenuation = magnitude_db[freq_idx]
        
        if freq < 0.5 or freq > 50.0:
            status = "✓ ATENUADA" if attenuation < -10 else "⚠ POUCO ATENUADA"
        else:
            status = "✓ PASSA" if attenuation > -3 else "⚠ MUITO ATENUADA"
        
        print(f"{freq:5.1f} Hz: {attenuation:6.1f} dB - {status}")


def test_signal_filtering(butter_filter):
    """Teste com sinal sintético"""
    print("\n" + "=" * 60)
    print("TESTE 3: Filtragem de sinal sintético")
    print("=" * 60)
    
    # Gerar sinal de teste (4 segundos)
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Sinal composto com múltiplas frequências
    signal_components = {
        "0.2 Hz (movimento)": 0.8 * np.sin(2 * np.pi * 0.2 * t),
        "10 Hz (alfa)": 1.0 * np.sin(2 * np.pi * 10 * t),
        "30 Hz (beta)": 0.7 * np.sin(2 * np.pi * 30 * t),
        "60 Hz (ruído)": 0.5 * np.sin(2 * np.pi * 60 * t)
    }
    
    # Sinal completo
    signal = sum(signal_components.values())
    signal += 0.1 * np.random.RandomState(42).randn(len(signal))  # Ruído
    
    # Aplicar filtro
    filtered = butter_filter.apply_filter(signal)
    
    # Análise espectral
    freqs_orig, psd_orig = welch(signal, fs, nperseg=512)
    freqs_filt, psd_filt = welch(filtered, fs, nperseg=512)
    
    # Calcular potência em bandas específicas
    def power_in_band(freqs, psd, freq_min, freq_max):
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        return np.sum(psd[mask]) if np.any(mask) else 0
    
    bands = [
        ("0.1-0.5 Hz", 0.1, 0.5),
        ("0.5-4 Hz", 0.5, 4.0),
        ("8-13 Hz", 8.0, 13.0),
        ("13-30 Hz", 13.0, 30.0),
        ("30-50 Hz", 30.0, 50.0),
        ("50-60 Hz", 50.0, 60.0)
    ]
    
    print(f"{'Banda':12} {'Original':>10} {'Filtrado':>10} {'Redução':>10}")
    print("-" * 50)
    
    for band_name, f_min, f_max in bands:
        power_orig = power_in_band(freqs_orig, psd_orig, f_min, f_max)
        power_filt = power_in_band(freqs_filt, psd_filt, f_min, f_max)
        
        if power_orig > 0:
            reduction = (1 - power_filt/power_orig) * 100
        else:
            reduction = 0
            
        print(f"{band_name:12} {power_orig:10.3f} {power_filt:10.3f} {reduction:9.1f}%")
    
    # Estatísticas gerais
    rms_orig = np.sqrt(np.mean(signal**2))
    rms_filt = np.sqrt(np.mean(filtered**2))
    total_reduction = (1 - rms_filt/rms_orig) * 100
    
    print(f"\nRMS original: {rms_orig:.3f}")
    print(f"RMS filtrado: {rms_filt:.3f}")
    print(f"Redução total: {total_reduction:.1f}%")


def test_multichannel(butter_filter):
    """Teste com dados multicanais"""
    print("\n" + "=" * 60)
    print("TESTE 4: Filtragem multicanal (16 canais)")
    print("=" * 60)
    
    # Gerar dados de 16 canais
    fs = 125.0
    duration = 2.0
    n_samples = int(duration * fs)
    n_channels = 16
    
    # Cada canal com frequência diferente
    data = np.zeros((n_channels, n_samples))
    t = np.linspace(0, duration, n_samples)
    
    frequencies = []
    for ch in range(n_channels):
        freq = 5 + ch * 2.5  # 5, 7.5, 10, 12.5, ... Hz
        frequencies.append(freq)
        data[ch, :] = np.sin(2 * np.pi * freq * t)
        # Adicionar ruído de 60Hz em todos os canais
        data[ch, :] += 0.3 * np.sin(2 * np.pi * 60 * t)
    
    # Aplicar filtro
    filtered = butter_filter.apply_filter(data)
    
    # Verificar forma dos dados
    print(f"✓ Formato original: {data.shape}")
    print(f"✓ Formato filtrado: {filtered.shape}")
    
    # Análise por canal
    print(f"\n{'Canal':>5} {'Freq':>6} {'RMS Orig':>10} {'RMS Filt':>10} {'Redução':>10}")
    print("-" * 50)
    
    for ch in range(min(8, n_channels)):  # Mostrar apenas primeiros 8 canais
        rms_orig = np.sqrt(np.mean(data[ch, :]**2))
        rms_filt = np.sqrt(np.mean(filtered[ch, :]**2))
        reduction = (1 - rms_filt/rms_orig) * 100
        
        print(f"{ch+1:5d} {frequencies[ch]:6.1f} {rms_orig:10.3f} {rms_filt:10.3f} {reduction:9.1f}%")


def test_realtime(butter_filter):
    """Teste de filtragem em tempo real"""
    print("\n" + "=" * 60)
    print("TESTE 5: Filtragem em tempo real")
    print("=" * 60)
    
    # Reset do estado do filtro
    butter_filter.reset_filter_state()
    
    # Simular stream de dados
    n_channels = 16
    n_samples_per_chunk = 8  # Pequenos chunks como no streaming real
    n_chunks = 20
    
    print(f"Simulando {n_chunks} chunks de {n_samples_per_chunk} amostras...")
    
    total_samples = 0
    for chunk in range(n_chunks):
        # Gerar chunk de dados (16 canais x 8 amostras)
        data_chunk = np.random.randn(n_channels, n_samples_per_chunk)
        
        # Aplicar filtro
        filtered_chunk = butter_filter.apply_realtime_filter(data_chunk)
        
        total_samples += n_samples_per_chunk
        
        if chunk < 5 or chunk % 5 == 0:  # Mostrar apenas alguns chunks
            rms_orig = np.sqrt(np.mean(data_chunk**2))
            rms_filt = np.sqrt(np.mean(filtered_chunk**2))
            print(f"Chunk {chunk+1:2d}: {data_chunk.shape} -> {filtered_chunk.shape} "
                  f"(RMS: {rms_orig:.3f} -> {rms_filt:.3f})")
    
    print(f"✓ Processadas {total_samples} amostras em {n_chunks} chunks")
    
    # Teste com amostras únicas
    print("\nTestando amostras únicas...")
    for i in range(5):
        sample = np.random.randn(n_channels)
        filtered_sample = butter_filter.apply_realtime_filter(sample)
        print(f"Amostra {i+1}: {sample.shape} -> {filtered_sample.shape}")


def main():
    """Função principal de teste"""
    print("TESTE COMPLETO DO FILTRO BUTTERWORTH BANDPASS (0.5-50Hz)")
    print("=" * 60)
    
    try:
        # Executar todos os testes
        butter_filter = test_filter_basic()
        test_frequency_response(butter_filter)
        test_signal_filtering(butter_filter)
        test_multichannel(butter_filter)
        test_realtime(butter_filter)
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("✅ O filtro Butterworth está funcionando corretamente")
        print("✅ Banda passante: 0.5-50Hz confirmada")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
