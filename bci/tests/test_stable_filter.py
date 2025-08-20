"""
Teste do filtro Butterworth ESTÁVEL em cascata
Valida se corta bem acima de 50Hz sem instabilidade
"""

import numpy as np
from scipy.signal import welch, freqz
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


def test_stable_filter():
    """Testa o filtro estável em cascata"""
    print("=" * 70)
    print("TESTE DO FILTRO BUTTERWORTH ESTÁVEL (CASCATA)")
    print("=" * 70)
    
    # Criar filtro estável
    filter_stable = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=6)
    
    info = filter_stable.get_filter_info()
    print(f"Tipo: {info['type']}")
    print(f"Ordem total: {info['order']}")
    print(f"Passa-alta: ordem {info['highpass_order']}")
    print(f"Passa-baixa: ordem {info['lowpass_order']}")
    print(f"Modo cascata: {info['cascade_mode']}")
    print()
    
    # Teste de estabilidade com sinal sintético
    print("TESTE DE ESTABILIDADE:")
    print("-" * 40)
    
    # Gerar sinal de teste (EEG + ruído 60Hz)
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Sinal composto
    eeg_10hz = np.sin(2 * np.pi * 10 * t)      # EEG alfa (deve passar)
    noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * t)  # Ruído elétrico (deve ser removido)
    combined = eeg_10hz + noise_60hz
    
    # Aplicar filtro
    try:
        filtered = filter_stable.apply_filter(combined)
        
        # Verificar estabilidade
        rms_original = np.sqrt(np.mean(combined**2))
        rms_filtered = np.sqrt(np.mean(filtered**2))
        rms_pure_eeg = np.sqrt(np.mean(eeg_10hz**2))
        
        print(f"RMS original (10Hz + 60Hz): {rms_original:.4f}")
        print(f"RMS EEG puro (10Hz):        {rms_pure_eeg:.4f}")
        print(f"RMS filtrado:               {rms_filtered:.4f}")
        
        # Verificar se o resultado é estável
        if rms_filtered < 10 * rms_pure_eeg:  # Deve estar próximo do EEG puro
            print("✅ FILTRO ESTÁVEL")
            stability_ok = True
        else:
            print(f"❌ FILTRO INSTÁVEL (RMS muito alto)")
            stability_ok = False
            
        # Verificar proximidade ao EEG puro
        error = abs(rms_filtered - rms_pure_eeg) / rms_pure_eeg * 100
        print(f"Erro relativo ao EEG puro: {error:.1f}%")
        
    except Exception as e:
        print(f"❌ ERRO na filtragem: {e}")
        stability_ok = False
    
    return stability_ok, filter_stable


def test_frequency_response(filter_stable):
    """Testa resposta em frequência do filtro estável"""
    print("\n" + "=" * 70)
    print("RESPOSTA EM FREQUÊNCIA DO FILTRO ESTÁVEL")
    print("=" * 70)
    
    # Calcular resposta em frequência
    w, h = freqz(filter_stable.b_high, filter_stable.a_high, worN=4096, fs=125.0)
    mag_high_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    w, h = freqz(filter_stable.b_low, filter_stable.a_low, worN=4096, fs=125.0)
    mag_low_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    # Resposta combinada (aproximada)
    mag_combined_db = mag_high_db + mag_low_db
    
    # Frequências de teste
    test_frequencies = [0.1, 0.5, 10.0, 30.0, 50.0, 55.0, 60.0, 70.0]
    
    print("ATENUAÇÃO POR FREQUÊNCIA:")
    print("-" * 50)
    print(f"{'Freq (Hz)':>8} {'Passa-Alta':>12} {'Passa-Baixa':>12} {'Combinado':>12} {'Status':>15}")
    print("-" * 50)
    
    good_attenuation = True
    
    for freq in test_frequencies:
        freq_idx = np.argmin(np.abs(w - freq))
        att_high = mag_high_db[freq_idx]
        att_low = mag_low_db[freq_idx]
        att_combined = att_high + att_low
        
        # Determinar status
        if freq < 0.5:
            status = "✓ ATENUADA" if att_combined < -20 else "⚠ POUCO"
            if att_combined >= -20:
                good_attenuation = False
        elif freq > 50.0:
            status = "✓ ATENUADA" if att_combined < -30 else "⚠ POUCO"
            if att_combined >= -30:
                good_attenuation = False
        else:
            status = "✓ PASSA" if att_combined > -6 else "⚠ ATENUADA"
        
        print(f"{freq:>8.1f} {att_high:>12.1f} {att_low:>12.1f} {att_combined:>12.1f} {status:>15}")
    
    return good_attenuation


def test_real_time_stability(filter_stable):
    """Testa estabilidade em tempo real"""
    print("\n" + "=" * 70)
    print("TESTE DE ESTABILIDADE EM TEMPO REAL")
    print("=" * 70)
    
    filter_stable.reset_filter_state()
    
    n_channels = 16
    n_samples_per_chunk = 8
    n_chunks = 50
    
    print(f"Processando {n_chunks} chunks de {n_samples_per_chunk} amostras...")
    
    rms_values = []
    stable = True
    
    for chunk in range(n_chunks):
        # Gerar chunk com sinal conhecido
        chunk_data = np.random.randn(n_channels, n_samples_per_chunk) * 0.1
        
        # Adicionar componente de 10Hz (deve passar)
        t_chunk = np.linspace(chunk * 0.064, (chunk + 1) * 0.064, n_samples_per_chunk)
        for ch in range(n_channels):
            chunk_data[ch, :] += np.sin(2 * np.pi * 10 * t_chunk)
        
        try:
            # Aplicar filtro
            filtered_chunk = filter_stable.apply_realtime_filter(chunk_data)
            
            # Verificar estabilidade
            rms_chunk = np.sqrt(np.mean(filtered_chunk**2))
            rms_values.append(rms_chunk)
            
            # Se RMS explodir, é instável
            if rms_chunk > 100:
                stable = False
                print(f"❌ Chunk {chunk}: RMS explodiu para {rms_chunk:.2f}")
                break
                
        except Exception as e:
            print(f"❌ Erro no chunk {chunk}: {e}")
            stable = False
            break
    
    if stable and len(rms_values) > 10:
        rms_mean = np.mean(rms_values)
        rms_std = np.std(rms_values)
        print(f"✅ Processamento estável")
        print(f"   RMS médio: {rms_mean:.4f}")
        print(f"   Desvio padrão: {rms_std:.4f}")
        print(f"   Coeficiente de variação: {(rms_std/rms_mean)*100:.1f}%")
    
    return stable


def main():
    """Função principal"""
    print("VALIDAÇÃO DO FILTRO BUTTERWORTH ESTÁVEL")
    print("Verificando se corta bem acima de 50Hz SEM instabilidade")
    print()
    
    # Teste de estabilidade
    stability_ok, filter_stable = test_stable_filter()
    
    if not stability_ok:
        print("\n❌ FILTRO INSTÁVEL - Interrompendo testes")
        return False
    
    # Teste de resposta em frequência
    freq_response_ok = test_frequency_response(filter_stable)
    
    # Teste de tempo real
    realtime_ok = test_real_time_stability(filter_stable)
    
    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    
    all_ok = stability_ok and freq_response_ok and realtime_ok
    
    print(f"✅ Estabilidade do filtro: {'PASSOU' if stability_ok else 'FALHOU'}")
    print(f"✅ Resposta em frequência: {'PASSOU' if freq_response_ok else 'FALHOU'}")
    print(f"✅ Tempo real: {'PASSOU' if realtime_ok else 'FALHOU'}")
    
    if all_ok:
        print("\n🎉 FILTRO VALIDADO COM SUCESSO!")
        print("   • Corta adequadamente acima de 50Hz")
        print("   • Mantém estabilidade numérica")
        print("   • Funciona bem em tempo real")
    else:
        print("\n⚠ FILTRO PRECISA DE AJUSTES")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
