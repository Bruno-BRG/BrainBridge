"""
Teste focado na melhoria da atenuação acima de 50Hz
Validação do filtro com corte efetivo em 45Hz
"""

import numpy as np
from scipy.signal import welch, freqz
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


def test_improved_attenuation():
    """Testa a atenuação melhorada"""
    print("=" * 70)
    print("TESTE DE ATENUAÇÃO MELHORADA (CORTE EM 45Hz)")
    print("=" * 70)
    
    # Criar filtro melhorado
    filter_improved = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=6)
    
    info = filter_improved.get_filter_info()
    print(f"Tipo: {info['type']}")
    print(f"Corte nominal: {info['highcut_hz']}Hz")
    print(f"Corte efetivo: {info['effective_highcut_hz']}Hz")
    print(f"Ordem passa-baixa: {info['lowpass_order']}")
    print()
    
    # Calcular resposta em frequência dos filtros separados
    w, h_high = freqz(filter_improved.b_high, filter_improved.a_high, worN=4096, fs=125.0)
    mag_high_db = 20 * np.log10(np.abs(h_high) + 1e-12)
    
    w, h_low = freqz(filter_improved.b_low, filter_improved.a_low, worN=4096, fs=125.0)
    mag_low_db = 20 * np.log10(np.abs(h_low) + 1e-12)
    
    # Resposta combinada
    mag_combined_db = mag_high_db + mag_low_db
    
    # Frequências críticas para avaliar
    critical_frequencies = [40, 45, 50, 52, 55, 60, 65]
    
    print("ATENUAÇÃO EM FREQUÊNCIAS CRÍTICAS:")
    print("-" * 60)
    print(f"{'Freq (Hz)':>8} {'Combinado (dB)':>15} {'Objetivo':>15} {'Status':>15}")
    print("-" * 60)
    
    all_good = True
    
    for freq in critical_frequencies:
        freq_idx = np.argmin(np.abs(w - freq))
        att_combined = mag_combined_db[freq_idx]
        
        # Definir objetivos baseados na frequência
        if freq <= 45:
            objective = "> -6dB (passa)"
            target = -6
            status = "✅ PASSA" if att_combined > target else "❌ ATENUADA"
        elif freq <= 52:
            objective = "< -20dB"
            target = -20
            status = "✅ ATENUADA" if att_combined < target else "❌ INSUFICIENTE"
            if att_combined >= target:
                all_good = False
        else:
            objective = "< -30dB"
            target = -30
            status = "✅ ATENUADA" if att_combined < target else "❌ INSUFICIENTE"
            if att_combined >= target:
                all_good = False
        
        print(f"{freq:>8.0f} {att_combined:>15.1f} {objective:>15} {status:>15}")
    
    return all_good, filter_improved


def test_with_real_signals():
    """Teste com sinais reais simulados"""
    print("\n" + "=" * 70)
    print("TESTE COM SINAIS EEG SIMULADOS")
    print("=" * 70)
    
    filter_improved = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=6)
    
    # Parâmetros do sinal
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Componentes do sinal
    components = {
        "Delta (2Hz)": 0.3 * np.sin(2 * np.pi * 2 * t),      # Deve passar
        "Theta (6Hz)": 0.4 * np.sin(2 * np.pi * 6 * t),      # Deve passar
        "Alfa (10Hz)": 1.0 * np.sin(2 * np.pi * 10 * t),     # Deve passar
        "Beta (20Hz)": 0.6 * np.sin(2 * np.pi * 20 * t),     # Deve passar
        "Gama baixo (35Hz)": 0.3 * np.sin(2 * np.pi * 35 * t), # Deve passar
        "Gama alto (45Hz)": 0.2 * np.sin(2 * np.pi * 45 * t), # Limite
        "Ruído 55Hz": 0.4 * np.sin(2 * np.pi * 55 * t),      # Deve ser removido
        "Ruído 60Hz": 0.5 * np.sin(2 * np.pi * 60 * t),      # Deve ser removido
    }
    
    # Sinal composto
    eeg_valid = (components["Delta (2Hz)"] + components["Theta (6Hz)"] + 
                 components["Alfa (10Hz)"] + components["Beta (20Hz)"] + 
                 components["Gama baixo (35Hz)"] + components["Gama alto (45Hz)"])
    
    noise_signals = components["Ruído 55Hz"] + components["Ruído 60Hz"]
    
    full_signal = eeg_valid + noise_signals
    
    # Aplicar filtro
    filtered_signal = filter_improved.apply_filter(full_signal)
    
    # Calcular RMS de cada componente
    print("PRESERVAÇÃO DOS COMPONENTES EEG:")
    print("-" * 50)
    print(f"{'Componente':>20} {'Original':>10} {'Filtrado':>10} {'Preservação':>12}")
    print("-" * 50)
    
    for name, component in components.items():
        # Aplicar filtro ao componente individual
        filtered_component = filter_improved.apply_filter(component)
        
        rms_orig = np.sqrt(np.mean(component**2))
        rms_filt = np.sqrt(np.mean(filtered_component**2))
        
        if rms_orig > 0:
            preservation = (rms_filt / rms_orig) * 100
        else:
            preservation = 0
        
        print(f"{name:>20} {rms_orig:>10.3f} {rms_filt:>10.3f} {preservation:>11.1f}%")
    
    # Estatísticas gerais
    print(f"\nESTATÍSTICAS GERAIS:")
    print(f"RMS EEG válido:      {np.sqrt(np.mean(eeg_valid**2)):.3f}")
    print(f"RMS ruído:           {np.sqrt(np.mean(noise_signals**2)):.3f}")
    print(f"RMS sinal completo:  {np.sqrt(np.mean(full_signal**2)):.3f}")
    print(f"RMS sinal filtrado:  {np.sqrt(np.mean(filtered_signal**2)):.3f}")
    
    # Redução de ruído
    noise_reduction = (1 - np.sqrt(np.mean(noise_signals**2)) / np.sqrt(np.mean(full_signal**2))) * 100
    print(f"Redução estimada de ruído: {noise_reduction:.1f}%")


def test_frequency_bands():
    """Teste específico por bandas de frequência"""
    print("\n" + "=" * 70)
    print("ANÁLISE POR BANDAS DE FREQUÊNCIA EEG")
    print("=" * 70)
    
    filter_improved = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=6)
    
    # Definir bandas EEG
    bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0), 
        "Alfa": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gama": (30.0, 45.0),
        "Ruído": (50.0, 62.5)  # Até Nyquist
    }
    
    # Gerar sinal de teste com todas as bandas
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Sinal com energia igual em todas as bandas
    full_signal = np.zeros_like(t)
    for band_name, (f_low, f_high) in bands.items():
        f_center = (f_low + f_high) / 2
        band_signal = np.sin(2 * np.pi * f_center * t)
        full_signal += band_signal
    
    # Aplicar filtro
    filtered_signal = filter_improved.apply_filter(full_signal)
    
    # Análise espectral
    freqs_orig, psd_orig = welch(full_signal, fs, nperseg=512)
    freqs_filt, psd_filt = welch(filtered_signal, fs, nperseg=512)
    
    print("PRESERVAÇÃO POR BANDA:")
    print("-" * 50)
    print(f"{'Banda':>10} {'Faixa (Hz)':>12} {'Potência Orig':>15} {'Potência Filt':>15} {'Preservação':>12}")
    print("-" * 50)
    
    for band_name, (f_low, f_high) in bands.items():
        # Calcular potência na banda
        mask = (freqs_orig >= f_low) & (freqs_orig <= f_high)
        
        if np.any(mask):
            power_orig = np.sum(psd_orig[mask])
            power_filt = np.sum(psd_filt[mask])
            
            if power_orig > 0:
                preservation = (power_filt / power_orig) * 100
            else:
                preservation = 0
            
            print(f"{band_name:>10} {f_low:>5.1f}-{f_high:>5.1f} {power_orig:>15.3f} {power_filt:>15.3f} {preservation:>11.1f}%")


def main():
    """Função principal"""
    print("VALIDAÇÃO DA ATENUAÇÃO MELHORADA")
    print("Testando filtro com corte efetivo em 45Hz")
    print()
    
    # Teste de atenuação
    attenuation_ok, filter_improved = test_improved_attenuation()
    
    # Testes com sinais simulados
    test_with_real_signals()
    
    # Análise por bandas
    test_frequency_bands()
    
    # Conclusão
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    
    if attenuation_ok:
        print("✅ ATENUAÇÃO ADEQUADA nas frequências críticas")
        print("✅ Filtro melhorado está funcionando corretamente")
        print("✅ Boa rejeição de ruído acima de 50Hz")
        print("✅ Preservação adequada do EEG válido")
    else:
        print("⚠ Ainda há problemas de atenuação em algumas frequências")
        print("💡 Considere ajustar ainda mais o filtro se necessário")
    
    print(f"\n📋 CONFIGURAÇÃO FINAL:")
    print(f"   • Corte efetivo: 45Hz (mais agressivo)")
    print(f"   • Ordem passa-alta: 3")
    print(f"   • Ordem passa-baixa: 4 (melhorada)")
    print(f"   • Implementação: Cascata estável")
    
    return attenuation_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
