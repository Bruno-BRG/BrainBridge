"""
Análise detalhada da resposta em frequência do filtro Butterworth
Verificando se está cortando adequadamente acima de 50Hz
"""

import numpy as np
from scipy.signal import freqz, welch
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


def analyze_filter_response():
    """Análise detalhada da resposta em frequência"""
    print("=" * 70)
    print("ANÁLISE DETALHADA DA RESPOSTA EM FREQUÊNCIA")
    print("=" * 70)
    
    # Criar filtro atual
    butter_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    
    # Calcular resposta em frequência com alta resolução
    w, h = freqz(butter_filter.b, butter_filter.a, worN=8192, fs=125.0)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)  # Evitar log(0)
    
    print(f"Filtro atual: {butter_filter.get_filter_info()['type']}")
    print(f"Parâmetros: {butter_filter.lowcut}-{butter_filter.highcut}Hz, ordem {butter_filter.order}")
    print()
    
    # Testar frequências específicas em detalhes
    test_frequencies = [
        # Frequências baixas (devem ser atenuadas)
        0.1, 0.2, 0.3, 0.4, 0.5,
        # Banda passante (devem passar)
        1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0,
        # Frequências altas (devem ser MUITO atenuadas)
        52.0, 55.0, 60.0, 65.0, 70.0, 80.0, 100.0, 120.0
    ]
    
    print("RESPOSTA EM FREQUÊNCIA DETALHADA:")
    print("-" * 70)
    print(f"{'Frequência':>10} {'Magnitude':>12} {'Status':>20} {'Esperado':>20}")
    print("-" * 70)
    
    problems = []
    
    for freq in test_frequencies:
        freq_idx = np.argmin(np.abs(w - freq))
        attenuation = magnitude_db[freq_idx]
        
        # Determinar status esperado e real
        if freq < 0.5:
            expected = "ATENUADA (<-20dB)"
            status = "✓ ATENUADA" if attenuation < -20 else "❌ INSUFICIENTE"
            if attenuation >= -20:
                problems.append(f"Frequência {freq}Hz: {attenuation:.1f}dB (deveria ser <-20dB)")
        elif freq > 50.0:
            expected = "MUITO ATENUADA (<-40dB)"
            status = "✓ MUITO ATENUADA" if attenuation < -40 else "❌ INSUFICIENTE"
            if attenuation >= -40:
                problems.append(f"Frequência {freq}Hz: {attenuation:.1f}dB (deveria ser <-40dB)")
        else:
            expected = "PASSA (>-3dB)"
            status = "✓ PASSA" if attenuation > -3 else "⚠ ATENUADA"
            if attenuation <= -3:
                problems.append(f"Frequência {freq}Hz: {attenuation:.1f}dB (deveria ser >-3dB)")
        
        print(f"{freq:>10.1f} {attenuation:>12.1f} dB {status:>20} {expected:>20}")
    
    return problems, butter_filter


def test_high_frequency_rejection():
    """Teste específico para rejeição de altas frequências"""
    print("\n" + "=" * 70)
    print("TESTE ESPECÍFICO: REJEIÇÃO DE ALTAS FREQUÊNCIAS")
    print("=" * 70)
    
    butter_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    
    # Gerar sinais de teste em diferentes frequências
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    high_frequencies = [55, 60, 65, 70, 80, 100, 120]
    
    print("Testando atenuação de frequências específicas:")
    print("-" * 50)
    print(f"{'Freq (Hz)':>8} {'RMS Original':>12} {'RMS Filtrado':>12} {'Atenuação':>12}")
    print("-" * 50)
    
    insufficient_attenuation = []
    
    for freq in high_frequencies:
        if freq > fs/2:  # Não testar frequências acima de Nyquist
            continue
            
        # Gerar sinal puro na frequência
        signal = np.sin(2 * np.pi * freq * t)
        
        # Aplicar filtro
        filtered = butter_filter.apply_filter(signal)
        
        # Calcular RMS
        rms_original = np.sqrt(np.mean(signal**2))
        rms_filtered = np.sqrt(np.mean(filtered**2))
        
        # Calcular atenuação em dB
        if rms_filtered > 0:
            attenuation_db = 20 * np.log10(rms_filtered / rms_original)
        else:
            attenuation_db = -120  # Muito baixo
        
        print(f"{freq:>8.0f} {rms_original:>12.4f} {rms_filtered:>12.4f} {attenuation_db:>12.1f} dB")
        
        # Verificar se a atenuação é suficiente (>40dB para freq > 50Hz)
        if attenuation_db > -40:
            insufficient_attenuation.append((freq, attenuation_db))
    
    return insufficient_attenuation


def suggest_better_filter():
    """Sugerir melhorias no filtro"""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO COM FILTROS MELHORADOS")
    print("=" * 70)
    
    filters_to_test = [
        ("Atual (ordem 5)", 5),
        ("Melhorado (ordem 8)", 8),
        ("Agressivo (ordem 10)", 10)
    ]
    
    test_freq = 60.0  # Frequência crítica para testar
    
    print(f"Testando atenuação em {test_freq}Hz:")
    print("-" * 50)
    print(f"{'Filtro':>20} {'Atenuação (dB)':>15} {'Status':>15}")
    print("-" * 50)
    
    best_filter = None
    best_attenuation = 0
    
    for filter_name, order in filters_to_test:
        try:
            test_filter = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=order)
            
            # Calcular resposta em frequência
            w, h = freqz(test_filter.b, test_filter.a, worN=8192, fs=125.0)
            
            # Encontrar atenuação em 60Hz
            freq_idx = np.argmin(np.abs(w - test_freq))
            attenuation = 20 * np.log10(np.abs(h[freq_idx]) + 1e-12)
            
            status = "✓ BOM" if attenuation < -40 else "⚠ INSUFICIENTE"
            
            print(f"{filter_name:>20} {attenuation:>15.1f} {status:>15}")
            
            if attenuation < best_attenuation:
                best_attenuation = attenuation
                best_filter = (filter_name, order, attenuation)
                
        except Exception as e:
            print(f"{filter_name:>20} {'ERRO':>15} {str(e)[:15]:>15}")
    
    return best_filter


def main():
    """Função principal"""
    print("DIAGNÓSTICO COMPLETO DO FILTRO BUTTERWORTH")
    print("Verificando se corta adequadamente acima de 50Hz")
    print()
    
    # Análise da resposta em frequência
    problems, current_filter = analyze_filter_response()
    
    # Teste de rejeição de altas frequências
    insufficient_att = test_high_frequency_rejection()
    
    # Sugestão de melhorias
    best_filter = suggest_better_filter()
    
    # Resumo dos problemas encontrados
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO FINAL")
    print("=" * 70)
    
    if problems:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for problem in problems:
            print(f"   • {problem}")
    else:
        print("✅ Resposta em frequência adequada")
    
    if insufficient_att:
        print(f"\n❌ ATENUAÇÃO INSUFICIENTE em altas frequências:")
        for freq, att in insufficient_att:
            print(f"   • {freq}Hz: apenas {att:.1f}dB (deveria ser <-40dB)")
    else:
        print("✅ Atenuação adequada em altas frequências")
    
    # Recomendações
    print("\n📋 RECOMENDAÇÕES:")
    
    if problems or insufficient_att:
        print("   1. ❗ O filtro atual (ordem 5) é INSUFICIENTE")
        print("   2. 🔧 AUMENTAR a ordem do filtro para 8 ou 10")
        print("   3. ⚡ Considerar filtro Chebyshev Type II para melhor rejeição")
        
        if best_filter:
            name, order, att = best_filter
            print(f"   4. 💡 Melhor opção: {name} (atenuação: {att:.1f}dB)")
    else:
        print("   ✅ Filtro atual está adequado")
    
    return len(problems) == 0 and len(insufficient_att) == 0


if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n🚨 AÇÃO NECESSÁRIA: Melhorar o filtro para maior atenuação!")
    exit(0 if success else 1)
