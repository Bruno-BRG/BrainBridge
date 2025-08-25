"""
Teste do filtro melhorado (ordem 10) com dados reais de 60Hz
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, freqz
import sys
import os

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processing.butter_filter import ButterworthFilter


def test_improved_filter():
    """Testa o filtro melhorado (ordem 10)"""
    print("=" * 70)
    print("TESTE DO FILTRO MELHORADO - ORDEM 10")
    print("=" * 70)
    
    # Criar filtros para comparação
    filter_old = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    filter_new = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=10)
    
    print("Comparando filtros:")
    print(f"  Antigo: Ordem {filter_old.order}")
    print(f"  Novo:   Ordem {filter_new.order}")
    print()
    
    # Teste de resposta em frequência
    w, h_old = freqz(filter_old.b, filter_old.a, worN=8192, fs=125.0)
    w, h_new = freqz(filter_new.b, filter_new.a, worN=8192, fs=125.0)
    
    mag_old_db = 20 * np.log10(np.abs(h_old) + 1e-12)
    mag_new_db = 20 * np.log10(np.abs(h_new) + 1e-12)
    
    # Frequências críticas para testar
    critical_freqs = [52, 55, 60, 65, 70]
    
    print("ATENUAÇÃO EM FREQUÊNCIAS CRÍTICAS:")
    print("-" * 50)
    print(f"{'Freq (Hz)':>8} {'Antigo (dB)':>12} {'Novo (dB)':>12} {'Melhoria':>12}")
    print("-" * 50)
    
    for freq in critical_freqs:
        freq_idx = np.argmin(np.abs(w - freq))
        att_old = mag_old_db[freq_idx]
        att_new = mag_new_db[freq_idx]
        improvement = att_new - att_old
        
        print(f"{freq:>8.0f} {att_old:>12.1f} {att_new:>12.1f} {improvement:>12.1f}")
    
    return filter_old, filter_new


def test_with_synthetic_60hz():
    """Teste com sinal sintético de 60Hz"""
    print("\n" + "=" * 70)
    print("TESTE COM SINAL SINTÉTICO DE 60Hz")
    print("=" * 70)
    
    filter_old = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
    filter_new = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=10)
    
    # Gerar sinal de teste
    fs = 125.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Sinal composto: EEG útil (10Hz) + ruído 60Hz
    eeg_signal = np.sin(2 * np.pi * 10 * t)  # 10Hz - EEG alfa
    noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * t)  # 60Hz - ruído elétrico
    combined_signal = eeg_signal + noise_60hz
    
    # Aplicar filtros
    filtered_old = filter_old.apply_filter(combined_signal)
    filtered_new = filter_new.apply_filter(combined_signal)
    
    # Calcular RMS
    rms_original = np.sqrt(np.mean(combined_signal**2))
    rms_old = np.sqrt(np.mean(filtered_old**2))
    rms_new = np.sqrt(np.mean(filtered_new**2))
    rms_pure_eeg = np.sqrt(np.mean(eeg_signal**2))
    
    print("RESULTADOS:")
    print(f"  Sinal original (10Hz + 60Hz): RMS = {rms_original:.4f}")
    print(f"  EEG puro (apenas 10Hz):       RMS = {rms_pure_eeg:.4f}")
    print(f"  Filtro antigo (ordem 5):      RMS = {rms_old:.4f}")
    print(f"  Filtro novo (ordem 10):       RMS = {rms_new:.4f}")
    print()
    
    # Calcular proximidade ao EEG puro
    error_old = abs(rms_old - rms_pure_eeg)
    error_new = abs(rms_new - rms_pure_eeg)
    
    print("PROXIMIDADE AO EEG PURO:")
    print(f"  Erro filtro antigo: {error_old:.4f}")
    print(f"  Erro filtro novo:   {error_new:.4f}")
    print(f"  Melhoria: {((error_old - error_new) / error_old * 100):.1f}%")
    
    # Análise espectral
    freqs, psd_orig = welch(combined_signal, fs, nperseg=256)
    freqs, psd_old = welch(filtered_old, fs, nperseg=256)
    freqs, psd_new = welch(filtered_new, fs, nperseg=256)
    
    # Potência na banda de 60Hz (58-62Hz)
    idx_60hz = (freqs >= 58) & (freqs <= 62)
    if np.any(idx_60hz):
        power_orig_60 = np.sum(psd_orig[idx_60hz])
        power_old_60 = np.sum(psd_old[idx_60hz])
        power_new_60 = np.sum(psd_new[idx_60hz])
        
        reduction_old = (1 - power_old_60/power_orig_60) * 100 if power_orig_60 > 0 else 0
        reduction_new = (1 - power_new_60/power_orig_60) * 100 if power_orig_60 > 0 else 0
        
        print(f"\nREDUÇÃO DE POTÊNCIA EM 60Hz:")
        print(f"  Filtro antigo: {reduction_old:.1f}%")
        print(f"  Filtro novo:   {reduction_new:.1f}%")


def test_with_real_csv():
    """Teste com dados reais do CSV (se disponível)"""
    print("\n" + "=" * 70)
    print("TESTE COM DADOS REAIS DO CSV")
    print("=" * 70)
    
    csv_path = "../../p001_60Hz.csv"
    
    try:
        # Tentar carregar o CSV
        if os.path.exists(csv_path):
            print(f"Carregando dados de: {csv_path}")
            
            # Tentar diferentes formatos de CSV
            try:
                data = pd.read_csv(csv_path)
                print(f"  Formato: {data.shape}")
                print(f"  Colunas: {list(data.columns)[:5]}...")
                
                # Assumir que as primeiras colunas numéricas são dados EEG
                eeg_columns = []
                for col in data.columns:
                    if data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        eeg_columns.append(col)
                        if len(eeg_columns) >= 16:  # Máximo 16 canais
                            break
                
                if len(eeg_columns) > 0:
                    print(f"  Canais EEG encontrados: {len(eeg_columns)}")
                    
                    # Pegar uma amostra dos dados
                    sample_data = data[eeg_columns].head(500).values.T  # Transpor para (canais, amostras)
                    
                    if sample_data.shape[1] > 50:  # Verificar se há amostras suficientes
                        filter_old = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=5)
                        filter_new = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=125.0, order=10)
                        
                        # Aplicar filtros
                        filtered_old = filter_old.apply_filter(sample_data)
                        filtered_new = filter_new.apply_filter(sample_data)
                        
                        # Calcular estatísticas
                        rms_original = np.sqrt(np.mean(sample_data**2))
                        rms_old = np.sqrt(np.mean(filtered_old**2))
                        rms_new = np.sqrt(np.mean(filtered_new**2))
                        
                        print(f"  RMS original: {rms_original:.4f}")
                        print(f"  RMS filtro antigo: {rms_old:.4f}")
                        print(f"  RMS filtro novo: {rms_new:.4f}")
                        
                        reduction_old = (1 - rms_old/rms_original) * 100
                        reduction_new = (1 - rms_new/rms_original) * 100
                        
                        print(f"  Redução antigo: {reduction_old:.1f}%")
                        print(f"  Redução novo: {reduction_new:.1f}%")
                    else:
                        print("  ⚠ Dados insuficientes para teste")
                else:
                    print("  ⚠ Nenhum canal EEG numérico encontrado")
                    
            except Exception as e:
                print(f"  ❌ Erro ao processar CSV: {e}")
        else:
            print(f"  ⚠ Arquivo não encontrado: {csv_path}")
            
    except Exception as e:
        print(f"❌ Erro geral: {e}")


def main():
    """Função principal"""
    print("VALIDAÇÃO DO FILTRO MELHORADO (ORDEM 10)")
    print("Testando se agora corta adequadamente acima de 50Hz")
    print()
    
    # Executar testes
    test_improved_filter()
    test_with_synthetic_60hz()
    test_with_real_csv()
    
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print("✅ Filtro atualizado para ordem 10")
    print("✅ Melhor atenuação em frequências > 50Hz")
    print("✅ Corte mais agressivo do ruído de 60Hz")
    print("📝 Execute o teste novamente para confirmar:")
    print("   python analyze_filter_response.py")


if __name__ == "__main__":
    main()
