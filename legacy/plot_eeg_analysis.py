import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d

def load_openbci_data(data_folder="eeg_data", n_samples_per_trial=400):
    """Simplified data loading function for visualization"""
    print("Carregando dados OpenBCI...")
    data_path = Path(data_folder)
    
    all_trials = []
    all_labels = []
    event_mapping = {'T1': 0, 'T2': 1}
    
    for subject_folder in sorted(data_path.glob('S*')):
        for csv_file in subject_folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, skiprows=4, low_memory=False)
                eeg_columns = [f'EXG Channel {i}' for i in range(16)]
                
                if not all(col in df.columns for col in eeg_columns):
                    continue
                    
                eeg_data = df[eeg_columns].values.astype(float)
                annotations = df['Annotations'].fillna('').astype(str)
                
                for idx, annotation in enumerate(annotations):
                    if annotation in event_mapping:
                        start_idx = max(0, idx - 50)
                        end_idx = min(len(eeg_data), start_idx + int(3.2 * 125))
                        
                        if end_idx - start_idx < int(3.2 * 125 * 0.8):
                            continue
                            
                        trial_data = eeg_data[start_idx:end_idx, :].T
                        if trial_data.shape[1] != n_samples_per_trial:
                            old_time = np.linspace(0, 1, trial_data.shape[1])
                            new_time = np.linspace(0, 1, n_samples_per_trial)
                            trial_resampled = np.zeros((16, n_samples_per_trial))
                            
                            for ch in range(16):
                                f = interp1d(old_time, trial_data[ch, :], kind='linear')
                                trial_resampled[ch, :] = f(new_time)
                            trial_data = trial_resampled
                        
                        all_trials.append(trial_data)
                        all_labels.append(event_mapping[annotation])
                        
            except Exception as e:
                print(f"Erro ao processar {csv_file.name}: {str(e)}")
                continue
    
    if all_trials:
        return np.array(all_trials), np.array(all_labels)
    else:
        raise ValueError("Nenhum dado carregado!")

def plot_channel_correlation(windows):
    """Plot da matriz de correlação entre canais"""
    plt.figure(figsize=(12, 10))
    
    # Calcular correlação média entre canais usando todos os trials
    mean_trial = np.mean(windows, axis=0)  # Média de todos os trials
    correlation_matrix = np.corrcoef(mean_trial)
    
    # Criar heatmap
    sns.heatmap(correlation_matrix, 
                cmap='coolwarm',
                center=0,
                vmin=-1, 
                vmax=1,
                annot=True,
                fmt='.2f',
                square=True)
    
    plt.title('Correlação entre Canais EEG')
    plt.xlabel('Número do Canal')
    plt.ylabel('Número do Canal')
    
    # Salvar figura
    plt.savefig('channel_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_t1_t2_difference(windows, labels):
    """Plot do mapa de diferença T2-T1 (Canais vs Tempo)"""
    plt.figure(figsize=(15, 8))
    
    # Calcular médias para cada classe
    mean_t1 = np.mean(windows[labels == 0], axis=0)
    mean_t2 = np.mean(windows[labels == 1], axis=0)
    
    # Calcular diferença T2-T1
    difference = mean_t2 - mean_t1
    
    # Criar heatmap da diferença
    im = plt.imshow(difference, 
                    aspect='auto',
                    cmap='RdBu_r',
                    interpolation='nearest')
    
    plt.colorbar(im, label='Diferença de Amplitude (μV)')
    plt.title('Diferença T2-T1 por Canal e Tempo\n(Vermelho = T2 > T1, Azul = T1 > T2)')
    plt.xlabel('Tempo (amostras)')
    plt.ylabel('Canais EEG')
    
    # Adicionar labels dos canais
    plt.yticks(range(16), [f'Canal {i}' for i in range(16)])
    
    # Salvar figura
    plt.savefig('t1_t2_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_mean_amplitude(windows, labels):
    """Plot da amplitude média ao longo do tempo para T1 e T2"""
    plt.figure(figsize=(15, 8))
    
    # Tempo em segundos
    time_axis = np.linspace(0, 3.2, windows.shape[2])
    
    # Calcular média e desvio padrão para cada classe
    for class_label, (name, color) in enumerate([('T1 (Mão Esquerda)', 'blue'), 
                                               ('T2 (Mão Direita)', 'red')]):
        class_data = windows[labels == class_label]
        mean_amplitude = np.mean(class_data, axis=(0, 1))  # Média sobre trials e canais
        std_amplitude = np.std(np.mean(class_data, axis=1), axis=0)  # Desvio padrão
        
        plt.plot(time_axis, mean_amplitude, label=name, color=color, linewidth=2)
        plt.fill_between(time_axis, 
                        mean_amplitude - std_amplitude,
                        mean_amplitude + std_amplitude, 
                        color=color, alpha=0.2)
    
    plt.title('Amplitude Média ao Longo do Tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude Média (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.savefig('mean_amplitude.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Carregar dados
        print("Iniciando análise dos dados EEG...")
        windows, labels = load_openbci_data()
        print(f"Dados carregados: {len(windows)} trials")
        
        # Gerar todas as visualizações
        print("\nGerando visualizações...")
        
        print("1. Gerando mapa de correlação entre canais...")
        plot_channel_correlation(windows)
        
        print("2. Gerando mapa de diferença T1-T2...")
        plot_t1_t2_difference(windows, labels)
        
        print("3. Gerando gráfico de amplitude média...")
        plot_mean_amplitude(windows, labels)
        
        print("\nVisualizações concluídas! Arquivos salvos:")
        print("- channel_correlation.png")
        print("- t1_t2_difference.png")
        print("- mean_amplitude.png")
        
    except Exception as e:
        print(f"Erro durante a análise: {e}")

if __name__ == "__main__":
    main()
