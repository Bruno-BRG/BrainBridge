import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_eeg_file(file_path):
    """
    Analisa um arquivo EEG individual para extrair informações sobre marcadores
    """
    try:
        # Lê o arquivo CSV
        df = pd.read_csv(file_path, skiprows=4)
        
        # Extrai as anotações (última coluna)
        annotations = df.iloc[:, -1].values
        
        # Conta os marcadores
        marker_counts = defaultdict(int)
        marker_positions = defaultdict(list)
        
        for i, annotation in enumerate(annotations):
            if annotation in ['T0', 'T1', 'T2']:
                marker_counts[annotation] += 1
                marker_positions[annotation].append(i)
        
        # Calcula durações dos segmentos
        durations = {'T1_to_T0': [], 'T2_to_T0': []}
        
        # Para T1 -> T0
        for t1_pos in marker_positions['T1']:
            next_t0 = None
            for t0_pos in marker_positions['T0']:
                if t0_pos > t1_pos:
                    next_t0 = t0_pos
                    break
            if next_t0:
                duration = (next_t0 - t1_pos) / 125.0  # Converte para segundos
                durations['T1_to_T0'].append(duration)
        
        # Para T2 -> T0
        for t2_pos in marker_positions['T2']:
            next_t0 = None
            for t0_pos in marker_positions['T0']:
                if t0_pos > t2_pos:
                    next_t0 = t0_pos
                    break
            if next_t0:
                duration = (next_t0 - t2_pos) / 125.0  # Converte para segundos
                durations['T2_to_T0'].append(duration)
        
        return {
            'file_path': file_path,
            'total_samples': len(df),
            'duration_seconds': len(df) / 125.0,
            'marker_counts': dict(marker_counts),
            'marker_positions': dict(marker_positions),
            'segment_durations': durations
        }
        
    except Exception as e:
        print(f"Erro ao analisar {file_path}: {str(e)}")
        return None

def analyze_all_data(data_dir):
    """
    Analisa todos os arquivos EEG no diretório
    """
    all_results = []
    
    # Percorre todos os sujeitos
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)
        
        if os.path.isdir(subject_path):
            print(f"Analisando sujeito: {subject_folder}")
            
            # Processa todos os arquivos CSV do sujeito
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.csv') and not file_name.endswith('_backup.csv'):
                    file_path = os.path.join(subject_path, file_name)
                    result = analyze_eeg_file(file_path)
                    
                    if result:
                        result['subject'] = subject_folder
                        result['file_name'] = file_name
                        all_results.append(result)
    
    return all_results

def generate_analysis_report(results):
    """
    Gera um relatório de análise dos dados
    """
    print("\n" + "="*50)
    print("RELATÓRIO DE ANÁLISE DOS DADOS EEG")
    print("="*50)
    
    # Estatísticas gerais
    total_files = len(results)
    total_subjects = len(set([r['subject'] for r in results]))
    
    print(f"\nEstatísticas Gerais:")
    print(f"  Total de arquivos analisados: {total_files}")
    print(f"  Total de sujeitos: {total_subjects}")
    
    # Análise de marcadores
    all_t0 = sum([r['marker_counts'].get('T0', 0) for r in results])
    all_t1 = sum([r['marker_counts'].get('T1', 0) for r in results])
    all_t2 = sum([r['marker_counts'].get('T2', 0) for r in results])
    
    print(f"\nMarcadores encontrados:")
    print(f"  Total T0: {all_t0}")
    print(f"  Total T1: {all_t1}")
    print(f"  Total T2: {all_t2}")
    
    # Análise de durações dos segmentos
    all_t1_durations = []
    all_t2_durations = []
    
    for result in results:
        all_t1_durations.extend(result['segment_durations']['T1_to_T0'])
        all_t2_durations.extend(result['segment_durations']['T2_to_T0'])
    
    if all_t1_durations:
        print(f"\nDurações dos segmentos T1->T0 (movimento esquerda):")
        print(f"  Média: {np.mean(all_t1_durations):.2f}s")
        print(f"  Mediana: {np.median(all_t1_durations):.2f}s")
        print(f"  Min: {np.min(all_t1_durations):.2f}s")
        print(f"  Max: {np.max(all_t1_durations):.2f}s")
        print(f"  Total de segmentos: {len(all_t1_durations)}")
    
    if all_t2_durations:
        print(f"\nDurações dos segmentos T2->T0 (movimento direita):")
        print(f"  Média: {np.mean(all_t2_durations):.2f}s")
        print(f"  Mediana: {np.median(all_t2_durations):.2f}s")
        print(f"  Min: {np.min(all_t2_durations):.2f}s")
        print(f"  Max: {np.max(all_t2_durations):.2f}s")
        print(f"  Total de segmentos: {len(all_t2_durations)}")
    
    # Análise por sujeito
    print(f"\nAnálise por sujeito:")
    subject_stats = defaultdict(lambda: {'files': 0, 'T1': 0, 'T2': 0, 'T0': 0})
    
    for result in results:
        subject = result['subject']
        subject_stats[subject]['files'] += 1
        subject_stats[subject]['T0'] += result['marker_counts'].get('T0', 0)
        subject_stats[subject]['T1'] += result['marker_counts'].get('T1', 0)
        subject_stats[subject]['T2'] += result['marker_counts'].get('T2', 0)
    
    for subject, stats in sorted(subject_stats.items()):
        print(f"  {subject}: {stats['files']} arquivos, T0={stats['T0']}, T1={stats['T1']}, T2={stats['T2']}")
    
    # Verifica qualidade dos dados
    print(f"\nVerificação de qualidade:")
    problematic_files = []
    
    for result in results:
        issues = []
        
        if result['marker_counts'].get('T1', 0) == 0 and result['marker_counts'].get('T2', 0) == 0:
            issues.append("Sem marcadores T1/T2")
        
        if result['marker_counts'].get('T0', 0) == 0:
            issues.append("Sem marcadores T0")
        
        t1_segments = len(result['segment_durations']['T1_to_T0'])
        t2_segments = len(result['segment_durations']['T2_to_T0'])
        
        if t1_segments == 0 and t2_segments == 0:
            issues.append("Nenhum segmento válido extraído")
        
        if issues:
            problematic_files.append({
                'file': f"{result['subject']}/{result['file_name']}",
                'issues': issues
            })
    
    if problematic_files:
        print(f"  Arquivos com problemas: {len(problematic_files)}")
        for problem in problematic_files[:10]:  # Mostra apenas os primeiros 10
            print(f"    {problem['file']}: {', '.join(problem['issues'])}")
        if len(problematic_files) > 10:
            print(f"    ... e mais {len(problematic_files) - 10} arquivos")
    else:
        print(f"  Todos os arquivos parecem estar OK!")
    
    return {
        'total_files': total_files,
        'total_subjects': total_subjects,
        'total_t1_segments': len(all_t1_durations),
        'total_t2_segments': len(all_t2_durations),
        'problematic_files': problematic_files
    }

def main():
    # Caminho para os dados EEG
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    
    print("Analisando dados EEG...")
    print(f"Diretório: {data_dir}")
    
    # Analisa todos os dados
    results = analyze_all_data(data_dir)
    
    # Gera relatório
    summary = generate_analysis_report(results)
    
    # Salva os resultados em um arquivo
    output_file = "eeg_data_analysis.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ANÁLISE DOS DADOS EEG\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total de arquivos: {summary['total_files']}\n")
        f.write(f"Total de sujeitos: {summary['total_subjects']}\n")
        f.write(f"Segmentos T1->T0: {summary['total_t1_segments']}\n")
        f.write(f"Segmentos T2->T0: {summary['total_t2_segments']}\n")
        f.write(f"Arquivos problemáticos: {len(summary['problematic_files'])}\n\n")
        
        if summary['problematic_files']:
            f.write("ARQUIVOS COM PROBLEMAS:\n")
            for problem in summary['problematic_files']:
                f.write(f"  {problem['file']}: {', '.join(problem['issues'])}\n")
    
    print(f"\nAnálise completa! Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()
