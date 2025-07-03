"""
Script para converter dados UDP (formato buffer) para formato CSV padrão OpenBCI.

Uso:
    python convert_udp_to_csv.py [arquivo_entrada] [arquivo_saida]
    
Exemplos:
    python convert_udp_to_csv.py raw_data_buffer_001_20250702_144226.csv
    python convert_udp_to_csv.py input.csv output.csv
    python convert_udp_to_csv.py  # Usa arquivo padrão
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse

def convert_udp_to_standard_csv(udp_csv_path, output_path=None, verbose=True):
    """
    Converte o arquivo CSV do UDP (formato buffer) para o formato padrão do OpenBCI.
    
    Args:
        udp_csv_path: Caminho para o arquivo CSV do UDP
        output_path: Caminho de saída (opcional, se não fornecido será gerado automaticamente)
        verbose: Se True, mostra informações detalhadas do processo
    
    Returns:
        str: Caminho do arquivo convertido
    """
    
    # Get the directory where the input file is located
    input_dir = os.path.dirname(os.path.abspath(udp_csv_path))
    
    # Se não foi fornecido caminho de saída, gera um automaticamente
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(udp_csv_path))[0]
        output_path = os.path.join(input_dir, f"{base_name}_converted.csv")
    
    if verbose:
        print(f"Convertendo: {udp_csv_path}")
        print(f"Para: {output_path}")
    
    # Verifica se o arquivo de entrada existe
    if not os.path.exists(udp_csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {udp_csv_path}")
    
    # Lê o arquivo UDP CSV
    try:
        df_udp = pd.read_csv(udp_csv_path)
    except Exception as e:
        raise Exception(f"Erro ao ler arquivo CSV: {e}")
    
    if verbose:
        print(f"Dados UDP carregados: {df_udp.shape[0]} registros")
    
    # Lista para armazenar todas as amostras convertidas
    all_samples = []
    sample_index = 0
    errors = 0
    
    # Processa cada linha do arquivo UDP
    for idx, row in df_udp.iterrows():
        try:
            # Extrai os arrays de dados de cada canal
            channel_arrays = []
            
            for ch in range(1, 17):  # Ch1 a Ch16
                ch_data = row[f'Ch{ch}']
                if isinstance(ch_data, str):
                    # Remove brackets, quotes e quebras de linha
                    clean_data = ch_data.strip('[]"').replace('\n', ' ')
                    # Converte para array numpy
                    arr = np.fromstring(clean_data, sep=' ')
                    if len(arr) == 0:  # Se não conseguiu parsear
                        channel_arrays = None
                        break
                    channel_arrays.append(arr)
                else:
                    # Se não é string (NaN, etc.), pula esta linha
                    channel_arrays = None
                    break
            
            if channel_arrays is None:
                errors += 1
                continue
                
            # Verifica se todos os arrays têm o mesmo tamanho
            array_lengths = [len(arr) for arr in channel_arrays]
            if len(set(array_lengths)) != 1:
                if verbose:
                    print(f"Aviso: Tamanhos diferentes de arrays na linha {idx}: {array_lengths}")
                errors += 1
                continue
                
            buffer_size = array_lengths[0]
            
            # Converte cada amostra do buffer em uma linha do CSV
            for sample_in_buffer in range(buffer_size):
                sample_data = {
                    'Sample Index': sample_index,
                    'EXG Channel 0': channel_arrays[0][sample_in_buffer],
                    'EXG Channel 1': channel_arrays[1][sample_in_buffer],
                    'EXG Channel 2': channel_arrays[2][sample_in_buffer],
                    'EXG Channel 3': channel_arrays[3][sample_in_buffer],
                    'EXG Channel 4': channel_arrays[4][sample_in_buffer],
                    'EXG Channel 5': channel_arrays[5][sample_in_buffer],
                    'EXG Channel 6': channel_arrays[6][sample_in_buffer],
                    'EXG Channel 7': channel_arrays[7][sample_in_buffer],
                    'EXG Channel 8': channel_arrays[8][sample_in_buffer],
                    'EXG Channel 9': channel_arrays[9][sample_in_buffer],
                    'EXG Channel 10': channel_arrays[10][sample_in_buffer],
                    'EXG Channel 11': channel_arrays[11][sample_in_buffer],
                    'EXG Channel 12': channel_arrays[12][sample_in_buffer],
                    'EXG Channel 13': channel_arrays[13][sample_in_buffer],
                    'EXG Channel 14': channel_arrays[14][sample_in_buffer],
                    'EXG Channel 15': channel_arrays[15][sample_in_buffer],
                    # Colunas adicionais preenchidas com zeros (como no formato original)
                    'Accel Channel 0': 0,
                    'Accel Channel 1': 0,
                    'Accel Channel 2': 0,
                    'Other': 0,
                    'Other.1': 0,
                    'Other.2': 0,
                    'Other.3': 0,
                    'Other.4': 0,
                    'Other.5': 0,
                    'Other.6': 0,
                    'Analog Channel 0': 0,
                    'Analog Channel 1': 0,
                    'Analog Channel 2': 0,
                    'Timestamp': 0,
                    'Other.7': 0,
                    'Timestamp (Formatted)': 0,
                    'Annotations': ''
                }
                
                all_samples.append(sample_data)
                sample_index += 1
                
        except Exception as e:
            if verbose:
                print(f"Erro processando linha {idx}: {e}")
            errors += 1
            continue
    
    if verbose:
        print(f"Total de amostras convertidas: {len(all_samples)}")
        if errors > 0:
            print(f"Erros encontrados: {errors}")
    
    if len(all_samples) == 0:
        raise Exception("Nenhuma amostra foi convertida com sucesso")
    
    # Cria DataFrame com todas as amostras
    df_converted = pd.DataFrame(all_samples)
    
    # Cria o cabeçalho no formato OpenBCI
    header_lines = [
        "%OpenBCI Raw EXG Data",
        "%Number of channels = 16",
        "%Sample Rate = 125 Hz",
        "%Board = OpenBCI_GUI$BoardCytonSerialDaisy"
    ]
    
    # Salva o arquivo com cabeçalho
    try:
        with open(output_path, 'w', newline='') as f:
            # Escreve o cabeçalho
            for line in header_lines:
                f.write(line + '\n')
            
            # Escreve os dados
            df_converted.to_csv(f, index=False)
    except Exception as e:
        raise Exception(f"Erro ao salvar arquivo: {e}")
    
    if verbose:
        print(f"Arquivo convertido salvo em: {output_path}")
        print(f"Formato final: {df_converted.shape[0]} amostras x {df_converted.shape[1]} colunas")
    
    return output_path

def main():
    """Função principal para executar a conversão via linha de comando"""
    parser = argparse.ArgumentParser(
        description='Converte dados UDP (formato buffer) para formato CSV padrão OpenBCI',
        epilog='Exemplo: python convert_udp_to_csv.py input.csv output.csv'
    )
    
    parser.add_argument('input_file', nargs='?', 
                       help='Arquivo CSV de entrada (formato UDP)')
    parser.add_argument('output_file', nargs='?',
                       help='Arquivo CSV de saída (opcional)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Modo silencioso (menos output)')
    
    args = parser.parse_args()
    
    # Se não foi fornecido arquivo de entrada, usa o padrão
    if args.input_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input_file = os.path.join(script_dir, 'raw_data_buffer_001_20250702_150642.csv')
    
    # Verifica se o arquivo existe
    if not os.path.exists(args.input_file):
        print(f"Erro: Arquivo não encontrado: {args.input_file}")
        sys.exit(1)
    
    try:
        # Converte o arquivo
        output_file = convert_udp_to_standard_csv(
            args.input_file, 
            args.output_file, 
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print("\n" + "="*50)
            print("CONVERSÃO CONCLUÍDA COM SUCESSO!")
            print("="*50)
            print(f"Arquivo original: {args.input_file}")
            print(f"Arquivo convertido: {output_file}")
            
            # Mostra estatísticas básicas
            try:
                df_converted = pd.read_csv(output_file, comment='%')
                print(f"\nEstatísticas do arquivo convertido:")
                print(f"- Número de amostras: {len(df_converted):,}")
                print(f"- Intervalo de Sample Index: {df_converted['Sample Index'].min()} a {df_converted['Sample Index'].max()}")
                print(f"- Canais EEG: EXG Channel 0 a EXG Channel 15")
                
                # Mostra range de valores dos primeiros canais
                for ch in range(3):
                    col = f'EXG Channel {ch}'
                    min_val = df_converted[col].min()
                    max_val = df_converted[col].max()
                    print(f"- {col}: {min_val:.3f} a {max_val:.3f}")
                
            except Exception as e:
                print(f"Erro ao ler arquivo convertido para estatísticas: {e}")
                
    except Exception as e:
        print(f"Erro durante a conversão: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
