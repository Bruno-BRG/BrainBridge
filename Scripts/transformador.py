import mne
import pandas as pd
import numpy as np
import os
 # OBSERVAÇÃO O ARQUIVO AQ GERADO VAI SAIR NA PASTA QUE ESTÁ O SEU ARQUIVO EDF COM O NOME DO ARQUIVO EDF + _csv_openbci
def processar_edf_para_openbci(diretorio_edf, filtragem, canais, algarismos_significativos=5):
    """
    A partir dos diretorio, canais desejados, filtragem que escolhe e a quantidade de algarismos significativos esse código filtra os dados
    transforma em microvolts o eeg e formata ele em csv colocando o cabeçalho necessário para 16 eletrodos. Assim o arquivo fica funcional para 
    ser utilizado no openbci_GUI. SE CANAIS FOR VAZIO VAI PEGAR TODOS OS ELETRODOS PRESENTE NO ARQUIVO EDF
    """

    # Carregar arquivo EDF
    raw = mne.io.read_raw_edf(diretorio_edf, preload=True)
    
    # Aplicar filtro personalizado
    if filtragem:
        raw.filter(filtragem[0], filtragem[1])
    
    # Selecionar canais
    if not canais:  # Se lista vazia
        canais = raw.ch_names  # Pega todos os canais
    else:  # Se lista não estiver vazia
        raw.pick(canais)  # Seleciona apenas os canais fornecidos
    
    # Processamento dos dados
    dados, tempos = raw[:, :]
    dados_volts = dados * 1e6  # Converter para microV
    dados_arredondados = np.round(dados_volts, algarismos_significativos)
    
    # Criar DataFrame
    df = pd.DataFrame(dados_arredondados.T, 
                     columns=[f'EXG Channel {i}' for i in range(len(raw.ch_names))])
    
    # Adicionar colunas complementares zeradas
    estrutura_colunas = [
        'Sample Index',
        *[f'EXG Channel {i}' for i in range(len(raw.ch_names))],
        'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
        *['Other']*7,
        'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2',
        'Timestamp', 'Other', 'Timestamp (Formatted)'
    ]
    
    df.insert(0, 'Sample Index', range(1, len(df)+1))
    for col in estrutura_colunas[len(raw.ch_names)+1:]:
        df[col] = 0
        
    df = df[estrutura_colunas]  # Ordenar colunas
    
    # Gerar nome do arquivo
    nome_base = os.path.splitext(os.path.basename(diretorio_edf))[0]
    caminho_saida = os.path.join(
        os.path.dirname(diretorio_edf),
        f"{nome_base}_csv_openbci.csv"
    )
    
    # Salvar CSV formatado
    df.to_csv(caminho_saida, 
             index=False, 
             float_format=f'%.{algarismos_significativos}g')
    
    # Adicionar cabeçalho personalizado
    with open(caminho_saida, 'r+') as f:
        conteudo = f.read()
        f.seek(0, 0)
        f.write(
            f"%OpenBCI Raw EXG Data\n"
            f"%Number of channels = {len(raw.ch_names)}\n"
            "%Sample Rate = 125 Hz\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
        )
        f.write(conteudo)
    
    print(f"Arquivo gerado: {caminho_saida}")
    return caminho_saida

# Exemplo de uso:
arquivo_edf = r"C:\Users\Chari\Documents\github\projetoBCI\S001R01.edf"
# Nova ordem dos canais: Fp1, Fp2, F7, F8, F3, F4, T7, T8, C3, C4, P7, P8, P3, P4, O1, O2
canais_desejados = ['Fp1.', 'Fp2.', 'F7..', 'F8..', 'F3..', 'F4..', 'T7..', 'T8..', 'C3..', 'C4..', 'P7..', 'P8..', 'P3..', 'P4..', 'O1..', 'O2..'] #CASO QUERA TODOS OS ELETRODOS DO ARQUIVO É SÓ DEIXAR ESSA VARIAVEL VAZIA
filtragem= (0.5,60)
algarismos_significativos=4 

processar_edf_para_openbci(arquivo_edf, filtragem, canais_desejados, algarismos_significativos)

