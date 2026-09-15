import mne
import pandas as pd
import numpy as np
import os
import re
 # OBSERVAÇÃO O ARQUIVO AQ GERADO VAI SAIR NA PASTA QUE ESTÁ O SEU ARQUIVO EDF COM O NOME DO ARQUIVO EDF + _csv_openbci
def processar_edf_para_openbci(diretorio_edf, filtragem, canais, algarismos_significativos=5):
    """
    A partir dos diretorio, canais desejados, filtragem que escolhe e a quantidade de algarismos significativos esse código filtra os dados
    transforma em microvolts o eeg e formata ele em csv colocando o cabeçalho necessário para 16 eletrodos. Assim o arquivo fica funcional para 
    ser utilizado no openbci_GUI. SE CANAIS FOR VAZIO VAI PEGAR TODOS OS ELETRODOS PRESENTE NO ARQUIVO EDF
    
    As anotações T0 (repouso), T1 (mão esquerda) e T2 (mão direita) são extraídas do arquivo EDF e adicionadas na coluna Annotations.
    
    Parâmetros de filtragem:
    - filtragem = (low_freq, high_freq) : Aplica filtro passa-banda
    - filtragem = (low_freq, None) : Aplica apenas filtro passa-alta (high-pass)
    - filtragem = (None, high_freq) : Aplica apenas filtro passa-baixa (low-pass)
    - filtragem = None : Sem filtragem

    A saida sempre usa 125 Hz (inclui anti-alias do resampling). A ordem dos
    canais selecionados e preservada, mas a montagem do runtime e desconhecida:
    nomes EDF nao comprovam correspondencia com os eletrodos da sessao real.
    """

    # Carregar arquivo EDF
    raw = mne.io.read_raw_edf(diretorio_edf, preload=True)
    
    # Aplicar filtro personalizado
    if filtragem:
        l_freq = filtragem[0]  # Frequência passa-alta (high-pass)
        h_freq = filtragem[1]  # Frequência passa-baixa (low-pass)
        
        if l_freq is not None or h_freq is not None:
            print(f"Aplicando filtro: High-pass={l_freq} Hz, Low-pass={h_freq} Hz")
            raw.filter(l_freq, h_freq)
        else:
            print("Nenhum filtro aplicado")
    else:
        print("Nenhum filtro aplicado")
    
    # Selecionar canais
    if not canais:  # Se lista vazia
        canais = raw.ch_names  # Pega todos os canais
    else:  # Se lista não estiver vazia
        raw.pick(canais)  # Seleciona apenas os canais fornecidos
    
    # Resample the signal before extracting samples or indexing annotations.
    raw.resample(125.0)

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
        'Timestamp', 'Other', 'Timestamp (Formatted)', 'Annotations'
    ]
    
    df.insert(0, 'Sample Index', range(1, len(df)+1))
    for col in estrutura_colunas[len(raw.ch_names)+1:]:
        df[col] = 0 if col != 'Annotations' else ''
    
    # Extrair anotações do arquivo EDF
    # As anotações no MNE estão em raw.annotations
    if raw.annotations is not None and len(raw.annotations) > 0:
        print(f"Encontradas {len(raw.annotations)} anotações no arquivo EDF")
        
        # Converter tempo das anotações para índices de amostras
        sfreq = raw.info['sfreq']
        
        for annot in raw.annotations:
            descricao = annot['description']
            onset_time = annot['onset']
            
            # Calcular o índice da amostra correspondente
            # Sample Index começa em 1, então usamos sample_idx - 1 para o dataframe
            sample_idx = int(round((onset_time - raw.first_time) * sfreq))
            
            # Verificar se o índice está dentro dos limites
            if 0 <= sample_idx < len(df):
                df.at[sample_idx, 'Annotations'] = descricao
                print(f"Anotação '{descricao}' adicionada no sample {sample_idx + 1}")
    else:
        print("Nenhuma anotação encontrada no arquivo EDF")
        
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
            f"%Signal Stage = {'filtered' if filtragem and any(f is not None for f in filtragem) else 'raw'}\n"
            f"%Channel Names = {', '.join(raw.ch_names)}\n"
            "%Runtime Montage = unknown; EDF channel names do not establish a match\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
        )
        f.write(conteudo)
    
    print(f"Arquivo gerado: {caminho_saida}")
    return caminho_saida

def processar_pasta_recursivamente(pasta_raiz, filtragem=None, canais=None, algarismos_significativos=5):
    """
    Processa recursivamente todos os arquivos EDF em uma pasta e suas subpastas.
    
    Parâmetros:
    - pasta_raiz: Caminho da pasta raiz a processar
    - filtragem: Tupla (low_freq, high_freq) ou None
    - canais: Lista de canais desejados ou None (para usar todos)
    - algarismos_significativos: Número de casas decimais
    """
    
    contador = {'sucesso': 0, 'erro': 0}
    
    print(f"\n{'='*70}")
    print(f"🔍 Iniciando processamento recursivo de: {pasta_raiz}")
    print(f"{'='*70}\n")
    
    # Caminhar por todas as subpastas
    for raiz, subpastas, arquivos in os.walk(pasta_raiz):
        print(f"\n📁 Processando pasta: {raiz}")
        
        # Filtrar apenas arquivos EDF com número ímpar (ex: S001R11.edf)
        arquivos_edf = []
        for f in arquivos:
            if f.lower().endswith('.edf'):
                # Extrai o número antes da extensão .edf
                match = re.search(r'(\d+)\.edf$', f, re.IGNORECASE)
                if match and int(match.group(1)) % 2 == 0:
                    arquivos_edf.append(f)
        
        if not arquivos_edf:
            print(f"   ℹ️  Nenhum arquivo EDF encontrado")
            continue
        
        print(f"   📊 Encontrados {len(arquivos_edf)} arquivo(s) EDF")
        
        # Processar cada arquivo EDF
        for arquivo_edf in arquivos_edf:
            caminho_completo = os.path.join(raiz, arquivo_edf)
            print(f"\n   ⏳ Processando: {arquivo_edf}")
            
            try:
                processar_edf_para_openbci(
                    caminho_completo,
                    filtragem,
                    canais,
                    algarismos_significativos
                )
                contador['sucesso'] += 1
                print(f"   ✅ Sucesso!")
            except Exception as e:
                contador['erro'] += 1
                print(f"   ❌ Erro ao processar {arquivo_edf}: {e}")
    
    # Resumo final
    print(f"\n{'='*70}")
    print(f"📊 RESUMO DO PROCESSAMENTO")
    print(f"{'='*70}")
    print(f"✅ Arquivos processados com sucesso: {contador['sucesso']}")
    print(f"❌ Arquivos com erro: {contador['erro']}")
    print(f"{'='*70}\n")
    
    return contador


if __name__ == "__main__":
    # Caminho da pasta com dados do PhysioNet
    pasta_dados = os.path.join(
        os.path.dirname(__file__),
        "data",
        "MNE-eegbci-data"
    )
    
    # Verificar se a pasta existe
    if not os.path.exists(pasta_dados):
        print(f"❌ Pasta não encontrada: {pasta_dados}")
        print(f"⚠️  Execute primeiro o script download_physionet.py para baixar os dados.")
    else:
        # Nova ordem dos canais: Fp1, Fp2, F7, F8, F3, F4, T7, T8, C3, C4, P7, P8, P3, P4, O1, O2
        canais_desejados = ['Fp1.', 'Fp2.', 'F7..', 'F8..', 'F3..', 'F4..', 'T7..', 'T8..', 'C3..', 'C4..', 'P7..', 'P8..', 'P3..', 'P4..', 'O1..', 'O2..'] 
        # CASO QUEIRA TODOS OS ELETRODOS DO ARQUIVO É SÓ DEIXAR ESSA VARIAVEL VAZIA: canais_desejados = []
        
        # Opções de filtragem:
        # filtragem = (0.5, 60)    # Filtro passa-banda: high-pass em 0.5 Hz e low-pass em 60 Hz
        # filtragem = (0.5, None)  # Apenas high-pass em 0.5 Hz (remove frequências abaixo de 0.5 Hz)
        # filtragem = (None, 60)   # Apenas low-pass em 60 Hz (remove frequências acima de 60 Hz)
        # filtragem = None         # Sem filtragem
        
        filtragem = None  # Preserve raw signal for the shared ML preprocessing.
        algarismos_significativos = 5
        
        # Processar toda a pasta recursivamente
        processar_pasta_recursivamente(
            pasta_dados,
            filtragem=filtragem,
            canais=canais_desejados,
            algarismos_significativos=algarismos_significativos
        )
