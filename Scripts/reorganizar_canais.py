import pandas as pd
import os
import shutil
from datetime import datetime

def reorganizar_canais_csv(arquivo_csv):
    """
    Reorganiza a ordem dos canais EEG em um arquivo CSV do OpenBCI
    
    Ordem atual: C3, C4, Fp1, Fp2, F7, F3, F4, F8, T7, T8, P7, P3, P4, P8, O1, O2
    Nova ordem:  Fp1, Fp2, F7, F8, F3, F4, T7, T8, C3, C4, P7, P8, P3, P4, O1, O2
    """
    
    print(f"Processando: {arquivo_csv}")
    
    # Backup do arquivo original
    backup_path = arquivo_csv.replace('.csv', '_backup.csv')
    shutil.copy2(arquivo_csv, backup_path)
    
    # Ler o cabeçalho personalizado (primeiras 4 linhas)
    cabecalho_personalizado = []
    with open(arquivo_csv, 'r') as f:
        for i in range(4):
            cabecalho_personalizado.append(f.readline().strip())
    
    # Ler o CSV a partir da linha 5 (onde começam os dados)
    df = pd.read_csv(arquivo_csv, skiprows=4)
    
    # Mapeamento da ordem atual para a nova ordem
    # Ordem atual nos EXG Channels: 0=C3, 1=C4, 2=Fp1, 3=Fp2, 4=F7, 5=F3, 6=F4, 7=F8, 8=T7, 9=T8, 10=P7, 11=P3, 12=P4, 13=P8, 14=O1, 15=O2
    # Nova ordem desejada:       0=Fp1, 1=Fp2, 2=F7, 3=F8, 4=F3, 5=F4, 6=T7, 7=T8, 8=C3, 9=C4, 10=P7, 11=P8, 12=P3, 13=P4, 14=O1, 15=O2
    
    mapeamento_canais = {
        'EXG Channel 0': df['EXG Channel 2'].copy(),   # Fp1 (era canal 2, agora canal 0)
        'EXG Channel 1': df['EXG Channel 3'].copy(),   # Fp2 (era canal 3, agora canal 1)
        'EXG Channel 2': df['EXG Channel 4'].copy(),   # F7  (era canal 4, agora canal 2)
        'EXG Channel 3': df['EXG Channel 7'].copy(),   # F8  (era canal 7, agora canal 3)
        'EXG Channel 4': df['EXG Channel 5'].copy(),   # F3  (era canal 5, agora canal 4)
        'EXG Channel 5': df['EXG Channel 6'].copy(),   # F4  (era canal 6, agora canal 5)
        'EXG Channel 6': df['EXG Channel 8'].copy(),   # T7  (era canal 8, agora canal 6)
        'EXG Channel 7': df['EXG Channel 9'].copy(),   # T8  (era canal 9, agora canal 7)
        'EXG Channel 8': df['EXG Channel 0'].copy(),   # C3  (era canal 0, agora canal 8)
        'EXG Channel 9': df['EXG Channel 1'].copy(),   # C4  (era canal 1, agora canal 9)
        'EXG Channel 10': df['EXG Channel 10'].copy(), # P7  (era canal 10, agora canal 10)
        'EXG Channel 11': df['EXG Channel 13'].copy(), # P8  (era canal 13, agora canal 11)
        'EXG Channel 12': df['EXG Channel 11'].copy(), # P3  (era canal 11, agora canal 12)
        'EXG Channel 13': df['EXG Channel 12'].copy(), # P4  (era canal 12, agora canal 13)
        'EXG Channel 14': df['EXG Channel 14'].copy(), # O1  (era canal 14, agora canal 14)
        'EXG Channel 15': df['EXG Channel 15'].copy(), # O2  (era canal 15, agora canal 15)
    }
    
    # Aplicar a reorganização
    for canal, novos_dados in mapeamento_canais.items():
        df[canal] = novos_dados
    
    # Salvar o arquivo reorganizado
    df.to_csv(arquivo_csv, index=False)
    
    # Adicionar o cabeçalho personalizado de volta
    with open(arquivo_csv, 'r') as f:
        conteudo = f.read()
    
    with open(arquivo_csv, 'w') as f:
        for linha in cabecalho_personalizado:
            f.write(linha + '\n')
        f.write(conteudo)
    
    print(f"✓ Reorganizado: {arquivo_csv}")
    return True

def reorganizar_dataset_completo(diretorio_eeg_data):
    """
    Reorganiza todos os arquivos CSV no dataset eeg_data
    """
    
    print("=== REORGANIZAÇÃO DE CANAIS EEG ===")
    print("Ordem atual: C3, C4, Fp1, Fp2, F7, F3, F4, F8, T7, T8, P7, P3, P4, P8, O1, O2")
    print("Nova ordem:  Fp1, Fp2, F7, F8, F3, F4, T7, T8, C3, C4, P7, P8, P3, P4, O1, O2")
    print()
    
    arquivos_processados = 0
    arquivos_com_erro = 0
    
    # Percorrer todas as pastas de sujeitos
    for pasta_sujeito in os.listdir(diretorio_eeg_data):
        caminho_pasta = os.path.join(diretorio_eeg_data, pasta_sujeito)
        
        if os.path.isdir(caminho_pasta):
            print(f"\nProcessando pasta: {pasta_sujeito}")
            
            # Percorrer todos os arquivos CSV na pasta do sujeito
            for arquivo in os.listdir(caminho_pasta):
                if arquivo.endswith('_csv_openbci.csv'):
                    caminho_arquivo = os.path.join(caminho_pasta, arquivo)
                    
                    try:
                        reorganizar_canais_csv(caminho_arquivo)
                        arquivos_processados += 1
                    except Exception as e:
                        print(f"✗ Erro ao processar {arquivo}: {e}")
                        arquivos_com_erro += 1
    
    print(f"\n=== RESUMO ===")
    print(f"Arquivos processados com sucesso: {arquivos_processados}")
    print(f"Arquivos com erro: {arquivos_com_erro}")
    print(f"Total: {arquivos_processados + arquivos_com_erro}")
    
    if arquivos_com_erro == 0:
        print("\n✓ Todos os arquivos foram reorganizados com sucesso!")
    else:
        print(f"\n⚠ {arquivos_com_erro} arquivos tiveram problemas durante o processamento.")
    
    print("\nNOTA: Backups dos arquivos originais foram criados com sufixo '_backup.csv'")

def verificar_reorganizacao(arquivo_csv):
    """
    Verifica se a reorganização foi aplicada corretamente
    """
    print(f"\nVerificando reorganização em: {os.path.basename(arquivo_csv)}")
    
    # Ler algumas linhas para verificar
    df = pd.read_csv(arquivo_csv, skiprows=4, nrows=5)
    
    print("Primeiras 5 amostras dos primeiros 8 canais (nova ordem):")
    print("Canal 0 (Fp1), Canal 1 (Fp2), Canal 2 (F7), Canal 3 (F8), Canal 4 (F3), Canal 5 (F4), Canal 6 (T7), Canal 7 (T8)")
    
    for i in range(5):
        valores = [f"{df.iloc[i][f'EXG Channel {j}']:.2f}" for j in range(8)]
        print(f"Amostra {i+1}: {', '.join(valores)}")

if __name__ == "__main__":
    # Diretório do dataset EEG
    diretorio_eeg = r"c:\Users\Chari\Documents\CIMATEC\BrainBridge\bci\eeg_data"
    
    # Verificar se o diretório existe
    if not os.path.exists(diretorio_eeg):
        print(f"Erro: Diretório não encontrado: {diretorio_eeg}")
        exit(1)
    
    # Confirmar antes de executar
    resposta = input(f"Deseja reorganizar todos os arquivos CSV em {diretorio_eeg}? (s/n): ")
    
    if resposta.lower() in ['s', 'sim', 'y', 'yes']:
        print(f"\nIniciando reorganização em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        reorganizar_dataset_completo(diretorio_eeg)
        
        # Verificar um arquivo como exemplo
        arquivo_exemplo = os.path.join(diretorio_eeg, "S001", "S001R04_csv_openbci.csv")
        if os.path.exists(arquivo_exemplo):
            verificar_reorganizacao(arquivo_exemplo)
        
        print(f"\nConcluído em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Operação cancelada.")
