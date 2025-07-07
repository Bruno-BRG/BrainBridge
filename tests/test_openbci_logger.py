"""
Teste do Logger OpenBCI - Verifica formato de saída
"""
import sys
import os
import time
import numpy as np

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from openbci_csv_logger import OpenBCICSVLogger

def test_openbci_logger():
    """Testa o logger OpenBCI com dados simulados"""
    
    print("=== Teste do Logger OpenBCI ===")
    
    # Criar logger
    logger = OpenBCICSVLogger(
        patient_id="TEST001",
        task="motor_imagery_test",
        base_path="../data/recordings"
    )
    
    print(f"Arquivo criado: {logger.filename}")
    
    # Simular 1200 amostras (cerca de 10 segundos a 125Hz)
    num_samples = 1200
    
    for i in range(num_samples):
        # Gerar dados EEG simulados (16 canais)
        eeg_data = np.random.randn(16) * 20  # Amplitudes típicas
        
        marker = None
        
        # Adicionar marcadores em pontos específicos
        if i == 400:  # T1 na amostra 400
            marker = "T1"
            print(f"Amostra {i}: Adicionando marcador T1")
        elif i == 1000:  # T2 na amostra 1000
            marker = "T2"
            print(f"Amostra {i}: Adicionando marcador T2")
        
        # Log da amostra
        logger.log_sample(eeg_data.tolist(), marker)
        
        # Mostrar progresso
        if i % 100 == 0:
            print(f"Amostras processadas: {i}")
    
    # Fechar logger testando ambos os métodos
    print("\nTestando stop_logging()...")
    logger.stop_logging()  # Método esperado pela interface
    
    print(f"\nTeste concluído! Arquivo salvo: {logger.filename}")
    print(f"Total de amostras: {num_samples}")
    
    # Verificar arquivo criado
    filepath = os.path.join("../data/recordings", logger.filename)
    if os.path.exists(filepath):
        print(f"Tamanho do arquivo: {os.path.getsize(filepath)} bytes")
        
        # Mostrar as primeiras linhas
        print("\nPrimeiras 10 linhas do arquivo:")
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f"Linha {i}: {line.strip()}")
                else:
                    break
    
    return filepath

def test_baseline_timer():
    """Testa o sistema de baseline"""
    print("\n=== Teste do Sistema de Baseline ===")
    
    logger = OpenBCICSVLogger(
        patient_id="TEST002",
        task="baseline_test",
        base_path="../data/recordings"
    )
    
    print("Iniciando baseline...")
    logger.start_baseline()
    
    # Testar por 10 segundos (baseline deveria durar 5 minutos)
    for i in range(10):
        remaining = logger.get_baseline_remaining()
        active = logger.is_baseline_active()
        print(f"Segundo {i}: Baseline ativo: {active}, Restante: {remaining:.1f}s")
        
        # Tentar adicionar marcador (deveria ser bloqueado)
        marker_result = logger.add_marker("T1")
        if marker_result is None:
            print(f"  -> Marcador T1 bloqueado durante baseline")
        
        time.sleep(1)
    
    logger.close()
    print("Teste de baseline concluído!")

if __name__ == "__main__":
    # Criar diretórios se não existirem
    os.makedirs("../data/recordings", exist_ok=True)
    
    # Executar testes
    csv_file = test_openbci_logger()
    test_baseline_timer()
    
    print(f"\nArquivos de teste salvos em: data/recordings/")
    print("Verifique o formato comparando com o arquivo de exemplo!")
