"""
Teste da correção de duplicação
"""

import numpy as np
import csv
import os
from datetime import datetime
from bci_interface import SimpleCSVLogger

def test_no_duplication():
    """Testa se não há duplicação no logger"""
    print("=== Teste de Duplicação Corrigida ===")
    
    # Criar logger de teste
    filename = "test_no_duplication.csv"
    logger = SimpleCSVLogger(filename)
    logger.start_logging()
    
    # Adicionar alguns dados de teste
    test_data = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1]),
        np.array([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2])
    ]
    
    # Simular o que acontece na interface
    for data in test_data:
        # Simular o que acontece em on_data_received
        logger.log_data(data)  # Apenas uma chamada agora
    
    logger.stop_logging()
    
    # Verificar o arquivo
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"Arquivo criado com {len(lines)} linhas")
        print("Linhas do arquivo:")
        for i, line in enumerate(lines):
            print(f"  {i+1}: {line.strip()}")
        
        # Verificar se há duplicação
        data_lines = lines[1:]  # Excluir cabeçalho
        unique_lines = set(data_lines)
        
        if len(data_lines) == len(unique_lines):
            print("✓ Nenhuma duplicação encontrada!")
        else:
            print(f"✗ Duplicação encontrada! {len(data_lines)} linhas, {len(unique_lines)} únicas")
        
        # Limpar arquivo de teste
        os.remove(filename)
        print("✓ Arquivo de teste removido")
    else:
        print("✗ Arquivo não foi criado")

if __name__ == "__main__":
    test_no_duplication()
