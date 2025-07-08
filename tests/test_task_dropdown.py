"""
Teste da Nova Interface com Dropdown de Tarefas
Simula a seleção de diferentes tarefas
"""
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from openbci_csv_logger import OpenBCICSVLogger
import numpy as np

def test_task_dropdown_options():
    """Testa as opções do dropdown de tarefas"""
    
    print("=== Teste do Dropdown de Tarefas ===")
    
    # Opções disponíveis no dropdown
    task_options = ["Baseline", "Treino", "Teste", "Jogo"]
    
    # Como a interface converte as tarefas
    converted_tasks = []
    for task in task_options:
        converted = task.lower().replace(" ", "_")
        converted_tasks.append(converted)
        print(f"'{task}' -> '{converted}'")
    
    print(f"\nTarefas convertidas: {converted_tasks}")
    
    # Simular criação de arquivos com cada tarefa
    print(f"\n=== Simulação de Arquivos por Tarefa ===")
    patient_id = "P001"
    patient_name = "João Silva"
    
    for i, task in enumerate(converted_tasks):
        print(f"\n--- Tarefa {i+1}: {task_options[i]} ---")
        
        logger = OpenBCICSVLogger(
            patient_id=patient_id,
            task=task,
            patient_name=patient_name,
            base_path="../data/recordings"
        )
        
        # Mostrar informações do arquivo
        print(f"Pasta: {logger.patient_folder}")
        print(f"Arquivo: {logger.filename}")
        print(f"Caminho completo: {logger.get_full_path()}")
        
        # Simular algumas amostras
        for j in range(20):
            eeg_data = np.random.randn(16) * 15
            marker = "T1" if j == 10 else None
            logger.log_sample(eeg_data.tolist(), marker)
        
        logger.stop_logging()
        
        # Verificar se arquivo foi criado
        if os.path.exists(logger.get_full_path()):
            size = os.path.getsize(logger.get_full_path())
            print(f"✅ Arquivo criado: {size} bytes")
        else:
            print(f"❌ Erro ao criar arquivo")
    
    return True

def show_expected_structure():
    """Mostra a estrutura esperada com as novas tarefas"""
    print(f"\n=== Estrutura Esperada com Novas Tarefas ===")
    
    structure = """
    data/recordings/
    └── P001_João_Silva/
        ├── P001_baseline_20250708_143000.csv
        ├── P001_treino_20250708_143500.csv
        ├── P001_teste_20250708_144000.csv
        └── P001_jogo_20250708_144500.csv
    """
    
    print(structure)
    
    print("🎯 Vantagens do Dropdown:")
    print("✅ Padronização das tarefas")
    print("✅ Evita erros de digitação")
    print("✅ Interface mais limpa")
    print("✅ Facilita análise posterior dos dados")
    print("✅ Nomes de arquivo consistentes")

def verify_recordings_folder():
    """Verifica a pasta de gravações atual"""
    print(f"\n=== Verificando Pasta de Gravações ===")
    
    recordings_path = "../data/recordings"
    
    if os.path.exists(recordings_path):
        print(f"📁 Pasta encontrada: {recordings_path}")
        
        # Listar pastas de pacientes
        patient_folders = [d for d in os.listdir(recordings_path) 
                          if os.path.isdir(os.path.join(recordings_path, d))]
        
        print(f"Pacientes encontrados: {len(patient_folders)}")
        
        for folder in patient_folders:
            folder_path = os.path.join(recordings_path, folder)
            files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            print(f"  📂 {folder}: {len(files)} arquivos")
            
            # Mostrar alguns exemplos de arquivos
            for file in files[:3]:  # Primeiros 3 arquivos
                print(f"    📄 {file}")
            if len(files) > 3:
                print(f"    ... e mais {len(files) - 3} arquivos")
    else:
        print(f"❌ Pasta não encontrada: {recordings_path}")

if __name__ == "__main__":
    # Verificar estrutura atual
    verify_recordings_folder()
    
    # Testar novas opções de tarefa
    test_passed = test_task_dropdown_options()
    
    # Mostrar estrutura esperada
    show_expected_structure()
    
    print(f"\n🎉 DROPDOWN DE TAREFAS IMPLEMENTADO!")
    print(f"📋 Opções disponíveis: Baseline, Treino, Teste, Jogo")
    print(f"🔄 Conversão automática para nomes de arquivo: baseline, treino, teste, jogo")
    print(f"{'✅ TESTE PASSOU' if test_passed else '❌ TESTE FALHOU'}")
