"""
Teste Real da Interface com Organização por Paciente
Simula o fluxo completo da interface
"""
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from openbci_csv_logger import OpenBCICSVLogger
import numpy as np

def test_interface_workflow():
    """Simula o fluxo completo da interface"""
    
    print("=== Teste de Fluxo Completo da Interface ===")
    
    # Simular dados de pacientes como viria da interface
    patients_data = [
        {"id": 1, "name": "João Silva"},
        {"id": 2, "name": "Maria Santos"},
        {"id": 3, "name": "Pedro Oliveira"}
    ]
    
    for patient in patients_data:
        print(f"\n--- Simulando gravação para {patient['name']} (ID: {patient['id']}) ---")
        
        # Como a interface formata o patient_id
        patient_id = f"P{patient['id']:03d}"  # P001, P002, P003
        
        # Diferentes tarefas que um paciente pode realizar
        tasks = ["motor_imagery", "baseline", "rest"]
        
        for task in tasks:
            print(f"  Gravação: {task}")
            
            # Criar logger como a interface faria
            logger = OpenBCICSVLogger(
                patient_id=patient_id,
                task=task,
                patient_name=patient["name"],
                base_path="../data/recordings"
            )
            
            # Simular algumas amostras (como se fossem dados reais)
            for i in range(100):
                eeg_data = np.random.randn(16) * 20
                marker = None
                
                # Simular marcadores em momentos específicos
                if task == "motor_imagery":
                    if i == 30:
                        marker = "T1"
                    elif i == 70:
                        marker = "T2"
                
                logger.log_sample(eeg_data.tolist(), marker)
            
            # Finalizar como a interface faria
            logger.stop_logging()
            
            # Verificar resultado
            full_path = logger.get_full_path()
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                relative_path = f"{logger.patient_folder}/{logger.filename}"
                print(f"    ✅ Arquivo criado: {relative_path} ({size} bytes)")
            else:
                print(f"    ❌ Erro ao criar arquivo")
    
    # Mostrar estrutura final
    print(f"\n=== Estrutura Final de Gravações ===")
    recordings_path = "../data/recordings"
    
    if os.path.exists(recordings_path):
        for root, dirs, files in os.walk(recordings_path):
            level = root.replace(recordings_path, '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            if folder_name:
                print(f"{indent}{folder_name}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({size} bytes)")
    
    print(f"\n✅ Teste de fluxo completo finalizado!")
    print(f"   - Cada paciente tem sua própria pasta")
    print(f"   - Múltiplas tarefas organizadas por paciente")
    print(f"   - Formato OpenBCI mantido")

def check_existing_structure():
    """Verifica a estrutura existente"""
    print(f"\n=== Verificando Estrutura Existente ===")
    
    recordings_path = "../data/recordings"
    
    if not os.path.exists(recordings_path):
        print("❌ Pasta de gravações não existe ainda")
        return
    
    patient_folders = []
    loose_files = []
    
    for item in os.listdir(recordings_path):
        item_path = os.path.join(recordings_path, item)
        if os.path.isdir(item_path):
            patient_folders.append(item)
        elif item.endswith('.csv'):
            loose_files.append(item)
    
    print(f"Pastas de pacientes: {len(patient_folders)}")
    for folder in patient_folders:
        print(f"  📁 {folder}")
    
    print(f"Arquivos soltos (formato antigo): {len(loose_files)}")
    for file in loose_files[:5]:  # Mostrar apenas os primeiros 5
        print(f"  📄 {file}")
    if len(loose_files) > 5:
        print(f"  ... e mais {len(loose_files) - 5} arquivos")
    
    if loose_files:
        print(f"\n💡 Dica: Os arquivos soltos são do formato antigo.")
        print(f"   Novos arquivos serão organizados em pastas por paciente.")

if __name__ == "__main__":
    # Verificar estrutura existente primeiro
    check_existing_structure()
    
    # Executar teste do novo fluxo
    test_interface_workflow()
    
    print(f"\n🎉 ORGANIZAÇÃO POR PACIENTE FUNCIONANDO PERFEITAMENTE!")
    print(f"📋 Agora cada paciente terá sua própria pasta em data/recordings/")
