"""
Teste da Nova Organização por Paciente
Verifica se cada paciente tem sua própria pasta
"""
import sys
import os
import shutil

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from openbci_csv_logger import OpenBCICSVLogger
import numpy as np

def test_patient_folder_organization():
    """Testa a organização por pastas de pacientes"""
    
    print("=== Teste de Organização por Pastas de Pacientes ===")
    
    # Criar pasta de teste temporária
    test_base_path = "../data/recordings_test"
    os.makedirs(test_base_path, exist_ok=True)
    
    # Teste com diferentes pacientes
    patients = [
        {"id": "P001", "name": "João Silva", "task": "motor_imagery"},
        {"id": "P002", "name": "Maria Santos", "task": "baseline"},
        {"id": "P001", "name": "João Silva", "task": "rest"},  # Mesmo paciente, tarefa diferente
        {"id": "P003", "name": "Ana Costa-Lima", "task": "motor_imagery"},  # Nome com caracteres especiais
    ]
    
    created_files = []
    
    for patient in patients:
        print(f"\n--- Testando {patient['name']} (ID: {patient['id']}) - Tarefa: {patient['task']} ---")
        
        # Criar logger
        logger = OpenBCICSVLogger(
            patient_id=patient["id"],
            task=patient["task"],
            patient_name=patient["name"],
            base_path=test_base_path
        )
        
        print(f"✅ Logger criado")
        print(f"   Pasta do paciente: {logger.patient_folder}")
        print(f"   Nome do arquivo: {logger.filename}")
        print(f"   Caminho completo: {logger.get_full_path()}")
        
        # Verificar se a pasta foi criada
        patient_dir = os.path.join(test_base_path, logger.patient_folder)
        if os.path.exists(patient_dir):
            print(f"✅ Pasta do paciente criada: {patient_dir}")
        else:
            print(f"❌ Pasta do paciente NÃO foi criada: {patient_dir}")
            continue
        
        # Simular algumas amostras de dados
        for i in range(50):
            eeg_data = np.random.randn(16) * 10
            marker = "T1" if i == 25 else None
            logger.log_sample(eeg_data.tolist(), marker)
        
        # Finalizar gravação
        logger.stop_logging()
        
        # Verificar se o arquivo foi criado
        full_path = logger.get_full_path()
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✅ Arquivo criado com sucesso: {size} bytes")
            created_files.append(full_path)
        else:
            print(f"❌ Arquivo NÃO foi criado: {full_path}")
    
    # Verificar estrutura final
    print(f"\n=== Estrutura Final ===")
    for root, dirs, files in os.walk(test_base_path):
        level = root.replace(test_base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\n=== Verificações Finais ===")
    
    # Verificar se P001 tem 2 arquivos (motor_imagery e rest)
    p001_files = [f for f in created_files if "P001_João_Silva" in f]
    print(f"Arquivos do P001 (João Silva): {len(p001_files)} arquivos")
    if len(p001_files) == 2:
        print("✅ P001 tem 2 arquivos corretos (motor_imagery e rest)")
    else:
        print("❌ P001 deveria ter 2 arquivos")
    
    # Verificar se cada paciente tem sua pasta separada
    patient_dirs = [d for d in os.listdir(test_base_path) if os.path.isdir(os.path.join(test_base_path, d))]
    expected_dirs = ["P001_João_Silva", "P002_Maria_Santos", "P003_Ana_Costa-Lima"]
    
    print(f"Pastas criadas: {patient_dirs}")
    
    all_dirs_exist = all(d in patient_dirs for d in expected_dirs)
    if all_dirs_exist:
        print("✅ Todas as pastas de pacientes foram criadas corretamente")
    else:
        print("❌ Algumas pastas de pacientes estão faltando")
    
    # Verificar sanitização de nomes
    ana_dir = [d for d in patient_dirs if "Ana_Costa-Lima" in d or "Ana_Costa_Lima" in d]
    if ana_dir:
        print(f"✅ Nome com caracteres especiais sanitizado: {ana_dir[0]}")
    else:
        print("❌ Sanitização de nome não funcionou")
    
    # Limpeza
    print(f"\n=== Limpeza ===")
    try:
        shutil.rmtree(test_base_path)
        print("✅ Pasta de teste removida")
    except Exception as e:
        print(f"⚠️  Erro ao remover pasta de teste: {e}")
    
    print(f"\n🎉 TESTE CONCLUÍDO!")
    return all_dirs_exist and len(p001_files) == 2

def test_patient_name_sanitization():
    """Testa a sanitização de nomes de pacientes"""
    print("\n=== Teste de Sanitização de Nomes ===")
    
    test_cases = [
        ("João Silva", "João_Silva"),
        ("Maria-José Santos", "Maria-José_Santos"),
        ("Ana/Costa\\Lima", "AnaCosta Lima"),  # Caracteres inválidos removidos
        ("Pedro<>Test:File", "PedroTestFile"),
        ("Nome Muito Longo Para Ser Usado Como Nome De Pasta Que Deveria Ser Truncado", "Nome_Muito_Longo_Para_Ser_Usado_Como_Nome_De_Past"),
    ]
    
    # Importar método de sanitização
    from openbci_csv_logger import OpenBCICSVLogger
    
    logger = OpenBCICSVLogger("TEST", "test", "test", "../data/test")
    
    all_passed = True
    for original, expected in test_cases:
        result = logger._sanitize_filename(original)
        if expected in result:  # Verificação parcial para casos complexos
            print(f"✅ '{original}' -> '{result}'")
        else:
            print(f"❌ '{original}' -> '{result}' (esperado algo como '{expected}')")
            all_passed = False
    
    logger.close()  # Limpar
    
    return all_passed

if __name__ == "__main__":
    # Executar testes
    test1_passed = test_patient_folder_organization()
    test2_passed = test_patient_name_sanitization()
    
    print(f"\n=== RESULTADO DOS TESTES ===")
    print(f"Organização por Pastas: {'✅ PASSOU' if test1_passed else '❌ FALHOU'}")
    print(f"Sanitização de Nomes: {'✅ PASSOU' if test2_passed else '❌ FALHOU'}")
    
    if test1_passed and test2_passed:
        print("🎉 TODOS OS TESTES PASSARAM - ORGANIZAÇÃO POR PACIENTE IMPLEMENTADA!")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM - VERIFICAR PROBLEMAS")
