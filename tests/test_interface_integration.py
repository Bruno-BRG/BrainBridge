"""
Teste de Integração - Interface + Logger OpenBCI
Verifica se o sistema funciona de ponta a ponta
"""
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from openbci_csv_logger import OpenBCICSVLogger
import numpy as np

def test_start_stop_recording():
    """Testa o ciclo completo de start/stop como a interface faz"""
    
    print("=== Teste de Integração Interface + Logger ===")
    
    # Simular como a interface inicia uma gravação
    logger = OpenBCICSVLogger(
        patient_id="TEST_INTEGRATION", 
        task="motor_imagery",
        base_path="../data/recordings"
    )
    
    print(f"✅ Logger criado: {logger.filename}")
    
    # Simular dados por alguns segundos
    for i in range(100):
        eeg_data = np.random.randn(16) * 10
        marker = None
        
        if i == 30:
            marker = "T1"
            print(f"✅ Marcador T1 adicionado na amostra {i}")
        
        logger.log_sample(eeg_data.tolist(), marker)
    
    # Simular como a interface para a gravação
    try:
        logger.stop_logging()  # Método esperado pela interface
        print("✅ stop_logging() executado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao executar stop_logging(): {e}")
        return False
    
    # Verificar se arquivo foi criado
    filepath = os.path.join("../data/recordings", logger.filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ Arquivo criado com sucesso: {size} bytes")
        
        # Verificar conteúdo
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        print(f"✅ Total de linhas: {len(lines)}")
        
        # Verificar headers
        expected_headers = [
            "%OpenBCI Raw EXG Data",
            "%Number of channels = 16", 
            "%Sample Rate = 125 Hz",
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy"
        ]
        
        for i, expected in enumerate(expected_headers):
            if lines[i].strip() == expected:
                print(f"✅ Header {i}: {expected}")
            else:
                print(f"❌ Header {i} incorreto: esperado '{expected}', encontrado '{lines[i].strip()}'")
                return False
        
        # Verificar se marcador T1 está presente
        t1_found = False
        for line in lines:
            if line.strip().endswith(",T1"):
                t1_found = True
                print(f"✅ Marcador T1 encontrado na linha")
                break
        
        if not t1_found:
            print("❌ Marcador T1 não encontrado")
            return False
        
        print("✅ Teste de integração passou em todas as verificações!")
        return True
    
    else:
        print("❌ Arquivo não foi criado")
        return False

def test_baseline_integration():
    """Testa o sistema de baseline"""
    print("\n=== Teste Baseline Integration ===")
    
    logger = OpenBCICSVLogger(
        patient_id="TEST_BASELINE", 
        task="baseline_test",
        base_path="../data/recordings"
    )
    
    # Ativar baseline (como a interface faria)
    logger.start_baseline()
    print("✅ Baseline iniciado")
    
    # Verificar status
    if logger.is_baseline_active():
        print("✅ Baseline está ativo")
        remaining = logger.get_baseline_remaining()
        print(f"✅ Tempo restante: {remaining:.1f}s")
    else:
        print("❌ Baseline não está ativo")
        return False
    
    # Tentar adicionar marcador (deveria ser bloqueado)
    result = logger.add_marker("T1")
    if result is None:
        print("✅ Marcador corretamente bloqueado durante baseline")
    else:
        print("❌ Marcador não foi bloqueado durante baseline")
        return False
    
    # Parar gravação
    logger.stop_logging()
    print("✅ Gravação de baseline parada com sucesso")
    
    return True

if __name__ == "__main__":
    # Criar diretório se não existir
    os.makedirs("../data/recordings", exist_ok=True)
    
    # Executar testes
    test1_passed = test_start_stop_recording()
    test2_passed = test_baseline_integration()
    
    print(f"\n=== RESULTADO DOS TESTES ===")
    print(f"Teste Start/Stop: {'✅ PASSOU' if test1_passed else '❌ FALHOU'}")
    print(f"Teste Baseline: {'✅ PASSOU' if test2_passed else '❌ FALHOU'}")
    
    if test1_passed and test2_passed:
        print("🎉 TODOS OS TESTES PASSARAM - SISTEMA PRONTO!")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM - VERIFICAR PROBLEMAS")
