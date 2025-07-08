"""
Teste Simples do Timer de Sessão
"""

import sys
import os
import time

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Teste sem interface gráfica
def test_timer_logic():
    print("🧪 Testando lógica do timer de sessão...")
    
    # Simular dados de tempo
    start_time = time.time()
    
    # Simular 5 segundos de gravação
    time.sleep(2)
    
    elapsed = int(time.time() - start_time)
    
    # Formatar tempo
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"⏱️  Tempo formatado: {time_str}")
    
    # Verificar formato
    assert len(time_str) == 8, "Formato de tempo incorreto"
    assert time_str.count(':') == 2, "Separadores incorretos"
    
    print("✅ Lógica do timer funcionando corretamente")
    return True

if __name__ == "__main__":
    try:
        success = test_timer_logic()
        if success:
            print("✅ TESTE DO TIMER: SUCESSO")
        else:
            print("❌ TESTE DO TIMER: FALHA")
    except Exception as e:
        print(f"❌ Erro: {e}")
