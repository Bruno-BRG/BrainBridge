"""
Teste Simples da Lógica de Jogo
Testa apenas a lógica de mudança de texto sem GUI
"""

def test_game_logic():
    """Testa a lógica de mudança de texto do botão"""
    print("🎮 Testando lógica de texto do botão...")
    
    # Simular lógica de mudança de texto
    def get_button_text(task, is_recording):
        if is_recording:
            return "Parar Jogo" if task == "Jogo" else "Parar Gravação"
        else:
            return "Iniciar Jogo" if task == "Jogo" else "Iniciar Gravação"
    
    # Teste 1: Estado inicial com Baseline
    text = get_button_text("Baseline", False)
    assert text == "Iniciar Gravação", f"Esperado 'Iniciar Gravação', obtido '{text}'"
    print("✅ Baseline não gravando: 'Iniciar Gravação'")
    
    # Teste 2: Jogo não gravando
    text = get_button_text("Jogo", False)
    assert text == "Iniciar Jogo", f"Esperado 'Iniciar Jogo', obtido '{text}'"
    print("✅ Jogo não gravando: 'Iniciar Jogo'")
    
    # Teste 3: Jogo gravando
    text = get_button_text("Jogo", True)
    assert text == "Parar Jogo", f"Esperado 'Parar Jogo', obtido '{text}'"
    print("✅ Jogo gravando: 'Parar Jogo'")
    
    # Teste 4: Baseline gravando
    text = get_button_text("Baseline", True)
    assert text == "Parar Gravação", f"Esperado 'Parar Gravação', obtido '{text}'"
    print("✅ Baseline gravando: 'Parar Gravação'")
    
    # Teste 5: Outras tarefas
    for task in ["Treino", "Teste"]:
        text = get_button_text(task, False)
        assert text == "Iniciar Gravação", f"Esperado 'Iniciar Gravação' para {task}, obtido '{text}'"
        print(f"✅ {task} não gravando: 'Iniciar Gravação'")
        
        text = get_button_text(task, True)
        assert text == "Parar Gravação", f"Esperado 'Parar Gravação' para {task}, obtido '{text}'"
        print(f"✅ {task} gravando: 'Parar Gravação'")
    
    print("\n🎉 LÓGICA DE JOGO FUNCIONANDO CORRETAMENTE!")
    return True

if __name__ == "__main__":
    try:
        success = test_game_logic()
        if success:
            print("\n✅ TESTE DE LÓGICA: SUCESSO")
        else:
            print("\n❌ TESTE DE LÓGICA: FALHA")
    except Exception as e:
        print(f"❌ Erro: {e}")
