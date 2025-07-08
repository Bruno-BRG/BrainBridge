"""
Teste da Funcionalidade de Jogo
Verifica se o botão muda para "Iniciar Jogo" quando a tarefa Jogo é selecionada.
"""

import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt5.QtWidgets import QApplication

def test_game_button_functionality():
    """Testa a funcionalidade do botão de jogo"""
    print("🎮 Testando Funcionalidade de Jogo...")
    
    app = QApplication([])
    
    try:
        from bci_interface import BCIMainWindow
        
        # Criar janela principal
        window = BCIMainWindow()
        
        # Acessar widget de streaming
        streaming_widget = window.streaming_widget
        
        # Teste 1: Verificar estado inicial
        print("\n📋 Teste 1: Estado Inicial")
        initial_text = streaming_widget.record_btn.text()
        assert initial_text == "Iniciar Gravação", f"Texto inicial incorreto: {initial_text}"
        print("✅ Texto inicial correto: 'Iniciar Gravação'")
        
        # Teste 2: Selecionar Jogo e verificar mudança
        print("\n🎮 Teste 2: Seleção de Jogo")
        streaming_widget.task_combo.setCurrentText("Jogo")
        
        # Disparar callback manualmente (já que não temos GUI rodando)
        streaming_widget.on_task_changed()
        
        game_text = streaming_widget.record_btn.text()
        assert game_text == "Iniciar Jogo", f"Texto de jogo incorreto: {game_text}"
        print("✅ Texto mudou corretamente para: 'Iniciar Jogo'")
        
        # Teste 3: Voltar para Baseline
        print("\n📊 Teste 3: Voltar para Baseline")
        streaming_widget.task_combo.setCurrentText("Baseline")
        streaming_widget.on_task_changed()
        
        baseline_text = streaming_widget.record_btn.text()
        assert baseline_text == "Iniciar Gravação", f"Texto de baseline incorreto: {baseline_text}"
        print("✅ Texto voltou para: 'Iniciar Gravação'")
        
        # Teste 4: Testar outras tarefas
        print("\n🔄 Teste 4: Outras Tarefas")
        for task in ["Treino", "Teste"]:
            streaming_widget.task_combo.setCurrentText(task)
            streaming_widget.on_task_changed()
            
            task_text = streaming_widget.record_btn.text()
            assert task_text == "Iniciar Gravação", f"Texto para {task} incorreto: {task_text}"
            print(f"✅ {task}: 'Iniciar Gravação'")
        
        # Teste 5: Simular estado de gravação com Jogo
        print("\n🔴 Teste 5: Estado de Gravação com Jogo")
        streaming_widget.task_combo.setCurrentText("Jogo")
        streaming_widget.is_recording = True
        streaming_widget.update_record_button_text()
        
        recording_game_text = streaming_widget.record_btn.text()
        assert recording_game_text == "Parar Jogo", f"Texto durante jogo incorreto: {recording_game_text}"
        print("✅ Durante gravação de jogo: 'Parar Jogo'")
        
        # Teste 6: Simular parada de gravação
        print("\n⏹️  Teste 6: Parar Gravação de Jogo")
        streaming_widget.is_recording = False
        streaming_widget.update_record_button_text()
        
        stopped_game_text = streaming_widget.record_btn.text()
        assert stopped_game_text == "Iniciar Jogo", f"Texto após parar jogo incorreto: {stopped_game_text}"
        print("✅ Após parar jogo: 'Iniciar Jogo'")
        
        print("\n🎉 TODOS OS TESTES DE JOGO PASSARAM!")
        print("📊 Funcionalidades testadas:")
        print("• ✅ Mudança automática do botão para 'Iniciar Jogo'")
        print("• ✅ Estado de gravação: 'Parar Jogo'")
        print("• ✅ Retorno ao estado inicial")
        print("• ✅ Outras tarefas mantêm 'Iniciar Gravação'")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        return False
    
    finally:
        app.quit()


if __name__ == "__main__":
    success = test_game_button_functionality()
    if success:
        print("\n✅ TESTE DE JOGO: SUCESSO")
    else:
        print("\n❌ TESTE DE JOGO: FALHA")
        sys.exit(1)
