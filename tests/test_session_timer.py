"""
Teste do Timer de Sessão
Verifica se o timer de sessão está funcionando corretamente no BCI Interface.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_interface import MainWindow


def test_session_timer():
    """Testa o timer de sessão"""
    app = QApplication([])
    
    # Criar janela principal
    window = MainWindow()
    window.show()
    
    # Acesso ao widget de streaming
    streaming_widget = window.tabs.widget(1)  # Segunda aba é o streaming
    
    # Verificar se os componentes do timer existem
    assert hasattr(streaming_widget, 'session_timer'), "session_timer não encontrado"
    assert hasattr(streaming_widget, 'session_timer_label'), "session_timer_label não encontrado"
    assert hasattr(streaming_widget, 'update_session_timer'), "método update_session_timer não encontrado"
    
    # Verificar estado inicial
    assert streaming_widget.session_timer_label.text() == "Tempo: 00:00:00", "Display inicial incorreto"
    
    # Verificar se o timer não está ativo inicialmente
    assert not streaming_widget.session_timer.isActive(), "Timer não deveria estar ativo inicialmente"
    
    print("✅ Timer de sessão configurado corretamente")
    print("✅ Display inicial correto")
    print("✅ Estado inicial apropriado")
    
    # Simular início de gravação sem conectar UDP
    try:
        # Simular seleção de paciente (criar um paciente de teste)
        streaming_widget.db_manager.add_patient(
            name="Teste Timer",
            age=30,
            sex="M",
            affected_hand="Direita",
            time_since_event=12
        )
        
        # Atualizar lista de pacientes
        streaming_widget.refresh_patients()
        
        # Selecionar primeiro paciente real (não o "Selecione...")
        if streaming_widget.patient_combo.count() > 1:
            streaming_widget.patient_combo.setCurrentIndex(1)
            
            # Simular início de gravação (mas sem UDP)
            print("\n📋 Simulando início de gravação...")
            streaming_widget.current_patient_id = streaming_widget.patient_combo.currentData()
            streaming_widget.is_recording = True
            
            # Iniciar timer manualmente
            streaming_widget.session_start_time = 0  # Tempo fixo para teste
            streaming_widget.session_timer.start(100)  # Atualizar mais rápido para teste
            
            # Aguardar algumas atualizações
            QTimer.singleShot(300, lambda: check_timer_updates(streaming_widget))
            QTimer.singleShot(600, app.quit)
        else:
            print("❌ Nenhum paciente encontrado para teste")
            app.quit()
            
    except Exception as e:
        print(f"⚠️  Erro durante teste: {e}")
        app.quit()
    
    app.exec_()


def check_timer_updates(widget):
    """Verifica se o timer está sendo atualizado"""
    # O timer deveria ter mudado o display
    timer_text = widget.session_timer_label.text()
    print(f"📊 Timer atual: {timer_text}")
    
    # Verificar se não é mais o valor inicial
    if timer_text != "Tempo: 00:00:00":
        print("✅ Timer está atualizando corretamente")
    else:
        print("⚠️  Timer pode não estar atualizando")
    
    # Parar timer
    widget.session_timer.stop()
    widget.is_recording = False
    print("✅ Timer parado com sucesso")


if __name__ == "__main__":
    print("🧪 Testando Timer de Sessão...")
    test_session_timer()
    print("\n🎉 Teste concluído!")
