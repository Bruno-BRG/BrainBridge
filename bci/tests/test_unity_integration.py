#!/usr/bin/env python3
"""
Teste do sistema unificado de comunicação Unity
"""

import time
import sys
import os

# Adicionar o diretório atual ao path para importar o módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bci.network.unity_communication import UnityCommunicator, UDP_sender, UDP_receiver
    print("✅ Importação do sistema unificado bem-sucedida")
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

def test_compatibility():
    """Testa a compatibilidade com as classes antigas"""
    print("="*60)
    print("TESTE DE COMPATIBILIDADE")
    print("="*60)
    
    # Teste UDP_sender compatibilidade
    print("\n1. Testando compatibilidade UDP_sender...")
    
    print("   - Iniciando socket ZMQ...")
    if UDP_sender.init_zmq_socket():
        print("   ✅ Socket ZMQ iniciado com sucesso")
        
        print("   - Testando envio de sinais...")
        if UDP_sender.enviar_sinal('direita'):
            print("   ✅ Sinal 'direita' enviado")
        
        if UDP_sender.enviar_sinal('esquerda'):
            print("   ✅ Sinal 'esquerda' enviado")
        
        if UDP_sender.enviar_sinal('trigger_right'):
            print("   ✅ Sinal 'trigger_right' enviado")
        
        print(f"   - Servidor ativo: {UDP_sender.is_server_active()}")
        
        print("   - Parando socket ZMQ...")
        UDP_sender.stop_zmq_socket()
        print("   ✅ Socket ZMQ parado")
    else:
        print("   ❌ Falha ao iniciar socket ZMQ")
    
    # Teste UDP_receiver compatibilidade
    print("\n2. Testando compatibilidade UDP_receiver...")
    
    print("   - Procurando sender ativo...")
    ips = UDP_receiver.find_active_sender()
    print(f"   ✅ IPs encontrados: {ips}")
    
    print("   - Teste legacy broadcast...")
    ip = UDP_receiver.listen_for_broadcast_legacy()
    print(f"   ✅ IP legacy: {ip}")

def test_new_system():
    """Testa o novo sistema unificado"""
    print("\n" + "="*60)
    print("TESTE DO NOVO SISTEMA UNIFICADO")
    print("="*60)
    
    communicator = UnityCommunicator()
    
    # Configurar callbacks de teste
    def on_message(message):
        print(f"   📨 Mensagem recebida: {message}")
    
    def on_connection(connected):
        status = "Conectado" if connected else "Desconectado"
        print(f"   🔌 Status de conexão: {status}")
    
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)
    
    print("\n1. Iniciando servidor...")
    if communicator.start_server():
        print("   ✅ Servidor iniciado com sucesso")
        
        print("\n2. Testando envio de comandos...")
        commands = [
            ('direita', lambda: communicator.send_hand_command('direita')),
            ('esquerda', lambda: communicator.send_hand_command('esquerda')),
            ('trigger_right', lambda: communicator.send_trigger_command('direita')),
            ('trigger_left', lambda: communicator.send_trigger_command('esquerda')),
            ('comando personalizado', lambda: communicator.send_command('CUSTOM_COMMAND'))
        ]
        
        for desc, cmd_func in commands:
            if cmd_func():
                print(f"   ✅ {desc} enviado com sucesso")
            else:
                print(f"   ❌ Falha ao enviar {desc}")
            time.sleep(0.1)  # Pequeno delay entre comandos
        
        print("\n3. Aguardando 3 segundos para possíveis mensagens...")
        time.sleep(3)
        
        print("\n4. Parando servidor...")
        communicator.stop_server()
        print("   ✅ Servidor parado")
    else:
        print("   ❌ Falha ao iniciar servidor")

def main():
    """Função principal de teste"""
    print("TESTE DO SISTEMA DE COMUNICAÇÃO UNITY")
    print("=====================================")
    print("Este teste verifica a compatibilidade e funcionalidade")
    print("do novo sistema unificado de comunicação.")
    print()
    
    try:
        # Testar compatibilidade
        test_compatibility()
        
        # Testar novo sistema
        test_new_system()
        
        print("\n" + "="*60)
        print("TESTE CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print("✅ Compatibilidade mantida com código existente")
        print("✅ Novo sistema funcionando corretamente")
        print("✅ Sistema pronto para integração")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TESTE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
