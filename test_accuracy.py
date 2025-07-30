#!/usr/bin/env python3
"""
Script para testar o sistema de acurácia
"""

import sys
import os

# Adicionar o caminho do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_sender import UDP
import time

def test_accuracy_messages():
    """Testa mensagens de acurácia"""
    print("=== Teste do Sistema de Acurácia ===")
    print("1. Iniciando servidor UDP...")
    
    try:
        # Inicializar servidor
        UDP.init_zmq_socket()
        print("✓ Servidor UDP iniciado")
        
        print("\n2. Aguardando 2 segundos para conexão...")
        time.sleep(2)
        
        print("\n3. Enviando mensagens de teste:")
        
        # Teste 1: Flor vermelha + ação esquerda (CORRETO)
        print("  → RED_FLOWER + TRIGGER_ACTION_LEFT (esperado: CORRETO)")
        UDP.enviar_sinal('redleft')
        time.sleep(1)
        
        # Teste 2: Flor vermelha + ação direita (ERRO)
        print("  → RED_FLOWER + TRIGGER_ACTION_RIGHT (esperado: ERRO)")
        UDP.enviar_sinal('redright')
        time.sleep(1)
        
        # Teste 3: Flor azul + ação direita (CORRETO)
        print("  → BLUE_FLOWER + TRIGGER_ACTION_RIGHT (esperado: CORRETO)")
        UDP.enviar_sinal('blueright')
        time.sleep(1)
        
        # Teste 4: Flor azul + ação esquerda (ERRO)
        print("  → BLUE_FLOWER + TRIGGER_ACTION_LEFT (esperado: ERRO)")
        UDP.enviar_sinal('blueleft')
        time.sleep(1)
        
        print("\n4. Teste concluído!")
        print("Acurácia esperada: 50% (2/4)")
        
    except Exception as e:
        print(f"Erro durante o teste: {e}")
    finally:
        print("\n5. Parando servidor...")
        UDP.stop_zmq_socket()
        print("✓ Servidor parado")

if __name__ == "__main__":
    test_accuracy_messages()
