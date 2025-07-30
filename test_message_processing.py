#!/usr/bin/env python3
"""
Teste simples para verificar se o processo_accuracy_message funciona
"""

# Simular a função process_accuracy_message
def process_accuracy_message(message):
    """Simula o processamento de mensagem de acurácia"""
    print(f"🔍 DEBUG: Mensagem recebida para acurácia: '{message}'")
    
    accuracy_data = []
    accuracy_correct = 0
    accuracy_total = 0
    
    try:
        # Parse da mensagem: "RED_FLOWER,TRIGGER_ACTION_LEFT"
        if "," in message:
            parts = message.strip().split(",")
            if len(parts) == 2:
                flower_color = parts[0].strip()
                trigger_action = parts[1].strip()
                
                # Mapear cor para ação esperada
                if flower_color == "RED_FLOWER":
                    expected_action = "LEFT"  # Vermelho = esquerda esperada
                elif flower_color == "BLUE_FLOWER":
                    expected_action = "RIGHT"  # Azul = direita esperada
                else:
                    print(f"Cor de flor desconhecida: {flower_color}")
                    return
                
                # Mapear trigger para ação real
                if trigger_action == "TRIGGER_ACTION_LEFT":
                    real_action = "LEFT"
                elif trigger_action == "TRIGGER_ACTION_RIGHT":
                    real_action = "RIGHT"
                else:
                    print(f"Trigger desconhecido: {trigger_action}")
                    return
                
                # Calcular se foi acerto
                is_correct = (expected_action == real_action)
                
                # Atualizar contadores
                accuracy_total += 1
                if is_correct:
                    accuracy_correct += 1
                
                # Armazenar dados
                accuracy_data.append((expected_action, real_action, is_correct))
                
                # Log para debug
                status = "✓" if is_correct else "✗"
                print(f"Acurácia: {flower_color} -> {expected_action} vs {trigger_action} -> {real_action} {status}")
                
                # Simular atualização da interface
                if accuracy_total == 0:
                    accuracy_percent = 0
                else:
                    accuracy_percent = (accuracy_correct / accuracy_total) * 100
                    
                print(f"Acurácia atual: {accuracy_percent:.1f}% ({accuracy_correct}/{accuracy_total})")
                
        else:
            print(f"Formato de mensagem inválido: {message}")
            
    except Exception as e:
        print(f"Erro ao processar mensagem de acurácia: {e}")

def test_messages():
    """Testa várias mensagens"""
    test_cases = [
        "RED_FLOWER,TRIGGER_ACTION_LEFT",    # Correto
        "RED_FLOWER,TRIGGER_ACTION_RIGHT",   # Erro
        "BLUE_FLOWER,TRIGGER_ACTION_RIGHT",  # Correto
        "BLUE_FLOWER,TRIGGER_ACTION_LEFT",   # Erro
        "INVALID_MESSAGE",                   # Formato inválido
    ]
    
    print("=== Teste do Processamento de Mensagens de Acurácia ===\n")
    
    for i, message in enumerate(test_cases, 1):
        print(f"Teste {i}: {message}")
        process_accuracy_message(message)
        print("-" * 50)

if __name__ == "__main__":
    test_messages()
