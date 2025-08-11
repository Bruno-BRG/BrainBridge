# Guia de Uso - Sistema Unity Integrado

## 🎉 Integração Concluída!

O sistema Unity foi **completamente integrado** substituindo `UDP_sender.py` e `udp_receiver.py` pelo sistema unificado baseado no `sender.py`. **Não há breaking changes** - todo código existente continua funcionando.

## ✅ O que foi feito

1. **Criado**: `bci/network/unity_communication.py` - Sistema unificado
2. **Substituído**: `UDP_sender.py` → Redirecionamento compatível  
3. **Substituído**: `udp_receiver.py` → Redirecionamento compatível
4. **Atualizado**: `bci/ui/streaming_widget.py` - Integração com callbacks
5. **Backup**: Arquivos originais salvos como `.backup`

## 🚀 Como usar (código existente continua igual)

### Na UI (StreamingWidget)
```python
# Funciona exatamente como antes
if not self.udp_server_active:
    UDP_sender.init_zmq_socket()  # ✅ Funciona
    self.udp_server_active = True

# Envio de comandos
UDP_sender.enviar_sinal('direita')     # ✅ Funciona
UDP_sender.enviar_sinal('esquerda')    # ✅ Funciona  
UDP_sender.enviar_sinal('trigger_right') # ✅ Funciona

# Parar servidor
UDP_sender.stop_zmq_socket()           # ✅ Funciona
```

### Código de teste/debugging
```python
from bci.network.UDP_sender import UDP_sender
from bci.network.udp_receiver import UDP_receiver

# Tudo funciona como antes
UDP_sender.init_zmq_socket()
ips = UDP_receiver.find_active_sender()
UDP_sender.enviar_sinal('direita')
```

## 🔧 Novo sistema avançado (opcional)

Para novos desenvolvimentos, você pode usar diretamente:

```python
from bci.network.unity_communication import UnityCommunicator

# Sistema singleton - uma instância para toda aplicação
communicator = UnityCommunicator()

# Callbacks para eventos
def on_message(msg):
    print(f"Unity disse: {msg}")

def on_connection(connected):
    print(f"Unity {'conectou' if connected else 'desconectou'}")

communicator.set_message_callback(on_message)
communicator.set_connection_callback(on_connection)

# Iniciar servidor
communicator.start_server()

# Enviar comandos
communicator.send_hand_command('direita')    # RIGHT_HAND_CLOSE
communicator.send_hand_command('esquerda')   # LEFT_HAND_CLOSE
communicator.send_trigger_command('direita') # TRIGGER_RIGHT
communicator.send_command('CUSTOM_COMMAND')  # Comando personalizado

# Parar quando necessário
communicator.stop_server()
```

## 📡 Protocolo de Comunicação

O sistema usa três canais de comunicação:

- **UDP 12346**: Broadcast contínuo de IPs para descoberta
- **TCP 12345**: Conexão bidirecional com Unity
- **ZMQ 5555**: Publisher para envio rápido de comandos

## 🔍 Vantagens da integração

### Antes (múltiplos arquivos)
- `UDP_sender.py` - Sistema complexo com threads manuais
- `udp_receiver.py` - Receiver separado com lógica duplicada
- `sender.py` - Terceiro arquivo com padrão diferente

### Depois (sistema unificado)
- ✅ **Um arquivo principal** (`unity_communication.py`)
- ✅ **100% compatível** com código existente
- ✅ **Baseado no sender.py** que você sabia que funcionava
- ✅ **Sistema de callbacks** para eventos assíncronos
- ✅ **Gerenciamento automático** de recursos
- ✅ **Padrão singleton** evita conflitos
- ✅ **Melhor tratamento de erros**

## 🧪 Testado e Aprovado

```bash
# Todos estes testes passaram:
✅ Compatibilidade UDP_sender
✅ Compatibilidade UDP_receiver  
✅ Integração com StreamingWidget
✅ Sistema unificado funcionando
✅ Envio de comandos
✅ Gerenciamento de recursos
✅ Callbacks funcionando
```

## 🚫 Zero Breaking Changes

Todo código existente continua funcionando:
- ✅ `UDP_sender.init_zmq_socket()`
- ✅ `UDP_sender.enviar_sinal()`
- ✅ `UDP_sender.stop_zmq_socket()`
- ✅ `UDP_receiver.find_active_sender()`
- ✅ `UDP_receiver.listen_for_broadcast()`
- ✅ Interface da UI inalterada

## 💾 Arquivos de Backup

Caso precise dos originais:
- `UDP_sender.py.backup`
- `udp_receiver.py.backup` 
- `sender.py.backup`

## 🎯 Resultado Final

**Você agora tem um sistema Unity mais robusto, baseado no padrão do `sender.py` que funcionava, mas mantendo 100% de compatibilidade com todo código existente.**

Pode usar o sistema normalmente - não precisa mudar nada no código atual!
