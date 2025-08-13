import zmq
import socket
import time
import threading

class TestReceiver:
    def __init__(self):
        self.context = None
        self.zmq_socket = None
        self.is_connected = False
        
    @staticmethod
    def find_active_sender():
        """
        Procura por senders ativos nas portas ZMQ conhecidas.
        Retorna o IP local se encontrar algum sender ativo.
        """
        try:
            # Obtém IP local
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            print(f"[TEST] Procurando senders ativos em {local_ip}...")
            return [local_ip]  # Retorna como lista para compatibilidade
            
        except Exception as e:
            print(f"[TEST] Erro ao obter IP local: {e}")
            return ['127.0.0.1']

    @staticmethod
    def try_connect_to_sender(ips, zmq_ports=[5555, 5556]):
        """
        Tenta conectar nas portas 5555 (UDP_sender) e 5556 (test_sender).
        Conecta em ambas para receber mensagens de qualquer máquina.
        """
        context = zmq.Context()
        socket_zmq = context.socket(zmq.SUB)
        socket_zmq.setsockopt_string(zmq.SUBSCRIBE, "")  # Recebe todas as mensagens
        socket_zmq.setsockopt(zmq.RCVTIMEO, 3000)  # Timeout de 3 segundos
        
        connected_endpoints = []
        
        for ip in ips:
            for port in zmq_ports:
                try:
                    endpoint = f"tcp://{ip}:{port}"
                    print(f"[TEST] Tentando conectar em {endpoint}...")
                    socket_zmq.connect(endpoint)
                    connected_endpoints.append(f"{ip}:{port}")
                    print(f"[TEST] Conectado em {endpoint}")
                        
                except Exception as e:
                    print(f"[TEST] Falha ao conectar em {ip}:{port}: {e}")
                    continue
        
        if connected_endpoints:
            print(f"[TEST] ✅ Conectado aos endpoints: {', '.join(connected_endpoints)}")
            print(f"[TEST] Aguardando mensagens de qualquer máquina...")
            return socket_zmq, context, ', '.join(connected_endpoints)
        else:
            # Se chegou aqui, nenhuma conexão funcionou
            socket_zmq.close()
            context.term()
            return None, None, None

    def connect(self, timeout=10.0):
        """
        Método principal para conectar ao sender aproveitando conexão TCP existente.
        """
        # Procura por senders ativos localmente
        ips = self.find_active_sender()
        if not ips:
            print("[TEST] Não foi possível obter IP local")
            return False
            
        # Tenta conectar usando ZMQ
        self.zmq_socket, self.context, connected_endpoint = self.try_connect_to_sender(ips)
        if self.zmq_socket is None:
            print("[TEST] ❌ Não foi possível conectar a nenhum sender")
            print("[TEST] Certifique-se de que o UDP_sender.py principal ou test_sender.py estão rodando")
            return False
            
        self.is_connected = True
        print(f"[TEST] ✅ Conectado com sucesso ao sender em {connected_endpoint}")
        return True

    def receive_messages(self, callback=None):
        """
        Recebe mensagens continuamente.
        callback: função opcional para processar cada mensagem recebida
        """
        if not self.is_connected:
            print("[TEST] Erro: Não conectado. Chame connect() primeiro.")
            return
            
        print("\n[TEST] Aguardando comandos...")
        print("[TEST] Pressione Ctrl+C para sair")
        
        try:
            while True:
                try:
                    # Remove timeout para operação normal
                    self.zmq_socket.setsockopt(zmq.RCVTIMEO, -1)
                    message = self.zmq_socket.recv_string()
                    print(f"[TEST] Comando recebido: {message}")
                    
                    # Chama callback se fornecido
                    if callback:
                        callback(message)
                        
                except zmq.Again:
                    continue
                except Exception as e:
                    print(f"[TEST] Erro ao receber mensagem: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n[TEST] Encerrando recepção...")
        finally:
            self.disconnect()

    def disconnect(self):
        """
        Desconecta e limpa recursos.
        """
        if self.zmq_socket is not None:
            try:
                self.zmq_socket.close()
                print("[TEST] Socket ZMQ fechado")
            except Exception as e:
                print(f"[TEST] Erro ao fechar socket ZMQ: {e}")
            finally:
                self.zmq_socket = None
                
        if self.context is not None:
            try:
                self.context.term()
                print("[TEST] Contexto ZMQ terminado")
            except Exception as e:
                print(f"[TEST] Erro ao terminar contexto ZMQ: {e}")
            finally:
                self.context = None
                
        self.is_connected = False
        print("[TEST] Desconectado com sucesso")

# Função legacy para compatibilidade (agora sem broadcast)
def listen_for_broadcast():
    """Versão legacy que retorna IP local"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def main():
    """Método main melhorado para aproveitar conexão TCP existente"""
    print("="*60)
    print("TEST RECEIVER - Aproveitando Conexão TCP Existente")
    print("="*60)
    print("[TEST] Este receiver aproveita a conexão TCP já estabelecida")
    print("[TEST] pelo sistema principal (UDP_sender.py)")
    print("="*60)
    
    receiver = TestReceiver()
    
    # Conecta ao sender (sem broadcast, usando conexão TCP existente)
    if not receiver.connect():
        print("[TEST] ❌ Falha na conexão. Possíveis soluções:")
        print("[TEST] 1. Execute primeiro o UDP_sender.py principal")
        print("[TEST] 2. Ou execute o test_sender.py")
        return
        
    # Define callback específico para processar mensagens de teste
    def process_test_message(message):
        print(f"[TEST] 📨 Processando: {message}")
        
        # Processamento específico para comandos de teste
        if "HAND" in message:
            if "RIGHT" in message:
                print("[TEST]   ➡️  Ação de mão direita detectada!")
            elif "LEFT" in message:
                print("[TEST]   ⬅️  Ação de mão esquerda detectada!")
        elif "TRIGGER" in message:
            if "RIGHT" in message:
                print("[TEST]   🎯 Gatilho direito ativado!")
            elif "LEFT" in message:
                print("[TEST]   🎯 Gatilho esquerdo ativado!")
        elif "FLOWER" in message:
            if "RED_FLOWER" in message:
                if "CORRECT" in message:
                    print("[TEST]   🌹 Flor vermelha - Resposta CORRETA!")
                elif "WRONG" in message:
                    print("[TEST]   🌹 Flor vermelha - Resposta ERRADA!")
            elif "BLUE_FLOWER" in message:
                if "TRIGGER_ACTION_LEFT" in message:
                    print("[TEST]   🌸 Flor azul - Gatilho esquerdo!")
                elif "TRIGGER_ACTION_RIGHT" in message:
                    print("[TEST]   🌸 Flor azul - Gatilho direito!")
        else:
            print(f"[TEST]   ❓ Comando não reconhecido: {message}")
            
    # Inicia recepção de mensagens
    receiver.receive_messages(callback=process_test_message)

def main_legacy():
    """Versão legacy do main para compatibilidade"""
    print("[TEST] Usando método legacy...")
    
    # Obtém IP local (sem broadcast)
    sender_ip = listen_for_broadcast()
    if not sender_ip:
        print("[TEST] Não foi possível obter IP local")
        return

    # Configura o socket ZMQ para receber as mensagens
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Recebe todas as mensagens
    
    # Conecta nas duas portas (5555 e 5556)
    connected_ports = []
    for port in [5555, 5556]:
        try:
            print(f"[TEST] Tentando conectar em tcp://{sender_ip}:{port}...")
            socket.connect(f"tcp://{sender_ip}:{port}")
            connected_ports.append(port)
            print(f"[TEST] Conectado na porta {port}")
        except Exception as e:
            print(f"[TEST] Falha na porta {port}: {e}")
            continue
    
    if not connected_ports:
        print("[TEST] Não foi possível conectar em nenhuma porta")
        socket.close()
        context.term()
        return
    
    print(f"\n[TEST] Conectado nas portas: {connected_ports}")
    print("[TEST] Aguardando comandos de qualquer máquina...")
    print("[TEST] Pressione Ctrl+C para sair")
    
    try:
        while True:
            # Recebe as mensagens
            message = socket.recv_string()
            print(f"[TEST] Comando recebido: {message}")
            
    except KeyboardInterrupt:
        print("\n[TEST] Encerrando comunicação...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
