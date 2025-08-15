# unity_communication.py
"""
Sistema unificado de comunicação com Unity
Substitui UDP_sender.py e udp_receiver.py por uma abordagem mais simples e robusta
"""

import socket
import threading
import time
import zmq
from typing import Optional, List, Callable

class UnityCommunicator:
    """
    Classe unificada para comunicação com Unity usando TCP + ZMQ
    Combina as funcionalidades de UDP_sender e udp_receiver em uma única interface
    """
    
    # Configurações
    UDP_PORT = 12346      # porta para broadcast de IPs
    TCP_PORT = 12345      # porta para servidor TCP Unity
    ZMQ_PORT = 5555       # porta para ZMQ publisher
    BROADCAST_INTERVAL = 1.0
    BUFFER_SIZE = 4096
    
    # Variáveis de classe para singleton
    _instance: Optional['UnityCommunicator'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implementa padrão singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnityCommunicator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa o comunicador"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        
        # Estado da conexão
        self.is_active = False
        self.tcp_connected = False
        
        # Sockets e contextos
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.tcp_connection: Optional[socket.socket] = None
        
        # Threads de controle
        self.broadcast_thread: Optional[threading.Thread] = None
        self.tcp_server_thread: Optional[threading.Thread] = None
        self.tcp_handler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks para eventos
        self.on_message_received: Optional[Callable[[str], None]] = None
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
    
    @staticmethod
    def get_all_ips() -> List[str]:
        """
        Retorna lista de IPs IPv4 locais usando stdlib.
        """
        ips = set()
        try:
            hostname = socket.gethostname()
            for res in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ips.add(res[4][0])
        except Exception:
            pass
        if not ips:
            ips.add('127.0.0.1')
        return list(ips)
    
    def start_server(self) -> bool:
        """
        Inicia o servidor de comunicação
        Retorna True se iniciado com sucesso
        """
        if self.is_active:
            print("Servidor já está ativo")
            return True
            
        try:
            # Inicializar ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://*:{self.ZMQ_PORT}")
            
            # Reset do evento de parada
            self.stop_event.clear()
            
            # Iniciar broadcast UDP
            self.broadcast_thread = threading.Thread(
                target=self._broadcast_ips, 
                daemon=True
            )
            self.broadcast_thread.start()
            
            # Iniciar servidor TCP
            self.tcp_server_thread = threading.Thread(
                target=self._tcp_server, 
                daemon=True
            )
            self.tcp_server_thread.start()
            
            self.is_active = True
            print(f"Servidor iniciado - ZMQ: {self.ZMQ_PORT}, TCP: {self.TCP_PORT}, UDP: {self.UDP_PORT}")
            return True
            
        except Exception as e:
            print(f"Erro ao iniciar servidor: {e}")
            self.stop_server()
            return False
    
    def stop_server(self):
        """Para o servidor e limpa recursos"""
        self.is_active = False
        self.stop_event.set()
        
        # Aguardar threads terminarem
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2.0)
            
        if self.tcp_server_thread and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.join(timeout=2.0)
            
        if self.tcp_handler_thread and self.tcp_handler_thread.is_alive():
            self.tcp_handler_thread.join(timeout=2.0)
        
        # Fechar conexão TCP
        if self.tcp_connection:
            try:
                self.tcp_connection.close()
            except Exception:
                pass
            self.tcp_connection = None
            
        # Fechar ZMQ
        if self.zmq_socket:
            try:
                self.zmq_socket.close()
            except Exception:
                pass
            self.zmq_socket = None
            
        if self.zmq_context:
            try:
                self.zmq_context.term()
            except Exception:
                pass
            self.zmq_context = None
        
        # Atualizar estado
        if self.tcp_connected:
            self.tcp_connected = False
            if self.on_connection_changed:
                self.on_connection_changed(False)
        
        print("Servidor parado e recursos limpos")
    
    def send_command(self, command: str) -> bool:
        """
        Envia comando para Unity via ZMQ e TCP
        Retorna True se enviado com sucesso
        """
        if not self.is_active:
            print("Servidor não está ativo")
            return False
            
        success = False
        
        # Enviar via ZMQ (sempre disponível quando servidor ativo)
        if self.zmq_socket:
            try:
                self.zmq_socket.send_string(command)
                print(f"[ZMQ] Comando enviado: {command}")
                success = True
            except Exception as e:
                print(f"[ZMQ] Erro ao enviar: {e}")
        
        # Enviar via TCP se conectado
        if self.tcp_connected and self.tcp_connection:
            try:
                message = command + '\n'
                self.tcp_connection.sendall(message.encode('utf-8'))
                print(f"[TCP] Comando enviado: {command}")
                success = True
            except Exception as e:
                print(f"[TCP] Erro ao enviar: {e}")
                self.tcp_connected = False
                if self.on_connection_changed:
                    self.on_connection_changed(False)
        
        return success
    
    def send_hand_command(self, hand: str) -> bool:
        """
        Envia comando de mão (direita/esquerda)
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("RIGHT_HAND_CLOSE")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("LEFT_HAND_CLOSE")
        else:
            print(f"Comando de mão inválido: {hand}")
            return False
    
    def send_trigger_command(self, hand: str) -> bool:
        """
        Envia comando de trigger
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("TRIGGER_RIGHT")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("TRIGGER_LEFT")
        else:
            print(f"Comando de trigger inválido: {hand}")
            return False
    
    def _broadcast_ips(self):
        """
        Thread para broadcast dos IPs via UDP
        """
        ips = self.get_all_ips()
        message = ','.join(ips).encode('utf-8')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        print(f"[UDP] Iniciando broadcast: {ips}")
        
        try:
            while not self.stop_event.is_set():
                sock.sendto(message, ('<broadcast>', self.UDP_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
        except Exception as e:
            print(f"[UDP] Erro no broadcast: {e}")
        finally:
            sock.close()
            print("[UDP] Broadcast parado")
    
    def _tcp_server(self):
        """
        Thread para servidor TCP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Timeout para permitir verificação de stop_event
        
        try:
            sock.bind(('', self.TCP_PORT))
            sock.listen(1)
            print(f"[TCP] Servidor ouvindo na porta {self.TCP_PORT}")
            
            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    print(f"[TCP] Unity conectado de {addr}")
                    
                    self.tcp_connection = conn
                    self.tcp_connected = True
                    
                    if self.on_connection_changed:
                        self.on_connection_changed(True)
                    
                    # Iniciar thread para lidar com esta conexão
                    self.tcp_handler_thread = threading.Thread(
                        target=self._handle_tcp_connection,
                        args=(conn, addr),
                        daemon=True
                    )
                    self.tcp_handler_thread.start()
                    
                    # Aguardar conexão terminar antes de aceitar nova
                    self.tcp_handler_thread.join()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"[TCP] Erro no servidor: {e}")
                    break
                    
        except Exception as e:
            print(f"[TCP] Erro ao iniciar servidor: {e}")
        finally:
            sock.close()
            print("[TCP] Servidor TCP parado")
    
    def _handle_tcp_connection(self, conn: socket.socket, addr):
        """
        Lida com uma conexão TCP específica
        """
        try:
            conn.settimeout(1.0)
            
            while not self.stop_event.is_set() and self.tcp_connected:
                try:
                    data = conn.recv(self.BUFFER_SIZE)
                    if not data:
                        print("[TCP] Unity desconectou")
                        break
                        
                    message = data.decode('utf-8', errors='ignore').strip()
                    print(f"[TCP] Recebido: {message}")
                    
                    if self.on_message_received:
                        self.on_message_received(message)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[TCP] Erro na recepção: {e}")
                    break
                    
        finally:
            try:
                conn.close()
            except Exception:
                pass
            
            self.tcp_connection = None
            self.tcp_connected = False
            
            if self.on_connection_changed:
                self.on_connection_changed(False)
            
            print("[TCP] Conexão encerrada")
    
    def set_message_callback(self, callback: Callable[[str], None]):
        """Define callback para mensagens recebidas"""
        self.on_message_received = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Define callback para mudanças de conexão"""
        self.on_connection_changed = callback


# Classe para compatibilidade com código existente
class UDP_sender:
    """Classe de compatibilidade que mapeia para UnityCommunicator"""
    
    _communicator = UnityCommunicator()
    
    @classmethod
    def init_zmq_socket(cls, broadcast_duration=3.0):
        """Inicializa o sistema de comunicação"""
        return cls._communicator.start_server()
    
    @classmethod
    def stop_zmq_socket(cls):
        """Para o sistema de comunicação"""
        cls._communicator.stop_server()
    
    @classmethod
    def enviar_sinal(cls, action: str) -> bool:
        """Envia sinal de ação"""
        if action.lower() == 'direita':
            return cls._communicator.send_hand_command('direita')
        elif action.lower() == 'esquerda':
            return cls._communicator.send_hand_command('esquerda')
        elif action.lower() == 'trigger_right':
            return cls._communicator.send_trigger_command('direita')
        elif action.lower() == 'trigger_left':
            return cls._communicator.send_trigger_command('esquerda')
        else:
            return cls._communicator.send_command(action)
    
    @classmethod
    def is_server_active(cls) -> bool:
        """Verifica se o servidor está ativo"""
        return cls._communicator.is_active
    
    @classmethod
    def restart_broadcast(cls, duration=3.0):
        """Reinicia o broadcast (não necessário na nova implementação)"""
        return True  # Broadcast é contínuo na nova implementação
    
    # Métodos legacy mantidos para compatibilidade
    @staticmethod
    def get_all_ips():
        return UnityCommunicator.get_all_ips()
    
    @staticmethod
    def get_local_ip():
        all_ips = UnityCommunicator.get_all_ips()
        for ip in all_ips:
            if ip != '127.0.0.1':
                return ip
        return all_ips[0] if all_ips else '127.0.0.1'


class UDP_receiver:
    """Classe de compatibilidade para recepção"""
    
    _communicator = UnityCommunicator()
    
    @staticmethod
    def find_active_sender():
        """Encontra sender ativo - para compatibilidade"""
        return UDP_sender.get_all_ips()
    
    @staticmethod
    def listen_for_broadcast(timeout=10.0):
        """Para compatibilidade - retorna IPs locais"""
        return UDP_receiver.find_active_sender()
    
    @staticmethod
    def listen_for_broadcast_legacy():
        """Versão legacy que retorna apenas o primeiro IP"""
        ips = UDP_receiver.find_active_sender()
        return ips[0] if ips else None


# Função principal para demonstração
def main():
    """Função principal para teste do sistema"""
    communicator = UnityCommunicator()
    
    def on_message(message):
        print(f"Mensagem recebida: {message}")
    
    def on_connection(connected):
        print(f"Conexão: {'Conectado' if connected else 'Desconectado'}")
    
    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)
    
    # Iniciar servidor
    if not communicator.start_server():
        print("Falha ao iniciar servidor")
        return
    
    print("\n" + "="*50)
    print("Sistema de Comunicação Unity Ativo")
    print("="*50)
    print("Comandos disponíveis:")
    print("  - direita       : Controla mão direita") 
    print("  - esquerda      : Controla mão esquerda")
    print("  - trigger_right : Gatilho mão direita")
    print("  - trigger_left  : Gatilho mão esquerda")
    print("  - <comando>     : Comando personalizado")
    print("  - sair          : Encerra o programa")
    print("="*50)
    
    try:
        while True:
            comando = input("\nDigite um comando: ").strip()
            
            if comando.lower() == 'sair':
                break
            elif comando.lower() == 'direita':
                communicator.send_hand_command('direita')
            elif comando.lower() == 'esquerda':
                communicator.send_hand_command('esquerda')
            elif comando.lower() == 'trigger_right':
                communicator.send_trigger_command('direita')
            elif comando.lower() == 'trigger_left':
                communicator.send_trigger_command('esquerda')
            elif comando:
                communicator.send_command(comando)
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        communicator.stop_server()
        print("Programa encerrado")


if __name__ == '__main__':
    main()
