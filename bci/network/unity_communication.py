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
from enum import Enum

class MessageLevel(Enum):
    """Níveis de verbosidade das mensagens"""
    SILENT = 0      # Sem mensagens
    MINIMAL = 1     # Apenas conexões e erros críticos
    NORMAL = 2      # Mensagens importantes (padrão)
    VERBOSE = 3     # Todas as mensagens incluindo comandos
    DEBUG = 4       # Máximo detalhe incluindo timeouts

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
        
        # Configuração de verbosidade
        self.message_level = MessageLevel.NORMAL
        
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
    
    def _print(self, message: str, level: MessageLevel = MessageLevel.NORMAL):
        """Print condicional baseado no nível de verbosidade"""
        if self.message_level.value >= level.value:
            print(message)
    
    def set_message_level(self, level: MessageLevel):
        """Define o nível de verbosidade das mensagens"""
        self.message_level = level
        if level == MessageLevel.SILENT:
            self._print("🔇 Modo silencioso ativado", MessageLevel.MINIMAL)
        elif level == MessageLevel.MINIMAL:
            self._print("🔇 Modo minimalista ativado", MessageLevel.MINIMAL)
        elif level == MessageLevel.NORMAL:
            self._print("🔊 Modo normal ativado", MessageLevel.MINIMAL)
        elif level == MessageLevel.VERBOSE:
            self._print("📢 Modo verboso ativado", MessageLevel.MINIMAL)
        elif level == MessageLevel.DEBUG:
            self._print("🐛 Modo debug ativado", MessageLevel.MINIMAL)
    
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
            self._print("🔄 Servidor já está ativo", MessageLevel.NORMAL)
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
            self._print(f"🚀 Servidor Unity Comunicação INICIADO", MessageLevel.MINIMAL)
            self._print(f"   📡 ZMQ Publisher: porta {self.ZMQ_PORT}", MessageLevel.NORMAL)
            self._print(f"   🔗 TCP Server: porta {self.TCP_PORT}", MessageLevel.NORMAL)
            self._print(f"   📻 UDP Broadcast: porta {self.UDP_PORT}", MessageLevel.NORMAL)
            return True
            
        except Exception as e:
            self._print(f"❌ Erro ao iniciar servidor: {e}", MessageLevel.MINIMAL)
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
        
        self._print("🛑 Servidor Unity Comunicação PARADO e recursos limpos", MessageLevel.MINIMAL)
    
    def send_command(self, command: str) -> bool:
        """
        Envia comando para Unity via ZMQ e TCP
        Retorna True se enviado com sucesso
        """
        if not self.is_active:
            self._print("⚠️  Servidor não está ativo", MessageLevel.NORMAL)
            return False
            
        success = False
        
        # Enviar via ZMQ (sempre disponível quando servidor ativo)
        if self.zmq_socket:
            try:
                self.zmq_socket.send_string(command)
                self._print(f"📡 [ZMQ] ➤ {command}", MessageLevel.VERBOSE)
                success = True
            except Exception as e:
                self._print(f"❌ [ZMQ] Erro ao enviar: {e}", MessageLevel.MINIMAL)
        
        # Enviar via TCP se conectado
        if self.tcp_connected and self.tcp_connection:
            try:
                message = command + '\n'
                self.tcp_connection.sendall(message.encode('utf-8'))
                self._print(f"🔗 [TCP] ➤ {command}", MessageLevel.VERBOSE)
                success = True
            except Exception as e:
                self._print(f"❌ [TCP] Erro ao enviar: {e}", MessageLevel.MINIMAL)
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
            self._print(f"⚠️  Comando de mão inválido: {hand}", MessageLevel.NORMAL)
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
            self._print(f"⚠️  Comando de trigger inválido: {hand}", MessageLevel.NORMAL)
            return False
    
    def _broadcast_ips(self):
        """
        Thread para broadcast dos IPs via UDP
        """
        ips = self.get_all_ips()
        message = ','.join(ips).encode('utf-8')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        self._print(f"📻 [UDP] Broadcast iniciado: {', '.join(ips)}", MessageLevel.NORMAL)
        
        try:
            while not self.stop_event.is_set():
                sock.sendto(message, ('<broadcast>', self.UDP_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
        except Exception as e:
            self._print(f"❌ [UDP] Erro no broadcast: {e}", MessageLevel.MINIMAL)
        finally:
            sock.close()
            self._print("📻 [UDP] Broadcast parado", MessageLevel.NORMAL)
    
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
            self._print(f"🔗 [TCP] Servidor aguardando Unity na porta {self.TCP_PORT}...", MessageLevel.NORMAL)
            
            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    self._print(f"✅ [TCP] Unity CONECTADO! 🎮 ({addr[0]}:{addr[1]})", MessageLevel.MINIMAL)
                    
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
                    # Timeout é normal, apenas continua verificando stop_event
                    self._print("🔍 [TCP] Verificando conexões...", MessageLevel.DEBUG)
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        self._print(f"❌ [TCP] Erro no servidor: {e}", MessageLevel.MINIMAL)
                    break
                    
        except Exception as e:
            self._print(f"❌ [TCP] Erro ao iniciar servidor: {e}", MessageLevel.MINIMAL)
        finally:
            sock.close()
            self._print("🔗 [TCP] Servidor TCP parado", MessageLevel.NORMAL)
    
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
                        self._print("🔌 [TCP] Unity desconectou", MessageLevel.MINIMAL)
                        break
                        
                    message = data.decode('utf-8', errors='ignore').strip()
                    self._print(f"🎮 [TCP] ⬅ {message}", MessageLevel.VERBOSE)
                    
                    if self.on_message_received:
                        self.on_message_received(message)
                        
                except socket.timeout:
                    # Timeout é normal, apenas continua verificando stop_event
                    self._print("⏰ [TCP] Timeout de recepção", MessageLevel.DEBUG)
                    continue
                except Exception as e:
                    self._print(f"❌ [TCP] Erro na recepção: {e}", MessageLevel.MINIMAL)
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
            
            self._print("🔌 [TCP] Conexão Unity encerrada", MessageLevel.MINIMAL)
    
    def set_message_callback(self, callback: Callable[[str], None]):
        """Define callback para mensagens recebidas"""
        self.on_message_received = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Define callback para mudanças de conexão"""
        self.on_connection_changed = callback
    
    def get_connection_status(self) -> bool:
        """Retorna se o Unity está conectado via TCP"""
        return self.tcp_connected
    
    def get_server_status(self) -> bool:
        """Retorna se o servidor está ativo"""
        return self.is_active
    
    def get_message_level(self) -> MessageLevel:
        """Retorna o nível atual de verbosidade"""
        return self.message_level


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
        print(f"📬 Mensagem Unity: {message}")
    
    def on_connection(connected):
        status = "🎮 CONECTADO" if connected else "🔌 DESCONECTADO"
        print(f"🔗 Status Unity: {status}")
    
    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)
    
    # Iniciar servidor
    if not communicator.start_server():
        print("❌ Falha ao iniciar servidor")
        return
    
    print("\n" + "="*60)
    print("🎮 SISTEMA DE COMUNICAÇÃO UNITY ATIVO 🎮")
    print("="*60)
    print("🎯 Comandos disponíveis:")
    print("   🤚 direita       : Controla mão direita") 
    print("   ✋ esquerda      : Controla mão esquerda")
    print("   🎯 trigger_right : Gatilho mão direita")
    print("   🎯 trigger_left  : Gatilho mão esquerda")
    print("   ⚙️  <comando>     : Comando personalizado")
    print("   � verbosity <n> : Mudar verbosidade (0-4)")
    print("   �🚪 sair          : Encerra o programa")
    print("="*60)
    print("📊 Níveis de verbosidade:")
    print("   0: SILENT - Sem mensagens")
    print("   1: MINIMAL - Apenas conexões e erros")
    print("   2: NORMAL - Mensagens importantes (padrão)")
    print("   3: VERBOSE - Inclui comandos enviados/recebidos")
    print("   4: DEBUG - Máximo detalhe")
    print("="*60)
    
    try:
        while True:
            comando = input("\n💭 Digite um comando: ").strip()
            
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
            elif comando.lower().startswith('verbosity '):
                try:
                    level_num = int(comando.split()[1])
                    if 0 <= level_num <= 4:
                        level = MessageLevel(level_num)
                        communicator.set_message_level(level)
                    else:
                        print("⚠️ Nível deve ser entre 0 e 4")
                except (ValueError, IndexError):
                    print("⚠️ Uso: verbosity <0-4>")
            elif comando:
                communicator.send_command(comando)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    finally:
        communicator.stop_server()
        print("👋 Programa encerrado")


if __name__ == '__main__':
    main()
