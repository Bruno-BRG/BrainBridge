import time
import socket
import zmq

class UDP:
    # Class variable for ZMQ socket
    zmq_socket = None

    @staticmethod
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print(f"Erro ao obter IP local: {e}")
            return "127.0.0.1"

    @staticmethod
    def send_ip_UPD(ip):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        broadcast_port = 12346
        try:
            sock.sendto(ip.encode(), ('<broadcast>', broadcast_port))
            print(f"Broadcast enviado: {ip}")
        except Exception as e:
            print(f"Erro ao enviar broadcast: {e}")
        finally:
            sock.close()

    @classmethod
    def enviar_sinal(cls, action):
        if cls.zmq_socket is None:
            print("Erro: Socket ZMQ não foi inicializado. Chame init_zmq_socket() primeiro.")
            return False
            
        try:
            if action.lower() == 'direita':
                cls.zmq_socket.send_string("RIGHT_HAND_CLOSE")
                # Enviar comando de abrir após um pequeno delay (sem bloquear)
                import threading
                def send_open():
                    time.sleep(0.1)  # Delay menor para não bloquear interface
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("RIGHT_HAND_OPEN")
                threading.Thread(target=send_open, daemon=True).start()
                print("Sinal mão direita enviado")
                return True
            elif action.lower() == 'esquerda':
                cls.zmq_socket.send_string("LEFT_HAND_CLOSE")
                # Enviar comando de abrir após um pequeno delay (sem bloquear)
                import threading
                def send_open():
                    time.sleep(0.1)  # Delay menor para não bloquear interface
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("LEFT_HAND_OPEN")
                threading.Thread(target=send_open, daemon=True).start()
                print("Sinal mão esquerda enviado")
                return True
            elif action.lower() == 'trigger_right':
                import threading
                def send_trigger():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("TRIGGER_RIGHT")
                threading.Thread(target=send_trigger, daemon=True).start()
                print("Sinal de gatilho mão direita enviado")
                return True
            elif action.lower() == 'trigger_left':
                import threading
                def send_trigger():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("TRIGGER_LEFT")
                threading.Thread(target=send_trigger, daemon=True).start()
                print("Sinal de gatilho mão esquerda enviado")
                return True
            else:
                print("Entrada inválida.")
                return False
        except Exception as e:
            print(f"Erro ao enviar sinal: {e}")
            return False

    @classmethod
    def init_zmq_socket(cls):
        """Inicializa o socket ZMQ global e envia broadcast do IP"""
        if cls.zmq_socket is not None:
            print("Socket ZMQ já está inicializado")
            return
            
        try:
            context = zmq.Context()
            cls.zmq_socket = context.socket(zmq.PUB)
            cls.zmq_socket.bind("tcp://*:5555")
            print("Socket ZMQ inicializado na porta 5555")
            
            # Enviar broadcast do IP automaticamente
            local_ip = cls.get_local_ip()
            print(f"IP Local detectado: {local_ip}")
            cls.send_ip_UPD(local_ip)
            
            # Aguardar um pouco para garantir que o broadcast foi enviado
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Erro ao inicializar socket ZMQ: {e}")
            cls.zmq_socket = None
            raise

    @classmethod
    def stop_zmq_socket(cls):
        """Para o socket ZMQ e limpa recursos"""
        if cls.zmq_socket is not None:
            try:
                cls.zmq_socket.close()
                print("Socket ZMQ fechado")
            except Exception as e:
                print(f"Erro ao fechar socket ZMQ: {e}")
            finally:
                cls.zmq_socket = None
        else:
            print("Socket ZMQ já estava parado")

    @classmethod
    def is_server_active(cls):
        """Verifica se o servidor ZMQ está ativo"""
        return cls.zmq_socket is not None

    @classmethod
    def main(cls):
        # Inicializa o socket ZMQ (já inclui o broadcast)
        cls.init_zmq_socket()

        print("\nInterface de Controle Manual")
        print("Digite 'sair' para encerrar o programa")
        
        try:
            while True:
                lado = input("\nQual mão deseja controlar? (direita/esquerda): ")
                if lado.lower() == 'sair':
                    break
                cls.enviar_sinal(lado)
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        finally:
            cls.stop_zmq_socket()
            print("Programa encerrado.")

if __name__ == '__main__':
    UDP.main()