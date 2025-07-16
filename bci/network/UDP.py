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
    def enviar_sinal(cls, lado):
        if cls.zmq_socket is None:
            print("Erro: Socket ZMQ não foi inicializado. Chame init_zmq_socket() primeiro.")
            return
            
        if lado.lower() == 'direita':
            cls.zmq_socket.send_string("RIGHT_HAND_CLOSE")
            time.sleep(1)
            cls.zmq_socket.send_string("RIGHT_HAND_OPEN")
            print("Mão direita fechada")
        elif lado.lower() == 'esquerda':
            cls.zmq_socket.send_string("LEFT_HAND_CLOSE")
            time.sleep(1)
            cls.zmq_socket.send_string("LEFT_HAND_OPEN")
            print("Mão esquerda fechada")
        else:
            print("Entrada inválida. Use 'direita' ou 'esquerda'")

    @classmethod
    def init_zmq_socket(cls):
        """Inicializa o socket ZMQ global"""
        context = zmq.Context()
        cls.zmq_socket = context.socket(zmq.PUB)
        cls.zmq_socket.bind("tcp://*:5555")
        print("Socket ZMQ inicializado na porta 5555")

    @classmethod
    def main(cls):
        # Inicializa o socket ZMQ
        cls.init_zmq_socket()

        # Envia IP local via broadcast
        local_ip = cls.get_local_ip()
        print(f"IP Local detectado: {local_ip}")
        cls.send_ip_UPD(local_ip)

        print("\nInterface de Controle Manual")
        print("Digite 'sair' para encerrar o programa")
        
        while True:
            lado = input("\nQual mão deseja controlar? (direita/esquerda): ")
            if lado.lower() == 'sair':
                break
            cls.enviar_sinal(lado)

        print("Encerrando programa.")

if __name__ == '__main__':
    UDP.main()