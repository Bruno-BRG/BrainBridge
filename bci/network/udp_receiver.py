import zmq
import socket
import time

class UDP_receiver:
    @staticmethod
    def listen_for_broadcast():
        # Configurar socket UDP para receber broadcast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', 12347))  # Porta do broadcast
    
        print("Aguardando broadcast do IP...")
        data, addr = sock.recvfrom(1024)
        ip = data.decode()
        print(f"IP recebido: {ip}")
        sock.close()
        return ip

    @staticmethod
    def main():
        # Primeiro, escuta o broadcast para obter o IP
        sender_ip = UDP_receiver.listen_for_broadcast()

        # Configura o socket ZMQ para receber as mensagens
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{sender_ip}:5556")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Recebe todas as mensagens
    
        print("\nConectado! Aguardando comandos...")
        print("Pressione Ctrl+C para sair")
    
        try:
            while True:
                # Recebe as mensagens
                message = socket.recv_string()
                print(f"Comando recebido: {message}")
            
        except KeyboardInterrupt:
            print("\nEncerrando teste...")
        finally:
            socket.close()
            context.term()

if __name__ == "__main__":
    UDP_receiver.main()
