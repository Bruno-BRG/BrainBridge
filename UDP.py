import time
import socket
import zmq

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

def enviar_sinal(zmq_socket, lado):
    if lado.lower() == 'direita':
        zmq_socket.send_string("RIGHT_HAND_CLOSE")
        time.sleep(1)
        zmq_socket.send_string("RIGHT_HAND_OPEN")
        print("Mão direita fechada")
    elif lado.lower() == 'esquerda':
        zmq_socket.send_string("LEFT_HAND_CLOSE")
        time.sleep(1)
        zmq_socket.send_string("LEFT_HAND_OPEN")
        print("Mão esquerda fechada")
    else:
        print("Entrada inválida. Use 'direita' ou 'esquerda'")

def main():
    # Configura o socket ZMQ
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind("tcp://*:5555")

    # Envia IP local via broadcast
    local_ip = get_local_ip()
    print(f"IP Local detectado: {local_ip}")
    send_ip_UPD(local_ip)

    print("\nInterface de Controle Manual")
    print("Digite 'sair' para encerrar o programa")
    
    while True:
        lado = input("\nQual mão deseja controlar? (direita/esquerda): ")
        if lado.lower() == 'sair':
            break
        enviar_sinal(zmq_socket, lado)

    print("Encerrando programa.")

if __name__ == '__main__':
    main()