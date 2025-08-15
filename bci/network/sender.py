# python_server.py

import socket
import threading
import time
import sys

# --- Configurações ---
UDP_PORT = 12346      # porta para broadcast de IPs
TCP_PORT = 12345      # porta para servidor TCP Unity
BROADCAST_INTERVAL = 1.0  # intervalo do broadcast em segundos
BUFFER_SIZE = 4096    # tamanho do buffer TCP


def get_all_ips():
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


def broadcast_ips(stop_event, udp_port=UDP_PORT):
    """
    Envia via broadcast UDP os IPs para que o Unity descubra o servidor.
    Para quando stop_event for sinalizado.
    """
    ips = get_all_ips()
    message = ','.join(ips).encode('utf-8')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[UDP] Iniciando broadcast em porta {udp_port}: {ips}")
    try:
        while not stop_event.is_set():
            sock.sendto(message, ('<broadcast>', udp_port))
            time.sleep(BROADCAST_INTERVAL)
    finally:
        sock.close()
        print("[UDP] Broadcast parado.")


def handle_unity_connection(conn, addr):
    """
    Lida com a conexão TCP com o Unity:
    - Inicia thread de recebimento e impressão de mensagens do Unity.
    - Faz loop interativo para enviar comandos ao Unity.
    """
    print(f"[TCP] Unity conectado de {addr}")

    # Thread para receber mensagens do Unity
    def recv_loop():
        try:
            while True:
                data = conn.recv(BUFFER_SIZE)
                if not data:
                    print("[TCP] Unity desconectou.")
                    break
                text = data.decode('utf-8', errors='ignore').strip()
                print(f"[TCP RECEIVED] {text}")
        except Exception as e:
            print(f"[TCP] Erro no recv: {e}")

    recv_thread = threading.Thread(target=recv_loop, daemon=True)
    recv_thread.start()

    # Loop interativo de envio
    try:
        print("Digite comandos para enviar ao Unity (ex: RIGHT_HAND_CLOSE). 'exit' para sair.")
        while True:
            cmd = input('> ').strip()
            if cmd.lower() in ('exit', 'quit', 'sair'):
                break
            if cmd:
                message = cmd + '\n'
                conn.sendall(message.encode('utf-8'))
                print(f"[TCP SENT] {cmd}")
    except KeyboardInterrupt:
        print("\n[TCP] Envio interrompido pelo usuário.")
    finally:
        conn.close()
        print("[TCP] Conexão encerrada.")


def tcp_server(stop_event, tcp_port=TCP_PORT):
    """
    Inicia servidor TCP para o Unity conectar.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', tcp_port))
    sock.listen(1)
    print(f"[TCP] Servidor ouvindo na porta {tcp_port}...")
    try:
        conn, addr = sock.accept()
        stop_event.set()  # para o broadcast
        handle_unity_connection(conn, addr)
    except Exception as e:
        print(f"[TCP] Erro no server: {e}")
    finally:
        sock.close()
        print("[TCP] Servidor parado.")


def main():
    # evento para parar broadcast quando Unity conectar
    stop_event = threading.Event()
    # inicia broadcast em thread
    udp_thread = threading.Thread(target=broadcast_ips, args=(stop_event,), daemon=True)
    udp_thread.start()
    # inicia servidor TCP (bloqueante)
    tcp_server(stop_event)

if __name__ == '__main__':
    main()