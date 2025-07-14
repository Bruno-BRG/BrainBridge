import time
import numpy as np
from collections import deque
import socket
import zmq
from keras.models import load_model
from pylsl import StreamInlet, resolve_stream
import keyboard

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

def predict_model(model, input_data):
    # Ajuste de dimensão conforme o modelo: nota o uso de transposição
    res = model(np.array([input_data.T]), training=False)
    return res.numpy()[0, 0]

def enviar_sinal(zmq_socket, resultado):
    if resultado == 1:
        zmq_socket.send_string("LEFT_HAND_OPEN")
        zmq_socket.send_string("RIGHT_HAND_CLOSE")
        print("Right hand closed")
    else:
        zmq_socket.send_string("RIGHT_HAND_OPEN")
        zmq_socket.send_string("LEFT_HAND_CLOSE")
        print("Left hand closed")

def main():
    # Carrega o modelo
    model = load_model(r'C:\Users\vivas\OneDrive\DocOneDrive\UFBA\CIMATEC\Projeto CC Tecnologias Assistivas\BCI EEG\CNN_Rafa\best_model_16.h5')
    epochsize = model.input_shape[2]

    # Inicializa a janela deslizante
    data_window = deque(maxlen=epochsize)

    print("Buscando stream EEG...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    time.sleep(1)

    # Configura o socket ZMQ (evite conflitar com o módulo socket)
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind("tcp://*:5555")

    # Envia IP local via broadcast
    local_ip = get_local_ip()
    print(f"IP Local detectado: {local_ip}")
    send_ip_UPD(local_ip)

    t0 = time.time()
    dt = 0.3
    tempo_a = None  # Para cálculo da frequência

    print("Iniciando aquisição de dados... (pressione 'f' para finalizar)")
    try:
        while not keyboard.is_pressed('f'):
            chunk, timestamps = inlet.pull_chunk()
            if chunk:
                for ind, sample in enumerate(chunk):
                    data_window.append(sample)

                    if keyboard.is_pressed('g'):
                        if tempo_a is not None and (timestamps[ind] - tempo_a) > 0:
                            freq = 1 / (timestamps[ind] - tempo_a)
                            print(f'Frequência: {freq:.2f} hz')
                    tempo_a = timestamps[ind]

                    if len(data_window) == epochsize and (time.time() - t0) > dt:
                        t0 = time.time()
                        resultado = predict_model(model, np.array(data_window))
                        enviar_sinal(zmq_socket, resultado)
    except KeyboardInterrupt:
        print("Encerramento solicitado pelo usuário.")
    finally:
        print("Encerrando aquisição de dados.")

if __name__ == '__main__':
    main()