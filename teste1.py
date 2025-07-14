from conec_unity import configurar_socket
import keyboard
import time

if __name__ == '__main__':
    # Configura rede e ZMQ
    socket_pub, context = configurar_socket()
    print('Conexão estabelecida com sucesso.')

    print('Pressione 1, 2 ou 3 para enviar comandos. Pressione ESC para encerrar.')

    while not keyboard.is_pressed('Esc'):
        if keyboard.is_pressed('1'):
            socket_pub.send_string("LEFT_HAND_CLOSE")
            socket_pub.send_string("RIGHT_HAND_OPEN")
            print("Comando enviado: LEFT_HAND_CLOSE / RIGHT_HAND_OPEN")
            time.sleep(0.3)  # Evita múltiplos envios

        elif keyboard.is_pressed('2'):
            socket_pub.send_string("RIGHT_HAND_CLOSE")
            socket_pub.send_string("LEFT_HAND_OPEN")
            print("Comando enviado: RIGHT_HAND_CLOSE / LEFT_HAND_OPEN")
            time.sleep(0.3)

        elif keyboard.is_pressed('3'):
            socket_pub.send_string("LEFT_HAND_OPEN")
            socket_pub.send_string("RIGHT_HAND_OPEN")
            print("Comando enviado: LEFT_HAND_OPEN / RIGHT_HAND_OPEN")
            time.sleep(0.3)

    print("Encerrando conexão.")
    socket_pub.close()
    context.term()