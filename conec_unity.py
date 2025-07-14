import zmq

def configurar_socket():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")  # ou connect, dependendo da arquitetura
    return socket, context