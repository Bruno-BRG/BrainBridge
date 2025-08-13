import time
import socket
import zmq
import threading

class UDP:
    # Class variables
    zmq_socket = None
    context = None

    @staticmethod
    def get_local_ip():
        """Mantém compatibilidade com código existente - pega o primeiro IP válido"""
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
    def find_tcp_connection():
        """
        Procura por uma conexão TCP ativa na porta 5555 (estabelecida pelo UDP_sender.py principal).
        Retorna o IP local se encontrar conexão ativa.
        """
        try:
            # Tenta conectar na porta TCP do sistema principal
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(2.0)
            local_ip = UDP.get_local_ip()

            # Testa se há servidor TCP rodando na porta 5555
            result = test_sock.connect_ex((local_ip, 5555))
            test_sock.close()
            
            if result == 0:
                print(f"[TEST] Encontrada conexão TCP ativa em {local_ip}:5555")
                return local_ip
            else:
                print(f"[TEST] Nenhuma conexão TCP ativa encontrada em {local_ip}:5555")
                return None
                
        except Exception as e:
            print(f"[TEST] Erro ao verificar conexão TCP: {e}")
            return None

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
            elif action.lower() == 'redleft':
                import threading
                def send_redleft():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("RED_FLOWER,CORRECT")
                threading.Thread(target=send_redleft, daemon=True).start()
                print("Sinal de flor vermelha mão esquerda enviado")
                return True
            elif action.lower() == 'redright':
                import threading
                def send_redright():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("RED_FLOWER, WRONG")
                threading.Thread(target=send_redright, daemon=True).start()
                print("Sinal de flor vermelha mão direita enviado")
                return True
            elif action.lower() == 'blueleft':
                import threading
                def send_blueleft():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("BLUE_FLOWER, WRONG")
                threading.Thread(target=send_blueleft, daemon=True).start()
                print("Sinal de flor azul mão esquerda enviado")
                return True
            elif action.lower() == 'blueright':
                import threading
                def send_blueright():
                    time.sleep(0.1)
                    if cls.zmq_socket is not None:
                        cls.zmq_socket.send_string("BLUE_FLOWER,CORRECT")
                threading.Thread(target=send_blueright, daemon=True).start()
                print("Sinal de flor azul mão direita enviado")
                return True
            else:
                print("Entrada inválida.")
                return False
        except Exception as e:
            print(f"Erro ao enviar sinal: {e}")
            return False

    @classmethod
    def init_zmq_socket(cls):
        """Inicializa o socket ZMQ na porta 5556 (porta dedicada para test_sender)"""
        if cls.zmq_socket is not None:
            print("[TEST] Socket ZMQ já está inicializado")
            return True
            
        # Verifica se há conexão TCP ativa (estabelecida pelo sistema principal)
        tcp_ip = cls.find_tcp_connection()
        if not tcp_ip:
            print("[TEST] ❌ ERRO: Nenhuma conexão TCP encontrada!")
            print("[TEST] Execute primeiro o UDP_sender.py principal para estabelecer a conexão TCP.")
            return False
            
        try:
            # Inicializa contexto ZMQ na porta 5556 (dedicada para test_sender)
            cls.context = zmq.Context()
            cls.zmq_socket = cls.context.socket(zmq.PUB)
            cls.zmq_socket.bind("tcp://*:5556")
            print("[TEST] Socket ZMQ inicializado na porta 5556 (porta dedicada test_sender)")
            print(f"[TEST] ✅ Aproveitando conexão TCP estabelecida em {tcp_ip}:12345")
            print("[TEST] Receivers podem escutar nas portas 5555 (UDP_sender) e 5556 (test_sender)")
            
            return True
                
        except Exception as e:
            print(f"[TEST] Erro ao inicializar socket ZMQ na porta 5556: {e}")
            cls.zmq_socket = None
            cls.context = None
            return False

    @classmethod
    def stop_zmq_socket(cls):
        """Para o socket ZMQ e limpa recursos"""
        # Fecha socket ZMQ
        if cls.zmq_socket is not None:
            try:
                cls.zmq_socket.close()
                print("[TEST] Socket ZMQ fechado")
            except Exception as e:
                print(f"[TEST] Erro ao fechar socket ZMQ: {e}")
            finally:
                cls.zmq_socket = None
                
        # Termina contexto ZMQ
        if cls.context is not None:
            try:
                cls.context.term()
            except Exception as e:
                print(f"[TEST] Erro ao terminar contexto ZMQ: {e}")
            finally:
                cls.context = None
                
        print("[TEST] Recursos limpos")

    @classmethod
    def is_server_active(cls):
        """Verifica se o servidor ZMQ está ativo"""
        return cls.zmq_socket is not None

    @classmethod
    def main(cls):
        print("\n" + "="*60)
        print("TEST SENDER - Aproveitando Conexão TCP Existente")
        print("="*60)
        print("[TEST] Este teste requer que o UDP_sender.py principal esteja rodando")
        print("[TEST] para aproveitar a conexão TCP já estabelecida.")
        print("="*60)
        
        # Inicializa o socket ZMQ (verifica conexão TCP existente)
        if not cls.init_zmq_socket():
            print("\n[TEST] ❌ Falha na inicialização!")
            print("[TEST] Certifique-se de que o UDP_sender.py principal está rodando.")
            return

        print("\n" + "="*60)
        print("Interface de Controle Manual - TEST SENDER")
        print("="*60)
        print("Comandos disponíveis:")
        print("  - direita       : Controla mão direita")
        print("  - esquerda      : Controla mão esquerda") 
        print("  - trigger_right : Gatilho mão direita")
        print("  - trigger_left  : Gatilho mão esquerda")
        print("  - redleft       : Flor vermelha mão esquerda (CORRECT)")
        print("  - redright      : Flor vermelha mão direita (WRONG)")
        print("  - blueleft      : Flor azul mão esquerda (WRONG)")
        print("  - blueright     : Flor azul mão direita (CORRECT)")
        print("  - sair          : Encerra o programa")
        print("="*60)
        
        try:
            while True:
                comando = input("\n[TEST] Digite um comando: ").strip().lower()
                
                if comando == 'sair':
                    break
                elif comando in ['direita', 'esquerda', 'trigger_right', 'trigger_left', 
                               'redleft', 'redright', 'blueleft', 'blueright']:
                    cls.enviar_sinal(comando)
                elif comando == '':
                    continue
                else:
                    print("[TEST] Comando inválido! Comandos disponíveis:")
                    print("direita, esquerda, trigger_right, trigger_left,")
                    print("redleft, redright, blueleft, blueright, sair")
                    
        except KeyboardInterrupt:
            print("\n[TEST] Interrompido pelo usuário")
        finally:
            cls.stop_zmq_socket()
            print("[TEST] Programa encerrado.")

if __name__ == '__main__':
    UDP.main()