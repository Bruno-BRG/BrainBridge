import socket
import json
import threading
import time
from typing import Callable, Optional, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UDPReceiver_BCI:
    """
    Classe para receber dados via UDP no localhost na porta 12345
    """
    
    def __init__(self, host: str = 'localhost', port: int = 12345):
        """
        Inicializa o receptor UDP
        
        Args:
            host (str): Endereço do host (padrão: localhost)
            port (int): Porta para escutar (padrão: 12345)
        """
        self.host = host
        self.port = port
        self.socket = None
        self.is_running = False
        self.thread = None
        self.data_callback = None
        self.received_data = []
        self.lock = threading.Lock()
        
    def set_callback(self, callback: Callable[[Any], None]):
        """
        Define uma função de callback para processar os dados recebidos
        
        Args:
            callback: Função que será chamada quando dados forem recebidos
        """
        self.data_callback = callback
    
    def start(self):
        """
        Inicia o receptor UDP em uma thread separada
        """
        if self.is_running:
            logger.warning("Receptor UDP já está rodando")
            return
            
        try:
            # Criar socket UDP
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(1.0)  # Timeout para permitir parada suave
            
            self.is_running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Receptor UDP iniciado em {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar receptor UDP: {e}")
            self.is_running = False
            
    def stop(self):
        """
        Para o receptor UDP
        """
        if not self.is_running:
            return
            
        logger.info("Parando receptor UDP...")
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if self.socket:
            self.socket.close()
            self.socket = None
            
        logger.info("Receptor UDP parado")
        
    def _receive_loop(self):
        """
        Loop principal para receber dados UDP
        """
        logger.info("Loop de recepção UDP iniciado")
        
        while self.is_running:
            try:
                # Receber dados
                data, addr = self.socket.recvfrom(4096)  # Buffer de 4KB
                
                # Decodificar dados
                decoded_data = self._decode_data(data)
                
                if decoded_data is not None:
                    # Armazenar dados recebidos
                    with self.lock:
                        self.received_data.append({
                            'timestamp': time.time(),
                            'address': addr,
                            'data': decoded_data
                        })
                    
                    # Chamar callback se definido
                    if self.data_callback:
                        try:
                            self.data_callback(decoded_data)
                        except Exception as e:
                            logger.error(f"Erro no callback: {e}")
                            
                    logger.debug(f"Dados recebidos de {addr}: {decoded_data}")
                    
            except socket.timeout:
                # Timeout normal, continuar loop
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Erro ao receber dados UDP: {e}")
                    
    def _decode_data(self, data: bytes) -> Optional[Any]:
        """
        Decodifica os dados recebidos
        
        Args:
            data: Dados em bytes recebidos
            
        Returns:
            Dados decodificados ou None se houver erro
        """
        try:
            # Tentar decodificar como UTF-8
            text_data = data.decode('utf-8')
            
            # Tentar interpretar como JSON
            try:
                return json.loads(text_data)
            except json.JSONDecodeError:
                # Se não for JSON, retornar como string
                return text_data
                
        except UnicodeDecodeError:
            # Se não conseguir decodificar como UTF-8, retornar dados brutos
            logger.warning("Dados recebidos não são UTF-8 válido")
            return data.hex()
            
    def get_latest_data(self, count: int = 1) -> list:
        """
        Obtém os dados mais recentes recebidos
        
        Args:
            count: Número de registros mais recentes para retornar
            
        Returns:
            Lista com os dados mais recentes
        """
        with self.lock:
            return self.received_data[-count:] if self.received_data else []
            
    def get_all_data(self) -> list:
        """
        Obtém todos os dados recebidos
        
        Returns:
            Lista com todos os dados recebidos
        """
        with self.lock:
            return self.received_data.copy()
            
    def clear_data(self):
        """
        Limpa o buffer de dados recebidos
        """
        with self.lock:
            self.received_data.clear()
            logger.info("Buffer de dados limpo")
            
    def get_data_count(self) -> int:
        """
        Retorna o número de mensagens recebidas
        
        Returns:
            Número de mensagens no buffer
        """
        with self.lock:
            return len(self.received_data)


def example_callback(data):
    """
    Exemplo de função callback para processar dados recebidos
    """
    print(f"[CALLBACK] Dados recebidos: {data}")


if __name__ == "__main__":
    # Exemplo de uso
    receiver = UDPReceiver_BCI()
    
    # Definir callback para processar dados
    receiver.set_callback(example_callback)
    
    try:
        # Iniciar receptor
        receiver.start()
        
        print("Receptor UDP rodando...")
        print("Pressione Ctrl+C para parar")
        
        # Manter programa rodando
        while True:
            time.sleep(1)
            
            # Mostrar estatísticas a cada 10 segundos
            if int(time.time()) % 10 == 0:
                count = receiver.get_data_count()
                print(f"Total de mensagens recebidas: {count}")
                
    except KeyboardInterrupt:
        print("\nParando receptor...")
        receiver.stop()
        print("Receptor parado")