import csv
import os
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from .udp_receiver_BCI import UDPReceiver_BCI

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataLogger:
    """
    Classe para capturar dados UDP e salvar em arquivo CSV
    """
    
    def __init__(self, csv_filename: Optional[str] = None, host: str = 'localhost', port: int = 12345):
        """
        Inicializa o logger CSV
        
        Args:
            csv_filename (str): Nome do arquivo CSV (se None, gera automaticamente)
            host (str): Host para receber dados UDP
            port (int): Porta para receber dados UDP
        """
        self.host = host
        self.port = port
        
        # Gerar nome do arquivo se não fornecido
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_filename = f"udp_data_{timestamp}.csv"
        else:
            self.csv_filename = csv_filename
            
        self.csv_filepath = os.path.join(os.getcwd(), self.csv_filename)
        
        # Receptor UDP
        self.udp_receiver = UDPReceiver_BCI(host=host, port=port)
        
        # Configurações para tempo real
        self.is_logging = False
        self.data_buffer = []
        self.buffer_size = 50   # Salvar a cada 50 registros (mais frequente)
        self.auto_flush_interval = 2  # Salvar a cada 2 segundos (mais frequente)
        self.last_flush_time = time.time()
        
        # Lock para thread safety em tempo real
        self.buffer_lock = threading.Lock()
        
        # Headers do CSV
        self.csv_headers = ['timestamp', 'source_ip', 'source_port', 'data_type', 'raw_data']
        self.headers_written = False
        
    def _process_udp_data(self, data: Any):
        """
        Processa dados recebidos via UDP e adiciona ao buffer
        
        Args:
            data: Dados recebidos do UDP
        """
        if not self.is_logging:
            return
            
        try:
            # Obter último dado recebido com informações completas
            latest_data = self.udp_receiver.get_latest_data(1)
            if latest_data:
                record = latest_data[0]
                
                # Preparar linha para CSV
                csv_row = {
                    'timestamp': datetime.fromtimestamp(record['timestamp']).isoformat(),
                    'source_ip': record['address'][0],
                    'source_port': record['address'][1],
                    'data_type': type(record['data']).__name__,
                    'raw_data': str(record['data'])
                }
                
                # Adicionar ao buffer com thread safety
                with self.buffer_lock:
                    self.data_buffer.append(csv_row)
                
                logger.debug(f"Dados adicionados ao buffer: {csv_row}")
                
                # Verificar se precisa salvar (não bloqueia a captura)
                self._check_flush_conditions()
                
        except Exception as e:
            logger.error(f"Erro ao processar dados UDP: {e}")
            
    def _check_flush_conditions(self):
        """
        Verifica se deve salvar o buffer no arquivo CSV (não bloqueia)
        """
        current_time = time.time()
        
        # Verificar condições sem lock prolongado
        should_flush = False
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
            
        # Salvar se buffer atingiu tamanho máximo ou tempo limite
        if (buffer_size >= self.buffer_size or 
            current_time - self.last_flush_time >= self.auto_flush_interval):
            # Flush em thread separada para não bloquear captura
            threading.Thread(target=self._flush_to_csv, daemon=True).start()
            
    def _flush_to_csv(self):
        """
        Salva o buffer no arquivo CSV (thread-safe)
        """
        # Copiar e limpar buffer atomicamente
        with self.buffer_lock:
            if not self.data_buffer:
                return
            data_to_save = self.data_buffer.copy()
            self.data_buffer.clear()
            
        try:
            # Verificar se arquivo existe e se headers foram escritos
            file_exists = os.path.exists(self.csv_filepath)
            
            with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                
                # Escrever headers se necessário
                if not file_exists or not self.headers_written:
                    writer.writeheader()
                    self.headers_written = True
                    
                # Escrever dados
                writer.writerows(data_to_save)
                
            logger.info(f"✅ Salvos {len(data_to_save)} registros no arquivo {self.csv_filename}")
            
            # Atualizar timestamp
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar no CSV: {e}")
            # Em caso de erro, recolocar dados no buffer
            with self.buffer_lock:
                self.data_buffer = data_to_save + self.data_buffer
            
    def start_logging(self):
        """
        Inicia a captura e logging de dados
        """
        if self.is_logging:
            logger.warning("Logging já está ativo")
            return
            
        try:
            # Configurar callback no receptor UDP
            self.udp_receiver.set_callback(self._process_udp_data)
            
            # Iniciar receptor UDP
            self.udp_receiver.start()
            
            self.is_logging = True
            logger.info(f"Logging iniciado. Dados serão salvos em: {self.csv_filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar logging: {e}")
            self.is_logging = False
            
    def stop_logging(self):
        """
        Para a captura e logging de dados
        """
        if not self.is_logging:
            return
            
        logger.info("Parando logging...")
        self.is_logging = False
        
        # Parar receptor UDP
        self.udp_receiver.stop()
        
        # Salvar dados restantes no buffer
        if self.data_buffer:
            # Flush final em thread principal para garantir que termine
            with self.buffer_lock:
                data_to_save = self.data_buffer.copy()
                self.data_buffer.clear()
                
            if data_to_save:
                try:
                    file_exists = os.path.exists(self.csv_filepath)
                    with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                        if not file_exists or not self.headers_written:
                            writer.writeheader()
                        writer.writerows(data_to_save)
                    logger.info(f"✅ Flush final: {len(data_to_save)} registros salvos")
                except Exception as e:
                    logger.error(f"❌ Erro no flush final: {e}")
            
        logger.info("Logging parado")
        
    def force_save(self):
        """
        Força a salvamento imediato dos dados no buffer
        """
        if self.data_buffer:
            self._flush_to_csv()
            logger.info("Salvamento forçado concluído")
        else:
            logger.info("Nenhum dado no buffer para salvar")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do logging (thread-safe)
        
        Returns:
            Dicionário com estatísticas
        """
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
            
        return {
            'is_logging': self.is_logging,
            'csv_file': self.csv_filepath,
            'buffer_size': buffer_size,
            'total_received': self.udp_receiver.get_data_count(),
            'file_exists': os.path.exists(self.csv_filepath),
            'file_size_bytes': os.path.getsize(self.csv_filepath) if os.path.exists(self.csv_filepath) else 0,
            'last_save': datetime.fromtimestamp(self.last_flush_time).strftime("%H:%M:%S")
        }
        
    def clear_csv_file(self):
        """
        Remove o arquivo CSV atual
        """
        if os.path.exists(self.csv_filepath):
            os.remove(self.csv_filepath)
            self.headers_written = False
            logger.info(f"Arquivo CSV removido: {self.csv_filepath}")
        else:
            logger.info("Arquivo CSV não existe")
            
    def set_buffer_size(self, size: int):
        """
        Define o tamanho do buffer antes de salvar
        
        Args:
            size: Novo tamanho do buffer
        """
        self.buffer_size = max(1, size)
        logger.info(f"Tamanho do buffer definido para: {self.buffer_size}")
        
    def set_auto_flush_interval(self, seconds: int):
        """
        Define intervalo de salvamento automático
        
        Args:
            seconds: Intervalo em segundos
        """
        self.auto_flush_interval = max(1, seconds)
        logger.info(f"Intervalo de salvamento definido para: {self.auto_flush_interval} segundos")


def main():
    """
    Exemplo de uso do CSVDataLogger
    """
    # Criar logger
    logger_csv = CSVDataLogger()
    
    # Configurar parâmetros
    logger_csv.set_buffer_size(50)  # Salvar a cada 50 registros
    logger_csv.set_auto_flush_interval(5)  # Ou a cada 5 segundos
    
    try:
        # Iniciar logging
        logger_csv.start_logging()
        
        print("CSV Data Logger rodando...")
        print(f"Arquivo CSV: {logger_csv.csv_filename}")
        print("Pressione Ctrl+C para parar")
        
        # Loop principal
        while True:
            time.sleep(1)
            
            # Mostrar estatísticas a cada 10 segundos
            if int(time.time()) % 10 == 0:
                stats = logger_csv.get_stats()
                print(f"\n--- Estatísticas ---")
                print(f"Total recebido: {stats['total_received']}")
                print(f"Buffer atual: {stats['buffer_size']}")
                print(f"Tamanho arquivo: {stats['file_size_bytes']} bytes")
                print("-------------------\n")
                
    except KeyboardInterrupt:
        print("\nParando logger...")
        logger_csv.stop_logging()
        
        # Mostrar estatísticas finais
        stats = logger_csv.get_stats()
        print(f"\n--- Estatísticas Finais ---")
        print(f"Total recebido: {stats['total_received']}")
        print(f"Arquivo salvo: {stats['csv_file']}")
        print(f"Tamanho final: {stats['file_size_bytes']} bytes")
        print("---------------------------")


if __name__ == "__main__":
    main()