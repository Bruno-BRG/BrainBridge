"""
Módulo de comunicação serial com ESP32
Envia comandos TRIGGER para ESP32 via porta serial COM3
"""

import serial
import serial.tools.list_ports
import threading
import time
from typing import Optional, Callable
import logging

class ESP32SerialCommunicator:
    """
    Comunicacao serial com ESP32 da ortese (CIMATEC).

    Protocolo compativel com Ortese/ortese.cpp (1 char, 115200 baud):
    - 'l' = flexao sob demanda (IA esquerda), 'e' = extensao (IA direita)
    - 'o' = parar, 'm' = entrar em modo IA, 'z' = ciclo automatico
    - 'p' = pausar/retomar, 'r' = reset/calibracao, 'w/s/d/a' = calibracao
    - '\\n/\\r/espaco' sao ignorados pelo firmware.

    Os metodos publicos mantem a API antiga (send_trigger_left/right,
    send_direction), mas os bytes enviados agora sao os chars acima —
    nunca mais "LEFT"/"RIGHT"/"PING" (que continham 'r'/'p' e resetavam
    ou pausavam a ortese sem querer).
    """

    # Comandos de movimento IA (inputs do firmware; outputs/motores intactos).
    CMD_FLEXAO = "l"
    CMD_EXTENSAO = "e"
    CMD_PARAR = "o"
    CMD_MODO_IA = "m"

    def __init__(self, port: str = "COM4", baudrate: int = 115200, timeout: float = 1.0):
        """
        Inicializa o comunicador serial
        
        Args:
            port: Porta serial (padrão COM4)
            baudrate: Taxa de transmissão (padrão 115200)
            timeout: Timeout para comunicação (padrão 1.0s)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Estado da conexão
        self.is_connected = False
        self.serial_connection: Optional[serial.Serial] = None
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        # Callback para mudanças de conexão
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Controle de duração do trigger (3 segundos)
        self.trigger_duration = 3.0  # segundos
        self.last_trigger_time = 0.0
        self.trigger_active = False
    
    @staticmethod
    def list_available_ports() -> list:
        """
        Lista todas as portas seriais disponíveis
        
        Returns:
            list: Lista de tuplas (porta, descrição)
        """
        ports = []
        try:
            for port in serial.tools.list_ports.comports():
                ports.append((port.device, port.description))
        except Exception as e:
            logging.error(f"Erro ao listar portas: {e}")
        return ports
    
    @staticmethod
    def port_exists(port: str) -> bool:
        """
        Verifica se uma porta serial existe

        Args:
            port: Nome da porta (ex: COM4)

        Returns:
            bool: True se porta existe
        """
        available_ports = [p[0] for p in ESP32SerialCommunicator.list_available_ports()]
        return port in available_ports

    @staticmethod
    def autodetect_port(preferred: Optional[str] = None) -> Optional[str]:
        """Retorna a porta preferida se existir, senao a 1a USB/serial livre."""
        try:
            available = [p[0] for p in ESP32SerialCommunicator.list_available_ports()]
        except Exception:
            return preferred
        if preferred and preferred in available:
            return preferred
        for cand in (preferred or "", "COM4", "COM3", "/dev/ttyUSB0", "/dev/ttyACM0"):
            if cand and cand in available:
                return cand
        return available[0] if available else preferred
    
    def connect(self) -> bool:
        """
        Conecta à porta serial
        
        Returns:
            bool: True se conectado com sucesso
        """
        with self._lock:
            if self.is_connected:
                self.logger.info("ESP32 já conectado")
                print(f"[ESP32] ℹ Já está conectado em {self.port}", flush=True)
                return True
            
            # Verificar se porta existe antes de tentar conectar (com fallback).
            if not self.port_exists(self.port):
                fallback = self.autodetect_port(self.port)
                if fallback and fallback != self.port and self.port_exists(fallback):
                    print(f"[ESP32] Porta {self.port} ausente; usando {fallback}", flush=True)
                    self.port = fallback
            if not self.port_exists(self.port):
                available_ports = self.list_available_ports()
                self.logger.error(f"Porta {self.port} não encontrada!")
                msg = f"[ESP32] ✗ Porta {self.port} não existe!"
                if available_ports:
                    msg += f"\nPortas disponíveis: {', '.join([p[0] for p in available_ports])}"
                    self.logger.error(f"Portas disponíveis: {available_ports}")
                print(msg, flush=True)
                return False
            
            try:
                print(f"[ESP32] Conectando em {self.port} @ {self.baudrate} baud...", flush=True)
                self.serial_connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                # Aguardar um pouco para estabilizar a conexão
                time.sleep(0.5)
                
                # Teste de comunicação
                if self.serial_connection.is_open:
                    self.is_connected = True
                    msg = f"[ESP32] ✓ Conectado em {self.port} @ {self.baudrate}"
                    self.logger.info(msg)
                    print(msg, flush=True)

                    # Sem PING: o firmware interpreta 'p' como pausar e 'r'
                    # (de RIGHT/PING) como reset. Conexao e silenciosa; o
                    # menu do firmware ja confirma via leitura serial.
                    self.logger.debug("Conexao silenciosa (sem PING)")

                    # Notificar mudança de conexão
                    if self.on_connection_changed:
                        self.on_connection_changed(True)

                    return True
                else:
                    self.logger.error("Falha ao abrir porta serial")
                    print("[ESP32] ✗ Falha ao abrir porta serial", flush=True)
                    return False
                    
            except serial.SerialException as e:
                msg = f"[ESP32] ✗ Erro ao conectar: {e}"
                self.logger.error(msg)
                print(msg, flush=True)
                self.serial_connection = None
                return False
            except Exception as e:
                msg = f"[ESP32] ✗ Erro inesperado: {e}"
                self.logger.error(msg)
                print(msg, flush=True)
                self.serial_connection = None
                return False
    
    def disconnect(self):
        """
        Desconecta da porta serial
        """
        with self._lock:
            if self.serial_connection and self.serial_connection.is_open:
                try:
                    self.serial_connection.close()
                    self.logger.info("ESP32 desconectado")
                except Exception as e:
                    self.logger.error(f"Erro ao desconectar ESP32: {e}")
                finally:
                    self.serial_connection = None
                    self.is_connected = False
                    
                    # Notificar mudança de conexão
                    if self.on_connection_changed:
                        self.on_connection_changed(False)
    
    def _send_raw_command_unlocked(self, command: str) -> bool:
        """
        Envia comando bruto para ESP32 (SEM lock - usar dentro de _send_raw_command)
        
        Args:
            command: Comando a ser enviado
            
        Returns:
            bool: True se enviado com sucesso
        """
        if not self.is_connected or not self.serial_connection:
            self.logger.warning(f"[ESP32] Não conectado - comando '{command}' ignorado")
            print(f"[ESP32] ✗ Não conectado para enviar: {command}", flush=True)
            return False
        
        try:
            # Adicionar quebra de linha se não houver
            if not command.endswith('\n'):
                command += '\n'
            
            # Enviar comando
            bytes_written = self.serial_connection.write(command.encode('utf-8'))
            self.serial_connection.flush()
            
            self.logger.info(f"[ESP32] Enviado: {command.strip()} ({bytes_written} bytes)")
            print(f"[ESP32] ✓ Comando enviado: {command.strip()}", flush=True)
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"[ESP32] Erro serial: {e}")
            self.is_connected = False
            print(f"[ESP32] ✗ Erro serial: {e}", flush=True)
            return False
        except Exception as e:
            self.logger.error(f"[ESP32] Erro inesperado: {e}")
            print(f"[ESP32] ✗ Erro: {e}", flush=True)
            return False
    
    def _send_raw_command(self, command: str) -> bool:
        """
        Envia comando bruto para ESP32 (COM lock)
        
        Args:
            command: Comando a ser enviado
            
        Returns:
            bool: True se enviado com sucesso
        """
        with self._lock:
            return self._send_raw_command_unlocked(command)
    
    def send_trigger_command(self, hand: str) -> bool:
        """
        Envia comando de trigger para ESP32 (protocolo 1-char do firmware).
        Mantém o trigger ativo por 3 segundos antes de liberar para o próximo
        (casa com TEMPO_MOVIMENTO 2s + TEMPO_PAUSA 1s da ortese).

        Args:
            hand: 'direita'/'right' ou 'esquerda'/'left'

        Returns:
            bool: True se enviado com sucesso, False se trigger ainda ativo
        """
        with self._lock:
            # Verificar se ainda há trigger ativo
            if self.trigger_active:
                tempo_restante = self.trigger_duration - (time.time() - self.last_trigger_time)
                if tempo_restante > 0:
                    msg = f"[ESP32] ⏳ Trigger bloqueado por {tempo_restante:.1f}s"
                    self.logger.warning(msg)
                    print(msg, flush=True)
                    return False
                else:
                    # Tempo expirou, liberar novo trigger
                    self.trigger_active = False

        # Mapeia para os inputs do firmware (nunca "LEFT"/"RIGHT").
        # Esquerda = flexao ('l'), direita = extensao ('e').
        normalized = (hand or "").strip().lower()
        if normalized in ['direita', 'right', 'e', 'extensao', 'extensão']:
            cmd = self.CMD_EXTENSAO
        elif normalized in ['esquerda', 'left', 'l', 'flexao', 'flexão']:
            cmd = self.CMD_FLEXAO
        else:
            self.logger.error(f"Comando de trigger inválido: {hand}")
            return False
        success = self._send_raw_command_unlocked(cmd)
        
        if success:
            # Registrar tempo do trigger
            self.last_trigger_time = time.time()
            self.trigger_active = True
            msg = f"[ESP32] ▶️  Trigger iniciado - bloqueado por {self.trigger_duration}s"
            self.logger.info(msg)
            print(msg, flush=True)
        
        return success
    
    def send_trigger_left(self) -> bool:
        """
        Envia trigger para mão esquerda
        
        Returns:
            bool: True se enviado com sucesso
        """
        return self.send_trigger_command('esquerda')
    
    def send_trigger_right(self) -> bool:
        """
        Envia trigger para mão direita
        
        Returns:
            bool: True se enviado com sucesso
        """
        return self.send_trigger_command('direita')
    
    def send_ping(self) -> bool:
        """
        Health-check silencioso: nao envia nada que o firmware interprete
        ('p' pausaria). Retorna apenas o estado da conexao.
        """
        return bool(self.is_connected and self.serial_connection is not None)

    def send_ia_mode(self) -> bool:
        """Coloca a ortese em modo IA (aguarda 'l'/'e' sob demanda)."""
        return self._send_raw_command(self.CMD_MODO_IA)

    def send_stop(self) -> bool:
        """Para os motores ('o')."""
        return self._send_raw_command(self.CMD_PARAR)

    def send_pause_toggle(self) -> bool:
        """Alterna pausa ('p')."""
        return self._send_raw_command("p")

    def send_reset(self) -> bool:
        """Volta para calibracao ('r')."""
        return self._send_raw_command("r")
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """
        Define callback para mudanças de conexão
        
        Args:
            callback: Função a ser chamada quando conexão muda
        """
        self.on_connection_changed = callback
    
    def get_connection_status(self) -> dict:
        """
        Retorna status da conexão
        
        Returns:
            dict: Informações sobre a conexão
        """
        with self._lock:
            tempo_restante = 0.0
            if self.trigger_active:
                tempo_restante = max(0.0, self.trigger_duration - (time.time() - self.last_trigger_time))
            
            return {
                'connected': self.is_connected,
                'port': self.port,
                'baudrate': self.baudrate,
                'timeout': self.timeout,
                'trigger_active': self.trigger_active,
                'trigger_remaining_time': tempo_restante
            }
    
    def is_trigger_ready(self) -> bool:
        """
        Verifica se pode enviar um novo trigger
        
        Returns:
            bool: True se pronto para enviar
        """
        with self._lock:
            if not self.trigger_active:
                return True
            
            tempo_restante = self.trigger_duration - (time.time() - self.last_trigger_time)
            return tempo_restante <= 0
    
    def get_trigger_remaining_time(self) -> float:
        """
        Obtém o tempo restante de blockeio do trigger em segundos
        
        Returns:
            float: Tempo em segundos (0 se nenhum trigger ativo)
        """
        with self._lock:
            if not self.trigger_active:
                return 0.0
            
            return max(0.0, self.trigger_duration - (time.time() - self.last_trigger_time))


# Instância singleton para fácil acesso
_esp32_communicator: Optional[ESP32SerialCommunicator] = None
_communicator_lock = threading.Lock()

def get_esp32_communicator() -> ESP32SerialCommunicator:
    """
    Retorna instância singleton do comunicador ESP32
    
    Returns:
        ESP32SerialCommunicator: Instância do comunicador
    """
    global _esp32_communicator
    
    with _communicator_lock:
        if _esp32_communicator is None:
            _esp32_communicator = ESP32SerialCommunicator()
        return _esp32_communicator


# Funções de conveniência para compatibilidade
def send_trigger_left() -> bool:
    """Envia trigger esquerdo via ESP32"""
    return get_esp32_communicator().send_trigger_left()

def send_trigger_right() -> bool:
    """Envia trigger direito via ESP32"""
    return get_esp32_communicator().send_trigger_right()

def connect_esp32() -> bool:
    """Conecta ao ESP32"""
    return get_esp32_communicator().connect()

def disconnect_esp32():
    """Desconecta do ESP32"""
    get_esp32_communicator().disconnect()

def is_esp32_connected() -> bool:
    """Verifica se ESP32 está conectado"""
    return get_esp32_communicator().is_connected


if __name__ == "__main__":
    """Teste básico do módulo"""
    import logging
    
    # Configurar logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Teste de comunicação
    esp32 = ESP32SerialCommunicator()
    
    print("Testando comunicação com ESP32...")
    
    if esp32.connect():
        print("✓ Conectado ao ESP32")
        
        # Testar comandos
        print("Testando PING...")
        esp32.send_ping()
        time.sleep(1)
        
        print("Testando TRIGGER_LEFT...")
        esp32.send_trigger_left()
        time.sleep(1)
        
        print("Testando TRIGGER_RIGHT...")
        esp32.send_trigger_right()
        time.sleep(1)
        
        esp32.disconnect()
        print("✓ Desconectado do ESP32")
    else:
        print("✗ Falha ao conectar ESP32")