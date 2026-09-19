# unity_communication.py
"""
Sistema unificado de comunicação com Unity
Substitui UDP_sender.py e udp_receiver.py por uma abordagem mais simples e robusta

Protocolo:
1. Sistema: Broadcast UDP com "Confirm"
2. VR: Responde com "Header: Confirm" (ou envia HEADER)
3. Sistema: Envia Dados Paciente (JSON + legível)
4. Sistema: Envia Tarefa ("Treino" ou "Jogo")
5. Sistema: Envia Trigger + LEFT/RIGHT_HAND_CLOSE
6. VR: Responde com LEFT_FLOWER ou RIGHT_FLOWER
7. Sistema: Envia END_TASK ou END_GAME com mensagem
8. VR: Confirma finalização
"""

import socket
import threading
import time
import zmq
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable
import logging

_log = logging.getLogger(__name__)

# ========================================================================
# ENUMS e CLASSES DE DADOS DO PROTOCOLO
# ========================================================================

class TaskType(Enum):
    """Tipo de tarefa no VR"""
    TREINO = "Treino"
    JOGO = "Jogo"


class SessionPhase(Enum):
    """Fases da sessão (para máquina de estados)"""
    IDLE = "idle"
    SETUP = "setup"
    READY = "ready"
    ACTIVE = "active"
    ENDING = "ending"


class ServerState(Enum):
    """Estados do servidor"""
    STOPPED = "stopped"
    RUNNING = "running"
    CONNECTED = "connected"


class TriggerCommand(Enum):
    """Comandos de trigger do protocolo."""
    START = "Trigger"


class ActionCommand(Enum):
    """Comandos de ação do sistema"""
    LEFT_HAND_CLOSE = "LEFT_HAND_CLOSE"
    RIGHT_HAND_CLOSE = "RIGHT_HAND_CLOSE"
    LEFT_FLOWER = "LEFT_FLOWER"
    RIGHT_FLOWER = "RIGHT_FLOWER"


class EndTaskCommand(Enum):
    """Comandos de finalização de tarefa"""
    END_TRAINING = "Finalizar_tarefa_treino"
    END_GAME = "Finalizar_tarefa_jogo"


@dataclass
class PatientData:
    """Dados do paciente para enviar ao VR"""
    nome: str
    nivel: int  # 0-11 (faixas usadas pelo protocolo atual)
    lado: str   # "Direito" ou "Esquerdo"
    sessoes: int = 0  # total de sessões já realizadas (progressão)

    def __post_init__(self):
        """Valida dados do paciente"""
        if type(self.nivel) is not int:
            raise ValueError(f"Nível deve ser inteiro, recebido: {type(self.nivel).__name__}")

        if not (0 <= self.nivel <= 11):
            raise ValueError(f"Nível deve estar entre 0 e 11, recebido: {self.nivel}")

        if self.lado not in ["Direito", "Esquerdo"]:
            raise ValueError(f"Lado deve ser 'Direito' ou 'Esquerdo', recebido: {self.lado}")

        if type(self.sessoes) is not int or self.sessoes < 0:
            raise ValueError(f"Sessões deve ser inteiro >= 0, recebido: {self.sessoes!r}")

    def format_message(self) -> str:
        """
        Formata dados do paciente para envio ao VR (formato legível)
        Nota: start_session agora prefere JSON, mas mantemos formato legível para compatibilidade
        """
        return f"Dados Paciente:\nNome: {self.nome}\nNivel: {self.nivel}\nLado: {self.lado}\nSessoes: {self.sessoes}"

    def to_json(self) -> str:
        """Converte para JSON para protocolo moderno"""
        import json
        return json.dumps({
            "nome": self.nome,
            "nivel": self.nivel,
            "lado": self.lado,
            "sessoes": self.sessoes,
        })


@dataclass
class SessionState:
    """Estado da sessão atual"""
    patient: Optional[PatientData] = None
    task_type: Optional[TaskType] = None
    phase: 'SessionPhase' = None  # Suporte para máquina de estados

    def __post_init__(self):
        """Inicializa a fase"""
        if self.phase is None:
            self.phase = SessionPhase.IDLE

    def reset(self):
        """Reseta o estado da sessão"""
        self.patient = None
        self.task_type = None
        self.phase = SessionPhase.IDLE

    def transition_to(self, new_phase: 'SessionPhase') -> bool:
        """
        Tenta transicionar para nova fase
        Retorna True se transição foi válida
        """
        current = self.phase

        # Idempotente: repetir a mesma fase é considerado válido.
        if new_phase == current:
            return True

        valid_transitions = {
            SessionPhase.IDLE: [SessionPhase.SETUP],
            SessionPhase.SETUP: [SessionPhase.READY, SessionPhase.IDLE],
            SessionPhase.READY: [SessionPhase.ACTIVE, SessionPhase.IDLE],
            SessionPhase.ACTIVE: [SessionPhase.ENDING],
            SessionPhase.ENDING: [SessionPhase.IDLE],
        }

        if current in valid_transitions and new_phase in valid_transitions[current]:
            self.phase = new_phase
            return True
        return False


class UnityCommunicator:
    """
    Classe unificada para comunicação com Unity usando TCP + ZMQ
    Combina as funcionalidades de UDP_sender e udp_receiver em uma única interface
    """

    # Configurações
    UDP_PORT = 12346      # porta para broadcast de IPs
    TCP_PORT = 12345      # porta para servidor TCP Unity
    ZMQ_PORT = 5555       # porta para ZMQ publisher
    BROADCAST_INTERVAL = 1.0
    BUFFER_SIZE = 4096

    # Protocolo
    CONFIRM_HEADER = "Confirm"

    # Variáveis de classe para singleton
    _instance: Optional['UnityCommunicator'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implementa padrão singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnityCommunicator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializa o comunicador"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Estado da conexão
        self.is_active = False
        self.tcp_connected = False
        self.server_state = ServerState.STOPPED

        # Estado da sessão (novo protocolo)
        self.session = SessionState()

        # Sockets e contextos
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.tcp_connection: Optional[socket.socket] = None

        # Threads de controle
        self.broadcast_thread: Optional[threading.Thread] = None
        self.tcp_server_thread: Optional[threading.Thread] = None
        self.tcp_handler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # evita reenvio repetido após confirmação
        self._sent_session_info = False

        # Callbacks para eventos
        self.on_message_received: Optional[Callable[[str], None]] = None
        self.on_connection_changed: Optional[Callable[[bool], None]] = None

        # Callbacks para novo protocolo
        self.on_confirmation: Optional[Callable[[], None]] = None  # VR confirmou
        self.on_flower_action: Optional[Callable[[ActionCommand], None]] = None  # VR enviou FLOWER

    def _transition_server_state(self, new_state: ServerState) -> bool:
        """
        Transiciona estado do servidor com validação simples.
        """
        valid = {
            ServerState.STOPPED: {ServerState.RUNNING},
            ServerState.RUNNING: {ServerState.CONNECTED, ServerState.STOPPED},
            ServerState.CONNECTED: {ServerState.RUNNING, ServerState.STOPPED},
        }
        current = self.server_state
        if current == new_state:
            return True
        if new_state in valid.get(current, set()):
            self.server_state = new_state
            return True
        return False

    def _transition_session_phase(self, new_phase: SessionPhase) -> bool:
        """
        Wrapper para transição da máquina de estados da sessão.
        """
        return self.session.transition_to(new_phase)

    def _is_server_operational(self) -> bool:
        """
        Retorna True se servidor está RUNNING ou CONNECTED.
        """
        return self.server_state in (ServerState.RUNNING, ServerState.CONNECTED)

    def _is_server_ready_for_session(self) -> bool:
        """
        Sessão pode iniciar somente com servidor conectado e fase IDLE.
        """
        return (
            self.server_state == ServerState.CONNECTED
            and self.tcp_connected
            and self.session.phase == SessionPhase.IDLE
        )

    def _is_session_waiting_trigger(self) -> bool:
        """
        Sessão está aguardando trigger quando está em READY.
        """
        return self.session.phase == SessionPhase.READY

    @staticmethod
    def get_all_ips() -> List[str]:
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

    def start_server(self) -> bool:
        """
        Inicia o servidor de comunicação
        Retorna True se iniciado com sucesso
        """
        if self._is_server_operational():
            _log.info("Servidor já está ativo")
            return True

        try:
            # Inicializar ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://*:{self.ZMQ_PORT}")

            # Reset do evento de parada
            self.stop_event.clear()
            self._sent_session_info = False

            # Iniciar broadcast UDP
            self.broadcast_thread = threading.Thread(
                target=self._broadcast_ips,
                daemon=True
            )
            self.broadcast_thread.start()

            # Iniciar servidor TCP
            self.tcp_server_thread = threading.Thread(
                target=self._tcp_server,
                daemon=True
            )
            self.tcp_server_thread.start()

            self.is_active = True
            self._transition_server_state(ServerState.RUNNING)
            _log.info(f"Servidor iniciado - ZMQ: {self.ZMQ_PORT}, TCP: {self.TCP_PORT}, UDP: {self.UDP_PORT}")
            return True

        except Exception as e:
            _log.info(f"Erro ao iniciar servidor: {e}")
            self.stop_server()
            return False

    def stop_server(self):
        """Para o servidor e limpa recursos"""
        self.is_active = False
        self.stop_event.set()

        # Aguardar threads terminarem
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2.0)

        if self.tcp_server_thread and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.join(timeout=2.0)

        if self.tcp_handler_thread and self.tcp_handler_thread.is_alive():
            self.tcp_handler_thread.join(timeout=2.0)

        # Fechar conexão TCP
        if self.tcp_connection:
            try:
                self.tcp_connection.close()
            except Exception:
                pass
            self.tcp_connection = None

        # Fechar ZMQ
        if self.zmq_socket:
            try:
                self.zmq_socket.close()
            except Exception:
                pass
            self.zmq_socket = None

        if self.zmq_context:
            try:
                self.zmq_context.term()
            except Exception:
                pass
            self.zmq_context = None

        # Atualizar estado
        if self.tcp_connected:
            self.tcp_connected = False
            if self.on_connection_changed:
                self.on_connection_changed(False)

        self._transition_server_state(ServerState.STOPPED)
        self.session.reset()
        self._sent_session_info = False

        _log.info("Servidor parado e recursos limpos")

    def send_command(self, command: str) -> bool:
        """
        Envia comando para Unity via ZMQ e TCP
        Garante que toda mensagem termina com newline
        Retorna True se enviado com sucesso
        """
        if not self.is_active and not self._is_server_operational():
            _log.info("Servidor não está ativo")
            return False

        # Garantir que a mensagem termina com newline
        if not command.endswith('\n'):
            command = command + '\n'

        success = False

        # DEBUG: mostra a payload completa que será enviada
        try:
            _log.info(f"[SEND_COMMAND] Preparando para enviar (len={len(command)}): {command!r}")
        except Exception:
            _log.info("[SEND_COMMAND] Preparando para enviar (impossível exibir payload)")

        # Enviar via ZMQ (sempre disponível quando servidor ativo)
        if self.zmq_socket:
            try:
                # ZMQ geralmente trabalha sem newline, mas enviamos mesmo assim (rstrip para visual)
                self.zmq_socket.send_string(command.rstrip('\n'))
                _log.info(f"[ZMQ] Comando enviado: {command.rstrip()}")
                success = True
            except Exception as e:
                _log.info(f"[ZMQ] Erro ao enviar: {e}")

        # Enviar via TCP se conectado
        if self.tcp_connected and self.tcp_connection:
            try:
                self.tcp_connection.sendall(command.encode('utf-8'))
                _log.info(f"[TCP] Comando enviado: {command.rstrip()}")
                success = True
            except Exception as e:
                _log.info(f"[TCP] Erro ao enviar: {e}")
                self.tcp_connected = False
                if self.on_connection_changed:
                    self.on_connection_changed(False)

        return success

    def _send_protocol_message(self, message: str) -> bool:
        """
        Método legado para compatibilidade com testes anteriores
        Alias para send_command
        """
        return self.send_command(message)

    def send_hand_command(self, hand: str) -> bool:
        """
        Envia comando de mão (direita/esquerda)
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("RIGHT_HAND_CLOSE")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("LEFT_HAND_CLOSE")
        else:
            _log.info(f"Comando de mão inválido: {hand}")
            return False

    def send_trigger_command(self, hand: str) -> bool:
        """
        Envia comando de trigger
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("TRIGGER_RIGHT")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("TRIGGER_LEFT")
        else:
            _log.info(f"Comando de trigger inválido: {hand}")
            return False

    # ========================================================================
    # NOVO PROTOCOLO - MÉTODOS DE SESSÃO E TAREFA
    # ========================================================================
    def start_session(self, patient: PatientData, task_type: TaskType) -> bool:
        """
        Inicia uma nova sessão com paciente e tipo de tarefa
        Envia dados do paciente e tipo de tarefa ao VR
        """
        if not self._is_server_operational():
            _log.info("Erro: Servidor não está ativo")
            return False

        if not self.tcp_connected:
            _log.info("Erro: VR não está conectado")
            return False

        # Validar paciente
        try:
            if not isinstance(patient, PatientData):
                raise ValueError("Paciente deve ser uma instância de PatientData")
        except Exception as e:
            _log.info(f"Erro ao validar paciente: {e}")
            return False

        # Atualizar estado da sessão e transição para setup
        self.session.patient = patient
        self.session.task_type = task_type
        if not self._transition_session_phase(SessionPhase.SETUP):
            _log.info(f"Erro: transição de sessão inválida para SETUP (estado atual: {self.session.phase})")
            return False

        # 1. Enviar dados do paciente em formato JSON
        _log.info(f"\n📋 Enviando dados do paciente (JSON)...")
        patient_json = patient.to_json()
        # debug explícito antes de enviar
        _log.info(f"[START_SESSION] JSON do paciente a ser enviado: {patient_json}")

        if not self.send_command(patient_json):
            _log.info("Erro ao enviar dados do paciente")
            self.session.reset()
            return False

        time.sleep(0.3)  # Pequeno delay entre mensagens

        # 2. Enviar tipo de tarefa
        _log.info(f"📌 Enviando tipo de tarefa: {task_type.value}")
        if not self.send_command(task_type.value):
            _log.info("Erro ao enviar tipo de tarefa")
            self.session.reset()
            return False

        _log.info("✅ Sessão iniciada com sucesso")
        # Marca como enviado para que o HEADER posterior do VR não duplique
        # o envio via _auto_send_session_info.
        self._sent_session_info = True
        return True

    def set_pending_session(self, patient: PatientData, task_type: TaskType) -> None:
        """
        Configura paciente/tarefa sem exigir envio imediato.

        Usado pela GUI no início da gravação: quando o VR enviar HEADER,
        o _auto_send_session_info usa esses dados reais em vez do debug padrão.
        Se a sessão estiver IDLE, transiciona para SETUP.
        Se o VR já estiver conectado, envia imediatamente (corrige caso o
        auto-send anterior tenha usado dados de debug).
        """
        self.session.patient = patient
        self.session.task_type = task_type
        if self.session.phase == SessionPhase.IDLE:
            self._transition_session_phase(SessionPhase.SETUP)
        # Permite (re)envio com dados reais.
        self._sent_session_info = False
        if self._is_server_operational() and self.tcp_connected:
            try:
                self._auto_send_session_info()
            except Exception as e:
                _log.info(f"[PENDING_SESSION] Erro ao publicar sessão real: {e}")

    def send_trigger(self) -> bool:
        """
        Envia trigger para iniciar tarefa no VR
        Envia Trigger seguido de comando de mão para o lado do paciente
        """
        if not self._is_session_waiting_trigger():
            _log.info("Erro: Sessão não está pronta para trigger (esperado READY)")
            return False
        if not self.session.patient:
            _log.info("Erro: Sessão sem paciente configurado")
            return False

        trigger_cmd = TriggerCommand.START.value
        _log.info(f"🎯 Enviando trigger: {trigger_cmd}")
        
        if not self.send_command(trigger_cmd):
            return False

        time.sleep(0.2)

        # Enviar comando de fechar mão baseado no lado do paciente
        hand_command = ActionCommand.RIGHT_HAND_CLOSE if self.session.patient.lado == "Direito" else ActionCommand.LEFT_HAND_CLOSE
        _log.info(f"👋 Enviando comando: {hand_command.value}")

        success = self.send_command(hand_command.value)

        if success:
            self._transition_session_phase(SessionPhase.ACTIVE)

        return success

    def send_hand_close(self, lado: str) -> bool:
        """
        Envia comando de fechar mão para o lado especificado
        Compatível com novo protocolo
        """
        if not self._is_session_active_for_commands():
            _log.info("Erro: Sessão não está ativa")
            return False

        if lado.lower() in ['direita', 'right']:
            return self.send_command(ActionCommand.RIGHT_HAND_CLOSE.value)
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command(ActionCommand.LEFT_HAND_CLOSE.value)
        else:
            _log.info(f"Lado inválido: {lado}")
            return False

    def send_hand_open(self, lado: str) -> bool:
        """
        Envia comando de abrir mão para o lado especificado
        Compatível com novo protocolo - libera a mão do gatilho
        """
        if not self.is_active:
            _log.info("Erro: Servidor não está ativo")
            return False

        if lado.lower() in ['direita', 'right']:
            return self.send_command("RIGHT_HAND_OPEN")
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command("LEFT_HAND_OPEN")
        else:
            _log.info(f"Lado inválido: {lado}")
            return False

    def send_flower_action(self, lado: str) -> bool:
        """
        Envia comando de ação de flor
        (simulação de resposta do VR)
        """
        if lado.lower() in ['direita', 'right']:
            return self.send_command(ActionCommand.RIGHT_FLOWER.value)
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command(ActionCommand.LEFT_FLOWER.value)
        else:
            _log.info(f"Lado inválido: {lado}")
            return False

    def end_task(self, message: str = "") -> bool:
        """
        Finaliza a tarefa atual (Treino ou Jogo)
        Envia END_TASK conforme protocolo Unity
        
        Uso:
        - Treino: end_task() -> finaliza tarefa de treino
        - Jogo: end_task() -> finaliza tarefa de jogo, depois chamar end_session(mensagem)
        
        Formato: "END_TASK" ou "END_TASK,mensagem"
        """
        if self.session.phase != SessionPhase.ACTIVE:
            _log.info("Erro: Nenhuma tarefa ativa para finalizar")
            return False

        if not self.tcp_connected:
            _log.info("Erro: VR não está conectado")
            return False

        # Construir mensagem conforme o protocolo do Unity
        if message:
            end_message = f"END_TASK,{message}"
        else:
            end_message = "END_TASK"

        _log.info(f"✋ Finalizando tarefa: {end_message}")

        success = self.send_command(end_message)

        if success:
            self._transition_session_phase(SessionPhase.ENDING)

        return success

    def end_session(self, message: str = "") -> bool:
        """
        Finaliza a sessão com mensagem motivacional
        Usado após end_task() para encerrar sessão de jogo/treino
        
        Envia END_SESSION,<mensagem motivacional> ao VR
        
        Exemplo de uso:
        - end_task()  # Finaliza a tarefa
        - end_session("Parabéns! Você acertou tudo!")  # Envia mensagem motivacional
        """
        if self.session.phase not in (SessionPhase.ACTIVE, SessionPhase.ENDING):
            _log.info("Erro: Nenhuma sessão ativa para finalizar")
            return False

        if self.session.phase == SessionPhase.ACTIVE and not self._transition_session_phase(SessionPhase.ENDING):
            _log.info(f"Erro: transição inválida para ENDING (estado atual: {self.session.phase})")
            return False

        if not self.tcp_connected:
            _log.info("Erro: VR não está conectado")
            return False

        # Se não houver mensagem, usa mensagem padrão
        if not message:
            message = "Sessão finalizada"

        end_message = f"END_SESSION,{message}"

        _log.info(f"🎉 Finalizando sessão com mensagem: {end_message}")

        success = self.send_command(end_message)

        if success:
            self.session.reset()

        return success

    def set_confirmation_callback(self, callback: Callable[[], None]):
        """Define callback para confirmação do VR"""
        self.on_confirmation = callback

    def set_flower_callback(self, callback: Callable[[ActionCommand], None]):
        """Define callback para ações de flor do VR"""
        self.on_flower_action = callback

    def _is_session_active_for_commands(self) -> bool:
        """
        Verifica se sessão está ativa e pronta para comandos
        Retorna True se a fase da sessão é ACTIVE.
        """
        return self.session.phase == SessionPhase.ACTIVE

    def _broadcast_ips(self):
        """
        Thread para broadcast dos IPs via UDP
        """
        ips = self.get_all_ips()
        message = ','.join(ips).encode('utf-8')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        _log.info(f"[UDP] Iniciando broadcast: {ips}")

        try:
            while not self.stop_event.is_set():
                # Enviar lista de IPs para descoberta
                sock.sendto(message, ('<broadcast>', self.UDP_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
        except Exception as e:
            _log.info(f"[UDP] Erro no broadcast: {e}")
        finally:
            sock.close()
            _log.info("[UDP] Broadcast parado")

    def _tcp_server(self):
        """
        Thread para servidor TCP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Timeout para permitir verificação de stop_event

        try:
            sock.bind(('', self.TCP_PORT))
            sock.listen(1)
            _log.info(f"[TCP] Servidor ouvindo na porta {self.TCP_PORT}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    _log.info(f"[TCP] Unity conectado de {addr}")

                    self.tcp_connection = conn
                    self.tcp_connected = True
                    self._transition_server_state(ServerState.CONNECTED)

                    if self.on_connection_changed:
                        self.on_connection_changed(True)

                    # Iniciar thread para lidar com esta conexão
                    self.tcp_handler_thread = threading.Thread(
                        target=self._handle_tcp_connection,
                        args=(conn, addr),
                        daemon=True
                    )
                    self.tcp_handler_thread.start()

                    # Aguardar conexão terminar antes de aceitar nova
                    self.tcp_handler_thread.join()

                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        _log.info(f"[TCP] Erro no servidor: {e}")
                    break

        except Exception as e:
            _log.info(f"[TCP] Erro ao iniciar servidor: {e}")
        finally:
            sock.close()
            _log.info("[TCP] Servidor TCP parado")

    def _handle_tcp_connection(self, conn: socket.socket, addr):
        """
        Lida com uma conexão TCP específica
        Processa mensagens do VR e dispara callbacks apropriados.

        Framing: mensagens são delimitadas por '\\n' (aceita '\\r\\n').
        Acumula fragmentos e processa cada linha completa separadamente,
        para não perder comandos quando o SO coalesceia pacotes.
        """
        recv_buffer = ""
        try:
            conn.settimeout(1.0)

            while not self.stop_event.is_set() and self.tcp_connected:
                try:
                    data = conn.recv(self.BUFFER_SIZE)
                    if not data:
                        _log.info("[TCP] Unity desconectou")
                        break

                    chunk = data.decode('utf-8', errors='ignore')
                    recv_buffer += chunk

                    # Processa todas as linhas completas; guarda fragmento final.
                    while '\n' in recv_buffer:
                        line, recv_buffer = recv_buffer.split('\n', 1)
                        message = line.strip().strip('\r')
                        if not message:
                            continue
                        _log.info(f"[TCP] Recebido: {message}")

                        # Processar mensagens do novo protocolo
                        try:
                            self._process_vr_message(message)
                        except Exception as e:
                            _log.info(f"[TCP] Erro ao processar mensagem: {e}")

                        # Callback genérico (compatibilidade) por mensagem
                        if self.on_message_received:
                            try:
                                self.on_message_received(message)
                            except Exception as e:
                                _log.info(f"[TCP] Erro em on_message_received: {e}")

                    # Evita crescimento ilimitado se VR nunca enviar newline
                    # (ex: SendPub sem '\\n'): processa como mensagem única
                    # após acumular fragmento razoável e sem novos dados.
                    # Mantemos o buffer; o próximo recv com '\\n' resolve.
                    # Se o buffer ficar muito grande sem newline, despacha
                    # por tentativa de comandos conhecidos grudados.
                    if len(recv_buffer) > self.BUFFER_SIZE * 4:
                        fallback = recv_buffer.strip()
                        if fallback:
                            _log.info(f"[TCP] Recebido (sem newline, fallback): {fallback}")
                            try:
                                self._process_vr_message(fallback)
                            except Exception as e:
                                _log.info(f"[TCP] Erro ao processar fallback: {e}")
                            if self.on_message_received:
                                try:
                                    self.on_message_received(fallback)
                                except Exception:
                                    pass
                        recv_buffer = ""

                except socket.timeout:
                    continue
                except Exception as e:
                    _log.info(f"[TCP] Erro na recepção: {e}")
                    break

        finally:
            try:
                conn.close()
            except Exception:
                pass

            # reset flag para próxima conexão
            self._sent_session_info = False

            self.tcp_connection = None
            self.tcp_connected = False
            if self._is_server_operational() and not self.stop_event.is_set():
                self._transition_server_state(ServerState.RUNNING)

            if self.on_connection_changed:
                self.on_connection_changed(False)

            _log.info("[TCP] Conexão encerrada")

    def _process_vr_message(self, message: str):
        """
        Processa mensagens recebidas do VR conforme o protocolo
        Mensagens esperadas:
        - "WRONG" - Ação errada do usuário
        - "LEFT_FLOWER" / "RIGHT_FLOWER" - Flor acionada
        - "confirm_end" / "Confirmar_finalização" - Confirmação de fim (MELHORADO)
        - "B;HEADER;..." ou "HEADER" - header inicial enviado pelo Unity (gatilho para auto-send)
        - "Treino" / "Jogo" - Confirmação de tipo de tarefa (NOVO)
        """
        msg_lower = message.lower().strip()

        # Reconhecer header enviado pelo Unity (ex: B;HEADER;0;;E ou HEADER)
        if "header" in msg_lower or message.strip().upper().startswith("B;HEADER"):
            _log.info(f"✅ VR enviou HEADER (gatilho para envio de sessão): {message}")
            # envia automaticamente os dados do paciente caso existam
            try:
                self._auto_send_session_info()
            except Exception as e:
                _log.info(f"[PROCESS] Erro ao auto-enviar session info: {e}")
            return

        # Reconhecer "WRONG" - erro na ação
        if msg_lower == "wrong":
            _log.info("⚠️  VR registrou ação incorreta (WRONG)")
            return

        # Reconhecer "CORRECT" - acerto do usuário (GUI usa via on_message_received)
        if msg_lower == "correct":
            _log.info("✅ VR registrou ação correta (CORRECT)")
            return

        # RESET vindo do VR (CvMobClient envia "B;RESET;E" ao receber RESET)
        if msg_lower == "b;reset;e" or msg_lower == "reset":
            _log.info("🔄 VR solicitou RESET")
            return

        # Dados de tracking do CvMobClient ("B;<elapsed>;...;E"): ignora silenciosamente,
        # não é comando de protocolo. Evita poluir log como "não reconhecida".
        if msg_lower.startswith("b;") and msg_lower.endswith(";e"):
            return

        # Reconhecer TRIGGER commands que o VR pode enviar como feedback
        if msg_lower == "trigger_left":
            _log.info("🎯 VR enviou TRIGGER_LEFT (feedback)")
            return

        if msg_lower == "trigger_right":
            _log.info("🎯 VR enviou TRIGGER_RIGHT (feedback)")
            return

        # Reconhecer ações de flor
        if "left_flower" in msg_lower:
            _log.info("🌸 VR acionou FLOR ESQUERDA")
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.LEFT_FLOWER)
            return

        if "right_flower" in msg_lower:
            _log.info("🌸 VR acionou FLOR DIREITA")
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.RIGHT_FLOWER)
            return

        # Confirmação de finalização de tarefa/sessão.
        if msg_lower == "confirm_end" or msg_lower == "confirmar_finalização" or "finalizar" in msg_lower:
            _log.info("✅ VR confirmou recebimento/finalização!")
            if self.on_confirmation:
                try:
                    self.on_confirmation()
                except Exception as e:
                    _log.info(f"[PROCESS] Erro em on_confirmation callback: {e}")
            if self.session.phase == SessionPhase.ENDING:
                self.session.reset()
            return

        # Confirmação de setup: sessão pronta para trigger.
        if "confirm" in msg_lower:
            _log.info("✅ VR confirmou setup/recebimento!")
            if self.session.phase == SessionPhase.SETUP:
                self._transition_session_phase(SessionPhase.READY)
            if self.on_confirmation:
                try:
                    self.on_confirmation()
                except Exception as e:
                    _log.info(f"[PROCESS] Erro em on_confirmation callback: {e}")
            return

        # Reconhecer tipo de tarefa confirmado pelo VR - NOVO
        if msg_lower == "treino" or msg_lower == "jogo":
            _log.info(f"[TAREFA] VR confirmou tipo de tarefa: {msg_lower.upper()}")
            return

        # Log genérico para mensagens não reconhecidas
        if message.strip():
            _log.info(f"📨 Mensagem VR não reconhecida: {message}")

    def _auto_send_session_info(self):
        """
        Envia automaticamente (uma vez) as informações da sessão ao VR:
        1) JSON do paciente (se existir)
        2) fallback legível (format_message)
        3) tipo de tarefa (TaskType.value)
        Gatilho: chamado quando o VR envia 'HEADER' ou 'confirm'.
        Caso nenhum paciente esteja configurado, usa debug padrão (João, nível 5, esquerdo).
        """
        try:
            # evita reenvio
            if getattr(self, "_sent_session_info", False):
                _log.info("[AUTO_SEND] Já enviou informações desta sessão (ignora).")
                return

            if not self.session:
                _log.info("[AUTO_SEND] Sessão não inicializada. Não envia.")
                return

            # Se não houver paciente ou tarefa, usa debug padrão
            if not self.session.patient or not self.session.task_type:
                _log.info("[AUTO_SEND] ⚙️  Usando paciente de DEBUG padrão (João, nível 5, esquerdo)")
                self.session.patient = PatientData(nome="João", nivel=5, lado="Esquerdo")
                self.session.task_type = TaskType.TREINO
                _log.info("[AUTO_SEND] Sessão configurada com valores de debug")

            # marca como enviado imediatamente para evitar race
            self._sent_session_info = True

            import json
            # 1) JSON (moderno, inclui sessoes para progressão no VR)
            try:
                patient_json = self.session.patient.to_json() if hasattr(self.session.patient, "to_json") else json.dumps({
                    "nome": self.session.patient.nome,
                    "nivel": self.session.patient.nivel,
                    "lado": self.session.patient.lado,
                    "sessoes": getattr(self.session.patient, "sessoes", 0),
                })
                _log.info(f"[AUTO_SEND] Enviando paciente (JSON): {patient_json}")
                sent_json = self.send_command(patient_json)
                _log.info(f"[AUTO_SEND] send_command(JSON) -> {sent_json}")
            except Exception as e:
                _log.info(f"[AUTO_SEND] Erro ao montar/enviar JSON: {e}")

            # pequeno intervalo para ordenar pacotes
            time.sleep(0.12)

            # 2) Fallback legível (compatibilidade)
            try:
                formatted = self.session.patient.format_message()
                _log.info(f"[AUTO_SEND] Enviando paciente (legível): {formatted!r}")
                sent_f = self.send_command(formatted)
                _log.info(f"[AUTO_SEND] send_command(formatted) -> {sent_f}")
            except Exception as e:
                _log.info(f"[AUTO_SEND] Erro ao enviar formato legível: {e}")

            time.sleep(0.12)

            # 3) Tipo de tarefa
            try:
                _log.info(f"[AUTO_SEND] Enviando tipo de tarefa: {self.session.task_type.value}")
                sent_t = self.send_command(self.session.task_type.value)
                _log.info(f"[AUTO_SEND] send_command(task_type) -> {sent_t}")
            except Exception as e:
                _log.info(f"[AUTO_SEND] Erro ao enviar tipo de tarefa: {e}")

            if self.session.phase == SessionPhase.IDLE:
                self._transition_session_phase(SessionPhase.SETUP)
            _log.info("[AUTO_SEND] Informações da sessão enviadas automaticamente.")
        except Exception as e:
            _log.info(f"[AUTO_SEND] Exceção geral: {e}")

    def set_message_callback(self, callback: Callable[[str], None]):
        """Define callback para mensagens recebidas"""
        self.on_message_received = callback

    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Define callback para mudanças de conexão"""
        self.on_connection_changed = callback


# Classe para compatibilidade com código existente
class UDP_sender:
    """Classe de compatibilidade que mapeia para UnityCommunicator"""

    _communicator = UnityCommunicator()
    # simple debounce state to avoid duplicate rapid sends
    _last_sent_times = {}
    _debounce_seconds = 0.2  # ignore same action within 200 ms

    @classmethod
    def init_zmq_socket(cls, broadcast_duration=3.0):
        """Inicializa o sistema de comunicação"""
        return cls._communicator.start_server()

    @classmethod
    def stop_zmq_socket(cls):
        """Para o sistema de comunicação"""
        cls._communicator.stop_server()

    @classmethod
    def enviar_sinal(cls, action: str) -> bool:
        """
        Envia sinal de ação
        Compatível com chamadas legadas
        """
        # Usa o singleton atual (não o _communicator capturado na definição
        # da classe): se a instância foi recriada, a antiga estaria parada
        # e o envio falharia com "Servidor não está ativo".
        communicator = UnityCommunicator()
        # debounce: avoid sending the same action repeatedly in a short window
        try:
            now = time.time()
            key = action.lower()
            last = cls._last_sent_times.get(key)
            if last is not None and (now - last) < cls._debounce_seconds:
                # skip duplicate
                _log.info(f"Debounce: pulando ação duplicada '{action}' (últim envio há {now-last:.3f}s)")
                return False
            cls._last_sent_times[key] = now
        except Exception:
            # if anything goes wrong in debounce, proceed to send (fail-open)
            pass

        # Mapear ações legadas
        if action.lower() == 'direita':
            return communicator.send_hand_command('direita')
        elif action.lower() == 'esquerda':
            return communicator.send_hand_command('esquerda')
        elif action.lower() == 'trigger_right':
            return communicator.send_trigger_command('direita')
        elif action.lower() == 'trigger_left':
            return communicator.send_trigger_command('esquerda')
        else:
            return communicator.send_command(action)

    @classmethod
    def is_server_active(cls) -> bool:
        """Verifica se o servidor está ativo"""
        return cls._communicator.is_active

    @classmethod
    def restart_broadcast(cls, duration=3.0):
        """Reinicia o broadcast (não necessário na nova implementação)"""
        return True  # Broadcast é contínuo na nova implementação

    # Métodos legacy mantidos para compatibilidade
    @staticmethod
    def get_all_ips():
        return UnityCommunicator.get_all_ips()

    @staticmethod
    def get_local_ip():
        all_ips = UnityCommunicator.get_all_ips()
        for ip in all_ips:
            if ip != '127.0.0.1':
                return ip
        return all_ips[0] if all_ips else '127.0.0.1'


# Função principal para demonstração
def main():

    """Função principal para teste do sistema"""
    communicator = UnityCommunicator()

    def on_message(message):
        _log.info(f"Mensagem recebida: {message}")

    def on_connection(connected):
        _log.info(f"Conexão: {'Conectado' if connected else 'Desconectado'}")

    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)

    # Iniciar servidor
    if not communicator.start_server():
        _log.info("Falha ao iniciar servidor")
        return

    _log.info("\n" + "="*50)
    _log.info("Sistema de Comunicação Unity Ativo")
    _log.info("="*50)
    _log.info("Comandos disponíveis:")
    _log.info("  - direita       : Controla mão direita")
    _log.info("  - esquerda      : Controla mão esquerda")
    _log.info("  - trigger_right : Gatilho mão direita")
    _log.info("  - trigger_left  : Gatilho mão esquerda")
    _log.info("  - <comando>     : Comando personalizado")
    _log.info("  - sair          : Encerra o programa")
    _log.info("="*50)

    try:
        while True:
            comando = input("\nDigite um comando: ").strip()

            if comando.lower() == 'sair':
                break
            elif comando.lower() == 'direita':
                communicator.send_hand_command('direita')
            elif comando.lower() == 'esquerda':
                communicator.send_hand_command('esquerda')
            elif comando.lower() == 'trigger_right':
                communicator.send_trigger_command('direita')
            elif comando.lower() == 'trigger_left':
                communicator.send_trigger_command('esquerda')
            elif comando:
                communicator.send_command(comando)

    except KeyboardInterrupt:
        _log.info("\nInterrompido pelo usuário")
    finally:
        communicator.stop_server()
        _log.info("Programa encerrado")


if __name__ == '__main__':
    main()
