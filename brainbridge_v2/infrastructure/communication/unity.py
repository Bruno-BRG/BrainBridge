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
            print("Servidor já está ativo", flush=True)
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
            print(f"Servidor iniciado - ZMQ: {self.ZMQ_PORT}, TCP: {self.TCP_PORT}, UDP: {self.UDP_PORT}", flush=True)
            return True

        except Exception as e:
            print(f"Erro ao iniciar servidor: {e}", flush=True)
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

        print("Servidor parado e recursos limpos", flush=True)

    def send_command(self, command: str) -> bool:
        """
        Envia comando para Unity via ZMQ e TCP
        Garante que toda mensagem termina com newline
        Retorna True se enviado com sucesso
        """
        if not self.is_active and not self._is_server_operational():
            print("Servidor não está ativo", flush=True)
            return False

        # Garantir que a mensagem termina com newline
        if not command.endswith('\n'):
            command = command + '\n'

        success = False

        # DEBUG: mostra a payload completa que será enviada
        try:
            print(f"[SEND_COMMAND] Preparando para enviar (len={len(command)}): {command!r}", flush=True)
        except Exception:
            print("[SEND_COMMAND] Preparando para enviar (impossível exibir payload)", flush=True)

        # Enviar via ZMQ (sempre disponível quando servidor ativo)
        if self.zmq_socket:
            try:
                # ZMQ geralmente trabalha sem newline, mas enviamos mesmo assim (rstrip para visual)
                self.zmq_socket.send_string(command.rstrip('\n'))
                print(f"[ZMQ] Comando enviado: {command.rstrip()}", flush=True)
                success = True
            except Exception as e:
                print(f"[ZMQ] Erro ao enviar: {e}", flush=True)

        # Enviar via TCP se conectado
        if self.tcp_connected and self.tcp_connection:
            try:
                self.tcp_connection.sendall(command.encode('utf-8'))
                print(f"[TCP] Comando enviado: {command.rstrip()}", flush=True)
                success = True
            except Exception as e:
                print(f"[TCP] Erro ao enviar: {e}", flush=True)
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
            print(f"Comando de mão inválido: {hand}", flush=True)
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
            print(f"Comando de trigger inválido: {hand}", flush=True)
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
            print("Erro: Servidor não está ativo", flush=True)
            return False

        if not self.tcp_connected:
            print("Erro: VR não está conectado", flush=True)
            return False

        # Validar paciente
        try:
            if not isinstance(patient, PatientData):
                raise ValueError("Paciente deve ser uma instância de PatientData")
        except Exception as e:
            print(f"Erro ao validar paciente: {e}", flush=True)
            return False

        # Atualizar estado da sessão e transição para setup
        self.session.patient = patient
        self.session.task_type = task_type
        if not self._transition_session_phase(SessionPhase.SETUP):
            print(f"Erro: transição de sessão inválida para SETUP (estado atual: {self.session.phase})", flush=True)
            return False

        # 1. Enviar dados do paciente em formato JSON
        print(f"\n📋 Enviando dados do paciente (JSON)...", flush=True)
        patient_json = patient.to_json()
        # debug explícito antes de enviar
        print(f"[START_SESSION] JSON do paciente a ser enviado: {patient_json}", flush=True)

        if not self.send_command(patient_json):
            print("Erro ao enviar dados do paciente", flush=True)
            self.session.reset()
            return False

        time.sleep(0.3)  # Pequeno delay entre mensagens

        # 2. Enviar tipo de tarefa
        print(f"📌 Enviando tipo de tarefa: {task_type.value}", flush=True)
        if not self.send_command(task_type.value):
            print("Erro ao enviar tipo de tarefa", flush=True)
            self.session.reset()
            return False

        print("✅ Sessão iniciada com sucesso", flush=True)
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
                print(f"[PENDING_SESSION] Erro ao publicar sessão real: {e}", flush=True)

    def send_trigger(self) -> bool:
        """
        Envia trigger para iniciar tarefa no VR
        Envia Trigger seguido de comando de mão para o lado do paciente
        """
        if not self._is_session_waiting_trigger():
            print("Erro: Sessão não está pronta para trigger (esperado READY)", flush=True)
            return False
        if not self.session.patient:
            print("Erro: Sessão sem paciente configurado", flush=True)
            return False

        trigger_cmd = TriggerCommand.START.value
        print(f"🎯 Enviando trigger: {trigger_cmd}", flush=True)
        
        if not self.send_command(trigger_cmd):
            return False

        time.sleep(0.2)

        # Enviar comando de fechar mão baseado no lado do paciente
        hand_command = ActionCommand.RIGHT_HAND_CLOSE if self.session.patient.lado == "Direito" else ActionCommand.LEFT_HAND_CLOSE
        print(f"👋 Enviando comando: {hand_command.value}", flush=True)

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
            print("Erro: Sessão não está ativa", flush=True)
            return False

        if lado.lower() in ['direita', 'right']:
            return self.send_command(ActionCommand.RIGHT_HAND_CLOSE.value)
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command(ActionCommand.LEFT_HAND_CLOSE.value)
        else:
            print(f"Lado inválido: {lado}", flush=True)
            return False

    def send_hand_open(self, lado: str) -> bool:
        """
        Envia comando de abrir mão para o lado especificado
        Compatível com novo protocolo - libera a mão do gatilho
        """
        if not self.is_active:
            print("Erro: Servidor não está ativo", flush=True)
            return False

        if lado.lower() in ['direita', 'right']:
            return self.send_command("RIGHT_HAND_OPEN")
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command("LEFT_HAND_OPEN")
        else:
            print(f"Lado inválido: {lado}", flush=True)
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
            print(f"Lado inválido: {lado}", flush=True)
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
            print("Erro: Nenhuma tarefa ativa para finalizar", flush=True)
            return False

        if not self.tcp_connected:
            print("Erro: VR não está conectado", flush=True)
            return False

        # Construir mensagem conforme o protocolo do Unity
        if message:
            end_message = f"END_TASK,{message}"
        else:
            end_message = "END_TASK"

        print(f"✋ Finalizando tarefa: {end_message}", flush=True)

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
            print("Erro: Nenhuma sessão ativa para finalizar", flush=True)
            return False

        if self.session.phase == SessionPhase.ACTIVE and not self._transition_session_phase(SessionPhase.ENDING):
            print(f"Erro: transição inválida para ENDING (estado atual: {self.session.phase})", flush=True)
            return False

        if not self.tcp_connected:
            print("Erro: VR não está conectado", flush=True)
            return False

        # Se não houver mensagem, usa mensagem padrão
        if not message:
            message = "Sessão finalizada"

        end_message = f"END_SESSION,{message}"

        print(f"🎉 Finalizando sessão com mensagem: {end_message}", flush=True)

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

        print(f"[UDP] Iniciando broadcast: {ips}", flush=True)

        try:
            while not self.stop_event.is_set():
                # Enviar lista de IPs para descoberta
                sock.sendto(message, ('<broadcast>', self.UDP_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
        except Exception as e:
            print(f"[UDP] Erro no broadcast: {e}", flush=True)
        finally:
            sock.close()
            print("[UDP] Broadcast parado", flush=True)

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
            print(f"[TCP] Servidor ouvindo na porta {self.TCP_PORT}", flush=True)

            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    print(f"[TCP] Unity conectado de {addr}", flush=True)

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
                        print(f"[TCP] Erro no servidor: {e}", flush=True)
                    break

        except Exception as e:
            print(f"[TCP] Erro ao iniciar servidor: {e}", flush=True)
        finally:
            sock.close()
            print("[TCP] Servidor TCP parado", flush=True)

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
                        print("[TCP] Unity desconectou", flush=True)
                        break

                    chunk = data.decode('utf-8', errors='ignore')
                    recv_buffer += chunk

                    # Processa todas as linhas completas; guarda fragmento final.
                    while '\n' in recv_buffer:
                        line, recv_buffer = recv_buffer.split('\n', 1)
                        message = line.strip().strip('\r')
                        if not message:
                            continue
                        print(f"[TCP] Recebido: {message}", flush=True)

                        # Processar mensagens do novo protocolo
                        try:
                            self._process_vr_message(message)
                        except Exception as e:
                            print(f"[TCP] Erro ao processar mensagem: {e}", flush=True)

                        # Callback genérico (compatibilidade) por mensagem
                        if self.on_message_received:
                            try:
                                self.on_message_received(message)
                            except Exception as e:
                                print(f"[TCP] Erro em on_message_received: {e}", flush=True)

                    # Evita crescimento ilimitado se VR nunca enviar newline
                    # (ex: SendPub sem '\\n'): processa como mensagem única
                    # após acumular fragmento razoável e sem novos dados.
                    # Mantemos o buffer; o próximo recv com '\\n' resolve.
                    # Se o buffer ficar muito grande sem newline, despacha
                    # por tentativa de comandos conhecidos grudados.
                    if len(recv_buffer) > self.BUFFER_SIZE * 4:
                        fallback = recv_buffer.strip()
                        if fallback:
                            print(f"[TCP] Recebido (sem newline, fallback): {fallback}", flush=True)
                            try:
                                self._process_vr_message(fallback)
                            except Exception as e:
                                print(f"[TCP] Erro ao processar fallback: {e}", flush=True)
                            if self.on_message_received:
                                try:
                                    self.on_message_received(fallback)
                                except Exception:
                                    pass
                        recv_buffer = ""

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[TCP] Erro na recepção: {e}", flush=True)
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

            print("[TCP] Conexão encerrada", flush=True)

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
            print(f"✅ VR enviou HEADER (gatilho para envio de sessão): {message}", flush=True)
            # envia automaticamente os dados do paciente caso existam
            try:
                self._auto_send_session_info()
            except Exception as e:
                print(f"[PROCESS] Erro ao auto-enviar session info: {e}", flush=True)
            return

        # Reconhecer "WRONG" - erro na ação
        if msg_lower == "wrong":
            print("⚠️  VR registrou ação incorreta (WRONG)", flush=True)
            return

        # Reconhecer "CORRECT" - acerto do usuário (GUI usa via on_message_received)
        if msg_lower == "correct":
            print("✅ VR registrou ação correta (CORRECT)", flush=True)
            return

        # RESET vindo do VR (CvMobClient envia "B;RESET;E" ao receber RESET)
        if msg_lower == "b;reset;e" or msg_lower == "reset":
            print("🔄 VR solicitou RESET", flush=True)
            return

        # Dados de tracking do CvMobClient ("B;<elapsed>;...;E"): ignora silenciosamente,
        # não é comando de protocolo. Evita poluir log como "não reconhecida".
        if msg_lower.startswith("b;") and msg_lower.endswith(";e"):
            return

        # Reconhecer TRIGGER commands que o VR pode enviar como feedback
        if msg_lower == "trigger_left":
            print("🎯 VR enviou TRIGGER_LEFT (feedback)", flush=True)
            return

        if msg_lower == "trigger_right":
            print("🎯 VR enviou TRIGGER_RIGHT (feedback)", flush=True)
            return

        # Reconhecer ações de flor
        if "left_flower" in msg_lower:
            print("🌸 VR acionou FLOR ESQUERDA", flush=True)
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.LEFT_FLOWER)
            return

        if "right_flower" in msg_lower:
            print("🌸 VR acionou FLOR DIREITA", flush=True)
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.RIGHT_FLOWER)
            return

        # Confirmação de finalização de tarefa/sessão.
        if msg_lower == "confirm_end" or msg_lower == "confirmar_finalização" or "finalizar" in msg_lower:
            print("✅ VR confirmou recebimento/finalização!", flush=True)
            if self.on_confirmation:
                try:
                    self.on_confirmation()
                except Exception as e:
                    print(f"[PROCESS] Erro em on_confirmation callback: {e}", flush=True)
            if self.session.phase == SessionPhase.ENDING:
                self.session.reset()
            return

        # Confirmação de setup: sessão pronta para trigger.
        if "confirm" in msg_lower:
            print("✅ VR confirmou setup/recebimento!", flush=True)
            if self.session.phase == SessionPhase.SETUP:
                self._transition_session_phase(SessionPhase.READY)
            if self.on_confirmation:
                try:
                    self.on_confirmation()
                except Exception as e:
                    print(f"[PROCESS] Erro em on_confirmation callback: {e}", flush=True)
            return

        # Reconhecer tipo de tarefa confirmado pelo VR - NOVO
        if msg_lower == "treino" or msg_lower == "jogo":
            print(f"[TAREFA] VR confirmou tipo de tarefa: {msg_lower.upper()}", flush=True)
            return

        # Log genérico para mensagens não reconhecidas
        if message.strip():
            print(f"📨 Mensagem VR não reconhecida: {message}", flush=True)

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
                print("[AUTO_SEND] Já enviou informações desta sessão (ignora).", flush=True)
                return

            if not self.session:
                print("[AUTO_SEND] Sessão não inicializada. Não envia.", flush=True)
                return

            # Se não houver paciente ou tarefa, usa debug padrão
            if not self.session.patient or not self.session.task_type:
                print("[AUTO_SEND] ⚙️  Usando paciente de DEBUG padrão (João, nível 5, esquerdo)", flush=True)
                self.session.patient = PatientData(nome="João", nivel=5, lado="Esquerdo")
                self.session.task_type = TaskType.TREINO
                print("[AUTO_SEND] Sessão configurada com valores de debug", flush=True)

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
                print(f"[AUTO_SEND] Enviando paciente (JSON): {patient_json}", flush=True)
                sent_json = self.send_command(patient_json)
                print(f"[AUTO_SEND] send_command(JSON) -> {sent_json}", flush=True)
            except Exception as e:
                print(f"[AUTO_SEND] Erro ao montar/enviar JSON: {e}", flush=True)

            # pequeno intervalo para ordenar pacotes
            time.sleep(0.12)

            # 2) Fallback legível (compatibilidade)
            try:
                formatted = self.session.patient.format_message()
                print(f"[AUTO_SEND] Enviando paciente (legível): {formatted!r}", flush=True)
                sent_f = self.send_command(formatted)
                print(f"[AUTO_SEND] send_command(formatted) -> {sent_f}", flush=True)
            except Exception as e:
                print(f"[AUTO_SEND] Erro ao enviar formato legível: {e}", flush=True)

            time.sleep(0.12)

            # 3) Tipo de tarefa
            try:
                print(f"[AUTO_SEND] Enviando tipo de tarefa: {self.session.task_type.value}", flush=True)
                sent_t = self.send_command(self.session.task_type.value)
                print(f"[AUTO_SEND] send_command(task_type) -> {sent_t}", flush=True)
            except Exception as e:
                print(f"[AUTO_SEND] Erro ao enviar tipo de tarefa: {e}", flush=True)

            if self.session.phase == SessionPhase.IDLE:
                self._transition_session_phase(SessionPhase.SETUP)
            print("[AUTO_SEND] Informações da sessão enviadas automaticamente.", flush=True)
        except Exception as e:
            print(f"[AUTO_SEND] Exceção geral: {e}", flush=True)

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
                print(f"Debounce: pulando ação duplicada '{action}' (últim envio há {now-last:.3f}s)", flush=True)
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
        print(f"Mensagem recebida: {message}", flush=True)

    def on_connection(connected):
        print(f"Conexão: {'Conectado' if connected else 'Desconectado'}", flush=True)

    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)

    # Iniciar servidor
    if not communicator.start_server():
        print("Falha ao iniciar servidor", flush=True)
        return

    print("\n" + "="*50, flush=True)
    print("Sistema de Comunicação Unity Ativo", flush=True)
    print("="*50, flush=True)
    print("Comandos disponíveis:", flush=True)
    print("  - direita       : Controla mão direita", flush=True)
    print("  - esquerda      : Controla mão esquerda", flush=True)
    print("  - trigger_right : Gatilho mão direita", flush=True)
    print("  - trigger_left  : Gatilho mão esquerda", flush=True)
    print("  - <comando>     : Comando personalizado", flush=True)
    print("  - sair          : Encerra o programa", flush=True)
    print("="*50, flush=True)

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
        print("\nInterrompido pelo usuário", flush=True)
    finally:
        communicator.stop_server()
        print("Programa encerrado", flush=True)


if __name__ == '__main__':
    main()
