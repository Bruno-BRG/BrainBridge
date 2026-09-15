#!/usr/bin/env python3
"""
🎮 SIMULADOR DE DISPOSITIVO UNITY - Segundo Aparelho Real
===========================================================

Script que simula um segundo dispositivo (como o VR/Unity) conectando ao sistema
via protocolo de comunicação. Atua como um cliente real, não apenas emula.

Funcionalidades:
- Escuta broadcast UDP para descobrir IPs do servidor
- Conecta via TCP ao servidor
- Recebe e confirma dados do paciente
- Recebe tarefa (Treino/Jogo)
- Responde a triggers
- Simula ações do usuário (HAND_CLOSE, FLOWER, etc)
- Finaliza sessão com confirmação

Uso:
    python tools/unity_device_simulator.py
    
Depois, use interativamente:
    > status           # Mostra status de conexão
    > start_session    # Inicia uma sessão de teste
    > hand_close left  # Simula fechar mão esquerda
    > flower right     # Simula ação de flor direita
    > end_session      # Finaliza a sessão
    > help             # Lista todos os comandos
    > quit             # Encerra o simulador
"""

import socket
import threading
import time
import sys
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass
from queue import Queue, Empty
import json
import re


class DeviceState(Enum):
    """Estados do dispositivo simulado"""
    IDLE = "idle"                    # Esperando por broadcast
    DISCOVERING = "discovering"      # Procurando servidor via broadcast
    CONNECTED = "connected"          # TCP conectado
    RECEIVING_DATA = "receiving_data"  # Recebendo dados do paciente
    RECEIVING_TASK = "receiving_task"  # Recebendo tipo de tarefa
    READY = "ready"                  # Pronto para receber trigger
    ACTIVE = "active"                # Sessão ativa, pode enviar comandos
    ENDING = "ending"                # Finalizando sessão


@dataclass
class SimulationConfig:
    """Configuração do simulador"""
    udp_listen_port: int = 12346
    tcp_server_port: int = 12345
    udp_discover_timeout: int = 10
    auto_confirm_data: bool = True
    auto_respond_trigger: bool = False
    verbose: bool = True


class UnityDeviceSimulator:
    """Simulador que atua como um dispositivo Unity real conectado ao sistema"""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Inicializa o simulador
        
        Args:
            config: Configuração personalizada (usa padrão se None)
        """
        self.config = config or SimulationConfig()
        
        # Estado
        self.state = DeviceState.IDLE
        self.server_ip: Optional[str] = None
        self.server_ips: List[str] = []
        
        # Dados da sessão atual
        self.patient_name: Optional[str] = None
        self.patient_nivel: Optional[int] = None
        self.patient_lado: Optional[str] = None
        self.task_type: Optional[str] = None
        
        # Sockets
        self.tcp_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        
        # Threads
        self.stop_event = threading.Event()
        self.receiver_thread: Optional[threading.Thread] = None
        self.udp_listen_thread: Optional[threading.Thread] = None
        
        # Fila de mensagens recebidas
        self.message_queue: Queue = Queue()
        
        # Stats
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "commands_executed": 0,
        }

    # =========================================================================
    # DESCOBERTA E CONEXÃO
    # =========================================================================

    def discover_server(self) -> bool:
        """
        Descobre servidor via broadcast UDP
        
        Returns:
            True se servidor descoberto, False caso contrário
        """
        print("\n🔍 Iniciando descoberta de servidor...")
        print(f"   Ouvindo broadcasts UDP na porta {self.config.udp_listen_port}")
        print(f"   Timeout: {self.config.udp_discover_timeout}s\n")
        
        self.state = DeviceState.DISCOVERING
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Bind para receber broadcasts
            self.udp_socket.bind(('', self.config.udp_listen_port))
            self.udp_socket.settimeout(self.config.udp_discover_timeout)
            
            print("⏳ Aguardando broadcast do servidor...")
            
            while not self.stop_event.is_set():
                try:
                    data, addr = self.udp_socket.recvfrom(4096)
                    message = data.decode('utf-8', errors='ignore').strip()

                    # Formato atual do BrainBridge: apenas "IP1,IP2,..." (sem prefixo).
                    # Aceita também o legado "Confirm:IP1,IP2".
                    candidate = message
                    if candidate.startswith("Confirm:"):
                        candidate = candidate.replace("Confirm:", "").strip()
                    # Heurística: contém ponto e vírgula? não. Contém IP?
                    parts = [p.strip() for p in candidate.split(',') if p.strip()]
                    is_ip_list = bool(parts) and all(
                        p.replace(".", "").replace(":", "").replace(" ", "").replace("-", "").isdigit()
                        or p.count(".") == 3
                        for p in parts
                    )
                    if is_ip_list:
                        self.server_ips = parts
                        self.server_ip = self.server_ips[0] if self.server_ips else None

                        if self.server_ip:
                            print("✅ Servidor descoberto!")
                            print(f"   IPs disponíveis: {self.server_ips}")
                            print(f"   Conectando ao: {self.server_ip}\n")
                            return True
                            
                except socket.timeout:
                    print("❌ Timeout na descoberta")
                    self.state = DeviceState.IDLE
                    return False
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"⚠️  Erro durante descoberta: {e}")
                    break
                    
        finally:
            if self.udp_socket:
                self.udp_socket.close()
                self.udp_socket = None

        return False

    def connect_to_server(self) -> bool:
        """
        Conecta ao servidor via TCP
        
        Returns:
            True se conectado com sucesso, False caso contrário
        """
        if not self.server_ip:
            print("❌ Erro: Servidor IP não definido. Execute descoberta primeiro.")
            return False
        
        print(f"\n🔗 Conectando ao servidor {self.server_ip}:{self.config.tcp_server_port}...")
        
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(5.0)
            self.tcp_socket.connect((self.server_ip, self.config.tcp_server_port))
            
            self.state = DeviceState.CONNECTED
            print("✅ Conectado ao servidor via TCP!\n")
            
            # Iniciar thread de recebimento
            self.receiver_thread = threading.Thread(
                target=self._receive_messages,
                daemon=True
            )
            self.receiver_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}\n")
            self.state = DeviceState.IDLE
            return False

    # =========================================================================
    # RECEBIMENTO DE MENSAGENS
    # =========================================================================

    def _receive_messages(self):
        """Thread para receber mensagens do servidor"""
        try:
            while not self.stop_event.is_set() and self.tcp_socket:
                try:
                    self.tcp_socket.settimeout(1.0)
                    data = self.tcp_socket.recv(4096)
                    
                    if not data:
                        print("\n⚠️  Servidor desconectou")
                        break
                    
                    message = data.decode('utf-8', errors='ignore').strip()
                    if message:
                        self.stats["messages_received"] += 1
                        self.message_queue.put(message)
                        self._process_message(message)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"⚠️  Erro na recepção: {e}")
                    break
                    
        finally:
            if self.tcp_socket:
                try:
                    self.tcp_socket.close()
                except:
                    pass
                self.tcp_socket = None
            self.state = DeviceState.IDLE
            print("⚠️  Receptor parado")

    def _process_message(self, message: str):
        """Processa mensagem recebida do servidor"""
        print(f"\n📥 [{self.state.value.upper()}] Recebido do servidor:")
        print(f"   {message}\n")

        stripped = message.strip()

        # 0) JSON do paciente {"nome","nivel","lado"} (formato moderno)
        if "{" in stripped and "}" in stripped and "nome" in stripped.lower():
            self._parse_patient_json(stripped)
            self.state = DeviceState.RECEIVING_DATA

            if self.config.auto_confirm_data:
                time.sleep(0.5)
                self.send_confirm()
            return

        # Parse de dados do paciente (formato legível)
        if "Dados Paciente:" in message or "Nome:" in message:
            self._parse_patient_data(message)
            self.state = DeviceState.RECEIVING_DATA

            if self.config.auto_confirm_data:
                time.sleep(0.5)
                self.send_confirm()
            return

        # Parse de tarefa (formato moderno: "Treino"/"Jogo" puro ou "Tarefa: ...")
        if stripped.lower() in ("treino", "jogo") or "Tarefa:" in message:
            self._parse_task(message)
            self.state = DeviceState.READY
            return

        # Trigger para iniciar
        if "Trigger" in message or "HAND_CLOSE" in message:
            print("🎯 TRIGGER RECEBIDO - Iniciando tarefa!")
            self.state = DeviceState.ACTIVE
            return

        # Finalização (moderno: END_TASK / END_SESSION)
        if "END_TASK" in message or "END_SESSION" in message or "Finalizar" in message:
            print("🏁 FINALIZAÇÃO RECEBIDA - Encerrando sessão!")
            self.state = DeviceState.ENDING
            time.sleep(0.5)
            self.send_end_confirmation()
            return

    def _parse_patient_json(self, message: str):
        """Extrai dados do paciente no formato JSON moderno."""
        try:
            payload = json.loads(message.strip().splitlines()[-1] if "\n" in message else message)
            # Pode vir com múltiplas linhas coladas; tenta extrair o objeto.
            if not isinstance(payload, dict):
                raise ValueError("JSON não é objeto")
            self.patient_name = str(payload.get("nome", self.patient_name or ""))
            self.patient_nivel = int(payload.get("nivel", self.patient_nivel or 0))
            self.patient_lado = str(payload.get("lado", self.patient_lado or ""))
            print(f"👤 Paciente: {self.patient_name}")
            print(f"📊 Nível: {self.patient_nivel}")
            print(f"🖐️  Lado: {self.patient_lado}")
        except Exception:
            # Fallback: regex tolerante
            try:
                m_nome = re.search(r'"nome"\s*:\s*"([^"]+)"', message, re.IGNORECASE)
                m_nivel = re.search(r'"nivel"\s*:\s*(\d+)', message, re.IGNORECASE)
                m_lado = re.search(r'"lado"\s*:\s*"([^"]+)"', message, re.IGNORECASE)
                if m_nome:
                    self.patient_name = m_nome.group(1).strip()
                if m_nivel:
                    self.patient_nivel = int(m_nivel.group(1))
                if m_lado:
                    self.patient_lado = m_lado.group(1).strip()
                print(f"👤 Paciente: {self.patient_name}")
                print(f"📊 Nível: {self.patient_nivel}")
                print(f"🖐️  Lado: {self.patient_lado}")
            except Exception as e:
                print(f"⚠️  Erro ao parsear JSON do paciente: {e}")

    def _parse_patient_data(self, message: str):
        """Extrai dados do paciente da mensagem"""
        try:
            lines = message.split('\n')
            for line in lines:
                if "Nome:" in line:
                    self.patient_name = line.split("Nome:")[1].strip()
                elif "Nivel:" in line:
                    nivel_str = line.split("Nivel:")[1].strip()
                    self.patient_nivel = int(nivel_str)
                elif "Lado:" in line:
                    self.patient_lado = line.split("Lado:")[1].strip()
            
            if self.patient_name:
                print(f"👤 Paciente: {self.patient_name}")
                print(f"📊 Nível: {self.patient_nivel}")
                print(f"🖐️  Lado: {self.patient_lado}")
                
        except Exception as e:
            print(f"⚠️  Erro ao parsear dados do paciente: {e}")

    def _parse_task(self, message: str):
        """Extrai tipo de tarefa da mensagem"""
        try:
            if "Treino" in message:
                self.task_type = "Treino"
                print("🎓 Tarefa: TREINO")
            elif "Jogo" in message:
                self.task_type = "Jogo"
                print("🎮 Tarefa: JOGO")
            
            self.state = DeviceState.READY
            print("\n✅ Pronto! Aguardando trigger...")
            
        except Exception as e:
            print(f"⚠️  Erro ao parsear tarefa: {e}")

    # =========================================================================
    # ENVIO DE COMANDOS
    # =========================================================================

    def send_message(self, message: str) -> bool:
        """
        Envia mensagem para o servidor
        
        Args:
            message: Mensagem a enviar
            
        Returns:
            True se enviado com sucesso
        """
        if not self.tcp_socket or self.state == DeviceState.IDLE:
            print("❌ Erro: Não conectado ao servidor")
            return False
        
        try:
            full_message = message + '\n'
            self.tcp_socket.sendall(full_message.encode('utf-8'))
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            print(f"❌ Erro ao enviar mensagem: {e}")
            return False

    def send_confirm(self) -> bool:
        """Envia confirmação de recebimento"""
        print("📤 Enviando confirmação...")
        success = self.send_message("Confirm: Dados recebidos com sucesso")
        if success:
            print("✅ Confirmação enviada\n")
        return success

    def send_hand_close(self, side: str) -> bool:
        """
        Simula fechar de mão
        
        Args:
            side: "left"/"esquerda" ou "right"/"direita"
        """
        if self.state != DeviceState.ACTIVE:
            print(f"❌ Erro: Sessão não está ativa (estado: {self.state.value})")
            return False
        
        side_lower = side.lower()
        if side_lower in ['left', 'esquerda']:
            command = "LEFT_HAND_CLOSE"
            label = "ESQUERDA"
        elif side_lower in ['right', 'direita']:
            command = "RIGHT_HAND_CLOSE"
            label = "DIREITA"
        else:
            print(f"❌ Lado inválido: {side}")
            return False
        
        print(f"✊ Simulando: Mão fechada ({label})")
        success = self.send_message(command)
        if success:
            print(f"✅ Comando enviado: {command}")
            self.stats["commands_executed"] += 1
        return success

    def send_flower_action(self, side: str) -> bool:
        """
        Simula ação de flor
        
        Args:
            side: "left"/"esquerda" ou "right"/"direita"
        """
        if self.state != DeviceState.ACTIVE:
            print(f"❌ Erro: Sessão não está ativa (estado: {self.state.value})")
            return False
        
        side_lower = side.lower()
        if side_lower in ['left', 'esquerda']:
            command = "LEFT_FLOWER"
            label = "ESQUERDA"
        elif side_lower in ['right', 'direita']:
            command = "RIGHT_FLOWER"
            label = "DIREITA"
        else:
            print(f"❌ Lado inválido: {side}")
            return False
        
        print(f"🌸 Simulando: Flor ({label})")
        success = self.send_message(command)
        if success:
            print(f"✅ Comando enviado: {command}")
            self.stats["commands_executed"] += 1
        return success

    def send_end_confirmation(self) -> bool:
        """Envia confirmação de finalização"""
        print("📤 Enviando confirmação de finalização...")
        success = self.send_message("Finalizar: Tarefa concluída")
        if success:
            print("✅ Confirmação de finalização enviada\n")
            self.state = DeviceState.READY
        return success

    # =========================================================================
    # GERENCIAMENTO E STATUS
    # =========================================================================

    def print_status(self):
        """Exibe status atual do simulador"""
        print("\n" + "="*60)
        print("📊 STATUS DO DISPOSITIVO SIMULADO")
        print("="*60)
        print(f"Estado: {self.state.value.upper()}")
        print(f"Conectado: {'✅ SIM' if self.state != DeviceState.IDLE else '❌ NÃO'}")
        
        if self.server_ip:
            print(f"Servidor: {self.server_ip}:{self.config.tcp_server_port}")
        
        if self.patient_name:
            print(f"\nSessão Atual:")
            print(f"  Paciente: {self.patient_name}")
            print(f"  Nível: {self.patient_nivel}")
            print(f"  Lado: {self.patient_lado}")
            print(f"  Tarefa: {self.task_type or 'N/A'}")
        
        print(f"\nEstatísticas:")
        print(f"  Mensagens recebidas: {self.stats['messages_received']}")
        print(f"  Mensagens enviadas: {self.stats['messages_sent']}")
        print(f"  Comandos executados: {self.stats['commands_executed']}")
        print("="*60 + "\n")

    def print_help(self):
        """Exibe menu de ajuda"""
        print("\n" + "="*60)
        print("📚 COMANDOS DISPONÍVEIS")
        print("="*60)
        print("""
CONEXÃO:
  discover          - Descobre servidor via broadcast UDP
  connect           - Conecta ao servidor via TCP
  status            - Exibe status atual

SESSÃO:
  confirm           - Envia confirmação de recebimento
  hand_close left   - Simula fechar mão esquerda
  hand_close right  - Simula fechar mão direita
  flower left       - Simula ação de flor esquerda
  flower right      - Simula ação de flor direita
  end_confirm       - Confirma finalização da sessão

UTILITÁRIOS:
  help              - Exibe este menu
  quit              - Encerra o simulador
        """)
        print("="*60 + "\n")

    def shutdown(self):
        """Encerra o simulador de forma limpa"""
        print("\n🛑 Encerrando simulador...")
        self.stop_event.set()
        
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except:
                pass
        
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
        
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2.0)
        
        print("✅ Simulador encerrado")

    # =========================================================================
    # LOOP INTERATIVO
    # =========================================================================

    def run_interactive(self):
        """Executa loop interativo do simulador"""
        print("\n" + "="*70)
        print("🎮 SIMULADOR DE DISPOSITIVO UNITY - SEGUNDO APARELHO REAL")
        print("="*70)
        print("\nEste simulador atua como um segundo dispositivo conectado ao sistema")
        print("via protocolo Unity. Use 'help' para listar comandos disponíveis.\n")
        
        self.print_help()
        
        try:
            while True:
                try:
                    command = input("unity-sim> ").strip().lower()
                    
                    if not command:
                        continue
                    
                    parts = command.split()
                    cmd = parts[0]
                    args = parts[1:] if len(parts) > 1 else []
                    
                    # Processar comandos
                    if cmd == "quit" or cmd == "exit":
                        break
                    
                    elif cmd == "help":
                        self.print_help()
                    
                    elif cmd == "discover":
                        if self.discover_server():
                            print("✅ Descoberta bem-sucedida!")
                        else:
                            print("❌ Falha na descoberta")
                    
                    elif cmd == "connect":
                        if self.connect_to_server():
                            print("✅ Conexão bem-sucedida!")
                        else:
                            print("❌ Falha na conexão")
                    
                    elif cmd == "status":
                        self.print_status()
                    
                    elif cmd == "confirm":
                        self.send_confirm()
                    
                    elif cmd == "hand_close":
                        if args:
                            self.send_hand_close(args[0])
                        else:
                            print("❌ Uso: hand_close <left|right>")
                    
                    elif cmd == "flower":
                        if args:
                            self.send_flower_action(args[0])
                        else:
                            print("❌ Uso: flower <left|right>")
                    
                    elif cmd == "end_confirm":
                        self.send_end_confirmation()
                    
                    else:
                        print(f"❌ Comando desconhecido: {cmd}")
                        print("   Digite 'help' para ver comandos disponíveis")
                
                except KeyboardInterrupt:
                    print("\n")
                    continue
                except Exception as e:
                    print(f"❌ Erro ao processar comando: {e}")
        
        except KeyboardInterrupt:
            print("\n")
        
        finally:
            self.shutdown()


def main():
    """Função principal"""
    simulator = UnityDeviceSimulator()
    simulator.run_interactive()


if __name__ == "__main__":
    main()
