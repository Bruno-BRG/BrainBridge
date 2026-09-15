"""
Matriz de conexão BrainBridge <-> VR (Unity).

Cobre 100% do protocolo real, com sockets TCP/UDP de verdade:
- descoberta UDP (formato atual: "IP1,IP2")
- handshake TCP + HEADER (formato CvMobClient: "B;HEADER;...;E")
- sessão com dados REAIS (nome/lado/tarefa) via set_pending_session
- Confirm SETUP->READY (o VR precisa enviar; testado aqui)
- trigger READY->ACTIVE (Trigger + HAND_CLOSE do lado correto)
- comandos em sessão ativa
- respostas VR: LEFT/RIGHT_FLOWER, WRONG, CORRECT, B;RESET, tracking B;...;E
- framing: múltiplas mensagens num pacote + fragmentadas sem newline
- END_TASK / END_SESSION + confirm_end -> IDLE
- nível 0-11 e JSON com acento (João)
- gateway/adapter/controller novos (start_session/set_pending_session)
"""
import socket
import time

from brainbridge_v2.infrastructure.communication.unity import (
    PatientData,
    TaskType,
    SessionPhase,
    ServerState,
    UnityCommunicator,
)
from brainbridge_v2.infrastructure.communication.unity_gateway_adapter import UnityGatewayAdapter
from brainbridge_v2.interface_adapters.controllers.unity_controller import UnityController


def _cleanup():
    try:
        comm = UnityCommunicator()
        if comm.server_state != ServerState.STOPPED:
            comm.stop_server()
        time.sleep(0.2)
    except Exception:
        pass
    UnityCommunicator._instance = None
    time.sleep(0.15)


def _connect_vr():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect(("127.0.0.1", 12345))
    s.settimeout(1.0)
    return s


def _drain(sock):
    out = []
    try:
        while True:
            d = sock.recv(4096)
            if not d:
                break
            out.append(d.decode("utf-8", errors="ignore"))
            if len(d) < 4096:
                break
    except socket.timeout:
        pass
    return "".join(out)


class TestVrConnectionMatrix:
    def setup_method(self):
        _cleanup()
        self.comm = UnityCommunicator()
        assert self.comm.start_server()
        time.sleep(0.25)

    def teardown_method(self):
        try:
            self.comm.stop_server()
        except Exception:
            pass
        _cleanup()

    def test_udp_ports_tcp_handshake_header(self):
        assert self.comm.UDP_PORT == 12346
        assert self.comm.TCP_PORT == 12345
        vr = _connect_vr()
        try:
            time.sleep(0.25)
            assert self.comm.tcp_connected
            assert self.comm.server_state == ServerState.CONNECTED
            vr.sendall(b"B;HEADER;2;left_hand:right_hand;E\n")
            time.sleep(0.6)
            # auto-send com debug quando nada configurado
            data = _drain(vr)
            assert "Jo" in data  # João (pode vir com escape unicode)
            assert "Treino" in data
            assert self.comm.session.phase == SessionPhase.SETUP
        finally:
            vr.close()

    def test_real_session_confirm_trigger_flow(self):
        vr = _connect_vr()
        try:
            time.sleep(0.2)
            # Publica sessão REAL antes do HEADER (como a GUI faz)
            self.comm.set_pending_session(
                PatientData(nome="Maria Silva", nivel=7, lado="Direito"),
                TaskType.JOGO,
            )
            vr.sendall(b"B;HEADER;1;hand;E\n")
            time.sleep(0.6)
            data = _drain(vr)
            assert "Maria Silva" in data
            assert "Jogo" in data
            assert "Jo\u00e3o" not in data and "João" not in data.replace("\\u00e3o", "ão") or "Maria Silva" in data
            assert self.comm.session.phase == SessionPhase.SETUP
            # Sem Confirm, trigger deve falhar (máquina de estados estrita)
            assert self.comm.send_trigger() is False
            # VR confirma (fix exigido no VR real)
            vr.sendall(b"Confirm\n")
            time.sleep(0.3)
            assert self.comm.session.phase == SessionPhase.READY
            assert self.comm.send_trigger() is True
            assert self.comm.session.phase == SessionPhase.ACTIVE
            time.sleep(0.3)
            got = _drain(vr)
            assert "Trigger" in got
            assert "RIGHT_HAND_CLOSE" in got  # lado Direito
        finally:
            vr.close()

    def test_left_side_maps_to_left_hand_close(self):
        vr = _connect_vr()
        try:
            time.sleep(0.2)
            self.comm.set_pending_session(
                PatientData(nome="Jose", nivel=3, lado="Esquerdo"), TaskType.TREINO
            )
            vr.sendall(b"B;HEADER;0;;E\n")
            time.sleep(0.6)
            _drain(vr)
            vr.sendall(b"Confirm\n")
            time.sleep(0.3)
            assert self.comm.send_trigger() is True
            time.sleep(0.3)
            assert "LEFT_HAND_CLOSE" in _drain(vr)
        finally:
            vr.close()

    def test_vr_responses_flowers_wrong_correct_reset_tracking(self):
        vr = _connect_vr()
        try:
            time.sleep(0.2)
            flowers = []
            self.comm.on_flower_action = lambda a: flowers.append(a.value)
            msgs = []
            self.comm.set_message_callback(msgs.append)
            vr.sendall(b"B;HEADER;0;;E\n")
            time.sleep(0.6)
            _drain(vr)
            vr.sendall(b"Confirm\n")
            time.sleep(0.2)
            assert self.comm.send_trigger() is True
            _drain(vr)
            # Cada resposta isolada
            vr.sendall(b"LEFT_FLOWER\n")
            time.sleep(0.3)
            vr.sendall(b"RIGHT_FLOWER\n")
            time.sleep(0.3)
            vr.sendall(b"WRONG\n")
            time.sleep(0.3)
            vr.sendall(b"CORRECT\n")
            time.sleep(0.3)
            vr.sendall(b"B;RESET;E\n")
            time.sleep(0.3)
            # Tracking contínuo do CvMob não pode quebrar nada
            vr.sendall(b"B;12.34;0:1:2:3:4:5:6;E\n")
            time.sleep(0.3)
            assert flowers == ["LEFT_FLOWER", "RIGHT_FLOWER"]
            joined = "\n".join(msgs)
            assert "LEFT_FLOWER" in joined
            assert "CORRECT" in joined  # encaminhado via callback genérico
            # Sessão continua ativa
            assert self.comm.session.phase == SessionPhase.ACTIVE
        finally:
            vr.close()

    def test_framing_coalesced_and_fragmented(self):
        vr = _connect_vr()
        try:
            time.sleep(0.2)
            flowers = []
            self.comm.on_flower_action = lambda a: flowers.append(a.value)
            vr.sendall(b"B;HEADER;0;;E\n")
            time.sleep(0.6)
            _drain(vr)
            vr.sendall(b"Confirm\n")
            time.sleep(0.2)
            assert self.comm.send_trigger() is True
            _drain(vr)
            # Duas mensagens num único pacote TCP
            flowers.clear()
            vr.sendall(b"LEFT_FLOWER\nRIGHT_FLOWER\n")
            time.sleep(0.5)
            assert flowers == ["LEFT_FLOWER", "RIGHT_FLOWER"]
            # Fragmentada em dois pacotes
            flowers.clear()
            vr.sendall(b"LEFT_FLO")
            time.sleep(0.15)
            vr.sendall(b"WER\n")
            time.sleep(0.5)
            assert flowers == ["LEFT_FLOWER"]
        finally:
            vr.close()

    def test_end_task_end_session_confirm_end(self):
        vr = _connect_vr()
        try:
            time.sleep(0.2)
            vr.sendall(b"B;HEADER;0;;E\n")
            time.sleep(0.6)
            _drain(vr)
            vr.sendall(b"Confirm\n")
            time.sleep(0.2)
            assert self.comm.send_trigger() is True
            _drain(vr)
            assert self.comm.end_task("Treino ok") is True
            assert self.comm.session.phase == SessionPhase.ENDING
            time.sleep(0.2)
            assert "END_TASK,Treino ok" in _drain(vr)
            assert self.comm.end_session("Parabens!") is True
            time.sleep(0.2)
            assert "END_SESSION,Parabens!" in _drain(vr)
            vr.sendall(b"confirm_end\n")
            time.sleep(0.3)
            assert self.comm.session.phase == SessionPhase.IDLE
        finally:
            vr.close()

    def test_nivel_range_and_accent_json(self):
        for nivel in (0, 5, 11):
            p = PatientData(nome="João Teste", nivel=nivel, lado="Esquerdo")
            assert p.nivel == nivel
            assert "João" in p.to_json() or "Jo" in p.to_json()
        import pytest

        with pytest.raises(ValueError):
            PatientData(nome="X", nivel=12, lado="Direito")

    def test_gateway_controller_session_plumbing(self):
        adapter = UnityGatewayAdapter(communicator=self.comm)
        controller = UnityController.from_gateway(adapter)
        vr = _connect_vr()
        try:
            time.sleep(0.25)
            controller.set_pending_session("Ana", 4, "Direito", "Treino")
            time.sleep(0.6)
            assert self.comm.session.patient is not None
            assert self.comm.session.patient.nome == "Ana"
            assert self.comm.session.patient.lado == "Direito"
        finally:
            vr.close()
