"""Loopback Unity/VR sem headset: simula o CvMobClient+AnimationTrigger.

Descoberta via broadcast UDP 12346 -> TCP 12345; roda o protocolo completo
(sessao JSON, tarefa, Confirm, Trigger, HAND_CLOSE, CORRECT, END_TASK,
confirm_end, END_SESSION) e valida cada passo. Uso: python tools/...py
"""
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainbridge_v2.infrastructure.communication.unity import (
    PatientData,
    TaskType,
    UnityCommunicator,
)

TCP_PORT = 12345
UDP_PORT = 12346
TIMEOUT = 8.0


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FALHOU'}] {name}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        raise SystemExit(f"falha em: {name} {detail}")


def wait_until(fn, timeout=TIMEOUT, desc=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.05)
    raise SystemExit(f"timeout aguardando: {desc}")


def main():
    comm = UnityCommunicator()
    received = []
    confirmed = []
    comm.on_message_received = received.append
    comm.set_confirmation_callback(lambda: confirmed.append(time.time()))
    check("servidor sobe (ZMQ+TCP+UDP)", comm.start_server() is True)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind(("0.0.0.0", UDP_PORT))
    udp.settimeout(TIMEOUT)
    data, _ = udp.recvfrom(4096)
    ips = [p.strip() for p in data.decode("utf-8", "ignore").split(",") if p.strip()]
    check("broadcast UDP traz IP", len(ips) > 0, data[:80].decode("utf-8", "ignore"))
    udp.close()

    tcp = socket.create_connection((ips[0], TCP_PORT), timeout=TIMEOUT)
    tcp.settimeout(TIMEOUT)
    wait_until(lambda: comm.tcp_connected, desc="aceite TCP")
    check("VR conecta via TCP", True)
    stream = tcp.makefile("r", encoding="utf-8", newline="\n")

    ok = comm.start_session(PatientData("VR Teste", 3, "Direito"), TaskType.JOGO)
    check("start_session envia JSON+tarefa", ok)
    line1 = stream.readline().strip()
    payload = json.loads(line1)
    check("paciente JSON tem nome/nivel/lado/sessoes",
            payload == {"nome": "VR Teste", "nivel": 3, "lado": "Direito", "sessoes": 0},
            line1[:120])
    line2 = stream.readline().strip()
    check("tarefa Jogo em mensagem isolada", line2 == "Jogo", line2)
    tcp.sendall(b"Confirm\n")
    wait_until(lambda: comm.session.phase.value == "ready", desc="fase READY")
    check("Confirm leva a READY", True)

    check("send_trigger dispara sequencia", comm.send_trigger() is True)
    check("VR recebe Trigger", stream.readline().strip() == "Trigger")
    check("VR recebe RIGHT_HAND_CLOSE", stream.readline().strip() == "RIGHT_HAND_CLOSE")
    tcp.sendall(b"CORRECT\n")
    wait_until(lambda: any("CORRECT" in m for m in received), desc="veredicto CORRECT")
    check("veredicto CORRECT chega ao host", True)

    check("end_task envia END_TASK,msg", comm.end_task("Boa!") is True)
    check("VR recebe END_TASK,Boa!", stream.readline().strip() == "END_TASK,Boa!")
    tcp.sendall(b"confirm_end\n")
    wait_until(lambda: len(confirmed) > 0, desc="confirm_end")
    check("confirm_end dispara on_confirmation", True)

    check("end_session envia END_SESSION", comm.end_session("Fim") is True)
    check("VR recebe END_SESSION,Fim", stream.readline().strip() == "END_SESSION,Fim")

    tcp.close()
    comm.stop_server()
    print("PASS: protocolo Unity/VR ponta a ponta via loopback")


if __name__ == "__main__":
    threading.current_thread().name = "vr-loopback-main"
    main()
