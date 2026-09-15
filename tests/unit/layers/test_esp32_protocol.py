"""Protocolo serial da ortese: regressao dos bytes enviados pelo BrainBridge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from brainbridge_v2.infrastructure.communication.esp32 import ESP32SerialCommunicator


class FakeSerial:
    def __init__(self):
        self.written = []
        self.is_open = True

    def write(self, data: bytes):
        self.written.append(bytes(data))
        return len(data)

    def flush(self):
        pass


def _communicator():
    comm = ESP32SerialCommunicator.__new__(ESP32SerialCommunicator)
    # init manual sem porta real
    import logging
    import threading
    import time

    comm.port = "TEST"
    comm.baudrate = 115200
    comm.timeout = 1.0
    comm.is_connected = True
    comm.serial_connection = FakeSerial()
    comm._lock = threading.Lock()
    comm.on_connection_changed = None
    comm.logger = logging.getLogger("test-esp32")
    comm.trigger_duration = 0.0  # sem cooldown nos testes
    comm.last_trigger_time = 0.0
    comm.trigger_active = False
    return comm


def test_trigger_sends_single_char_compatible_with_firmware():
    comm = _communicator()
    assert comm.send_trigger_command("esquerda") is True
    assert comm.send_trigger_command("direita") is True
    sent = b"".join(comm.serial_connection.written).decode()
    assert sent.replace("\n", "") == "le"
    assert "LEFT" not in sent and "RIGHT" not in sent and "PING" not in sent


def test_send_ping_is_silent():
    comm = _communicator()
    assert comm.send_ping() is True
    assert comm.serial_connection.written == []


def test_firmware_inputs_cover_new_protocol():
    src = (Path(__file__).resolve().parents[3] / "Ortese" / "ortese.cpp").read_text()
    for token in ("modoIA", "'m'", "'l'", "'e'", "'o'"):
        assert token in src
    # Outputs de motor intactos: pinos, tempos e funcoes originais.
    for token in ("pinoIN1_E = 27", "pinoIN2_E = 14", "pinoIN3_F = 13",
                  "pinoIN4_F = 12", "TEMPO_MOVIMENTO = 2000",
                  "TEMPO_PAUSA     = 1000", "void moverFlexao()",
                  "void moverExtensao()", "void pararTudo()"):
        assert token in src
