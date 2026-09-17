"""Loopback serial da ortese (sem hardware): framing e mapeamento via loop://."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

serial = pytest.importorskip("serial")
from brainbridge_v2.infrastructure.communication.esp32 import (
    ESP32SerialCommunicator,
)


def _loopback(duration=0.0):
    comm = ESP32SerialCommunicator(port="loop://", timeout=1.0)
    comm.serial_connection = serial.serial_for_url("loop://", timeout=1.0)
    comm.is_connected = True
    comm.trigger_duration = duration
    comm.last_trigger_time = 0.0
    comm.trigger_active = False
    return comm


def _read(comm, n):
    data = comm.serial_connection.read(n)
    assert len(data) == n, f"timeout lendo loopback: {data!r}"
    return data


def test_loopback_single_char_framing_matches_firmware():
    comm = _loopback()
    assert comm.send_trigger_command("esquerda") is True
    assert _read(comm, 2) == b"l\n"
    comm.trigger_active = False
    assert comm.send_trigger_command("direita") is True
    assert _read(comm, 2) == b"e\n"
    assert comm.send_ia_mode() is True
    assert _read(comm, 2) == b"m\n"
    assert comm.send_stop() is True
    assert _read(comm, 2) == b"o\n"
    assert comm.send_pause_toggle() is True
    assert _read(comm, 2) == b"p\n"
    assert comm.send_reset() is True
    assert _read(comm, 2) == b"r\n"
    comm.disconnect()


def test_loopback_never_sends_banned_words():
    comm = _loopback()
    assert comm.send_trigger_command("left") is True
    assert _read(comm, 2) == b"l\n"
    comm.trigger_active = False
    assert comm.send_trigger_command("right") is True
    assert _read(comm, 2) == b"e\n"
    stream = b"l\n" + b"e\n"
    assert b"LEFT" not in stream and b"RIGHT" not in stream and b"PING" not in stream
    comm.disconnect()


def test_trigger_cooldown_blocks_double_fire():
    comm = _loopback(duration=30.0)
    assert comm.send_trigger_command("esquerda") is True
    assert _read(comm, 2) == b"l\n"
    assert comm.send_trigger_command("direita") is False
    assert comm.serial_connection.in_waiting == 0
    comm.disconnect()


def test_send_fails_closed_when_disconnected():
    comm = _loopback()
    comm.disconnect()
    assert comm.send_trigger_command("esquerda") is False
    assert comm.send_ia_mode() is False
