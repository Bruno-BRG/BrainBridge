"""Ingestao EEG sem Qt para o servidor.

- UDP real via UDPReceiver_BCI (sockets puros) + parser local dos formatos
  do StreamingThread (timeSeriesRaw / Ch1..Ch16 / channels / lista de 16).
- Modo simulado: gerador sintetico 16ch @125Hz (senoides + ruido).
- Publicacao thread-safe para filas asyncio dos WebSockets.
"""

import asyncio
import json
import threading
import time
from typing import Callable, List, Optional

import numpy as np


RAW_STREAM = "timeSeriesRaw"
FILTERED_STREAM = "timeSeries"


def extract_eeg_samples(data, stream: str = "raw") -> Optional[List[List[float]]]:
    """Extrai amostras 16ch de um payload UDP. Retorna lista de amostras ou None."""
    try:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        if isinstance(data, dict):
            msg_type = str(data.get("type") or "")
            if msg_type in (RAW_STREAM, FILTERED_STREAM) and "data" in data:
                if stream == "raw" and msg_type != RAW_STREAM:
                    return None
                if stream == "filtered" and msg_type != FILTERED_STREAM:
                    return None
                arr = np.asarray(data["data"], dtype=float)
                if arr.ndim == 2 and arr.shape[0] == 16:
                    return [row.tolist() for row in arr.T]
                return None
            if "Ch1" in data:
                values = [data.get(f"Ch{ch}") for ch in range(1, 17)]
                if any(isinstance(v, list) for v in values):
                    return extract_eeg_samples({"type": RAW_STREAM, "data": values}, stream)
                if all(v is not None for v in values):
                    return [[float(v) for v in values]]
                return None
            if "channels" in data:
                return extract_eeg_samples(data["channels"], stream)
            return None
        if isinstance(data, list) and len(data) == 16:
            sample = np.asarray(data, dtype=float)
            if sample.shape == (16,):
                return [sample.tolist()]
        return None
    except Exception:
        return None


class SyntheticEEG:
    """Gerador sintetico 16ch @125Hz para demonstracao sem hardware."""

    def __init__(self, sample_rate: float = 125.0, seed: int = 7):
        self.sample_rate = float(sample_rate)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def next(self) -> List[float]:
        t = self._t / self.sample_rate
        self._t += 1
        freqs = 8 + np.arange(16) / 4
        wave = 20 * np.sin(2 * np.pi * freqs * t)
        noise = self._rng.normal(0, 3, size=16)
        return (wave + noise).tolist()


class EEGHub:
    """Fonte unica de EEG do servidor (0 ou 1 ingestao ativa por vez)."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._mode = "idle"  # idle | udp | synth
        self._receiver = None
        self._synth_thread: Optional[threading.Thread] = None
        self._subs: List[tuple] = []  # (loop, asyncio.Queue)
        self._sync_subs: List[Callable] = []  # fn(message) na thread de ingestao
        self._latest: Optional[List[float]] = None
        self._samples = 0
        self._started_at: Optional[float] = None
        self._stream = "raw"
        self._last_packet_at: Optional[float] = None

    # -- ciclo de vida ----------------------------------------------------
    def connect(self, host: str = "localhost", port: int = 12345,
                simulate: bool = False, stream: str = "raw") -> dict:
        with self._lock:
            self._stop_locked()
            if simulate:
                self._start_synth_locked()
            else:
                selected = (stream or "raw").strip().lower()
                if selected not in ("raw", "filtered"):
                    raise ValueError(f"Stream EEG invalido: {stream!r} (use 'raw' ou 'filtered').")
                self._stream = selected
                try:
                    from brainbridge_v2.infrastructure.acquisition.udp_receiver import (
                        UDPReceiver_BCI)
                    receiver = UDPReceiver_BCI(host, port)
                    receiver.set_callback(self._on_udp)
                    receiver.start()
                    if not receiver.is_running:
                        raise RuntimeError(
                            f"Nao foi possivel escutar {host}:{port} para o OpenBCI.")
                    self._receiver = receiver
                    self._mode = "udp"
                    self._last_packet_at = None
                except OSError as exc:
                    raise RuntimeError(
                        f"Porta UDP {port} indisponivel em {host} ({exc}). "
                        f"Verifique se o OpenBCI GUI esta enviando para {host}:{port} "
                        f"e se outro programa nao esta usando a porta.") from exc
            self._running = True
            self._started_at = time.time()
            return self.status()

    def disconnect(self) -> dict:
        with self._lock:
            self._stop_locked()
            return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "mode": self._mode,
                "mock_mode": self._mode == "synth",
                "samples": self._samples,
                "started_at": self._started_at,
                "stream": getattr(self, "_stream", "raw"),
                "host": getattr(self._receiver, "host", None) if self._receiver else None,
                "port": getattr(self._receiver, "port", None) if self._receiver else None,
                "last_packet_at": getattr(self, "_last_packet_at", None),
            }

    # -- pub/sub -----------------------------------------------------------
    def subscribe(self) -> "asyncio.Queue":
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            self._subs.append((loop, queue))
        return queue

    def unsubscribe(self, queue) -> None:
        with self._lock:
            self._subs = [(loop, q) for loop, q in self._subs if q is not queue]

    def subscribe_sync(self, callback: Callable) -> Callable:
        """Assinatura thread-safe p/ gravacao: fn(message) na thread de ingestao."""
        with self._lock:
            self._sync_subs.append(callback)

        def _off():
            with self._lock:
                try:
                    self._sync_subs.remove(callback)
                except ValueError:
                    pass

        return _off

    def _publish(self, sample: List[float]) -> None:
        self._latest = sample
        self._samples += 1
        message = {"type": "eeg_sample", "t": time.time(), "data": sample}
        for loop, queue in list(self._subs):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, message)
            except Exception:
                continue
        for callback in list(self._sync_subs):
            try:
                callback(message)
            except Exception:
                continue

    # -- ingestao ----------------------------------------------------------
    def _on_udp(self, data) -> None:
        try:
            samples = extract_eeg_samples(data, getattr(self, "_stream", "raw"))
        except Exception:
            samples = None
        if not samples:
            return
        self._last_packet_at = time.time()
        for sample in samples:
            self._publish([float(v) for v in sample[:16]])

    def _start_synth_locked(self) -> None:
        synth = SyntheticEEG()
        stop = threading.Event()

        def _loop():
            period = 1.0 / 125.0
            while not stop.is_set():
                self._publish(synth.next())
                time.sleep(period)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        self._synth_thread = thread
        self._synth_stop = stop
        self._mode = "synth"

    def _stop_locked(self) -> None:
        stop = getattr(self, "_synth_stop", None)
        if stop is not None:
            try:
                stop.set()
            except Exception:
                pass
            self._synth_stop = None
        self._synth_thread = None
        receiver, self._receiver = self._receiver, None
        if receiver is not None:
            try:
                receiver.stop()
            except Exception:
                pass
        self._running = False
        self._mode = "idle"


_hub: Optional[EEGHub] = None
_hub_lock = threading.Lock()


def get_hub() -> EEGHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = EEGHub()
        return _hub
