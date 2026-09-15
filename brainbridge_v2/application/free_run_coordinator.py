"""
Free-run inference coordination (modo Livre, sem VR obrigatorio).

Diferente do GameInferenceCoordinator (1 janela por tarefa do VR),
aqui a IA roda em loop continuo sobre o EEG ao vivo:
- buffer circular de window_size amostras (default 250 @ 125 Hz = 2 s);
- a cada `stride` amostras novas, emite uma janela pronta;
- sem waiting_for_response, sem generation lock do VR;
- VR/ortese sao sinks opcionais decididos pela UI.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence


@dataclass(frozen=True)
class FreeRunWindowResult:
    status: str  # "collecting" | "ready" | "inactive"
    samples_collected: int
    window: Optional[Sequence[Sequence[float]]] = None
    windows_emitted: int = 0


class FreeRunInferenceCoordinator:
    STATUS_INACTIVE = "inactive"
    STATUS_COLLECTING = "collecting"
    STATUS_READY = "ready"

    def __init__(self, *, window_size: int = 250, channels: int = 16,
                 stride: int = 125):
        if window_size <= 0:
            raise ValueError("window_size deve ser maior que zero.")
        if channels <= 0:
            raise ValueError("channels deve ser maior que zero.")
        if stride <= 0:
            raise ValueError("stride deve ser maior que zero.")
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.stride = int(stride)
        self.eeg_buffer: Deque[list[float]] = deque(maxlen=self.window_size)
        self.samples_since_last_emit = 0
        self.total_samples = 0
        self.windows_emitted = 0
        self.running = False

    def start(self) -> None:
        self.eeg_buffer.clear()
        self.samples_since_last_emit = 0
        self.total_samples = 0
        self.windows_emitted = 0
        self.running = True

    def stop(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.start()

    def configure(self, *, window_size: Optional[int] = None,
                  channels: Optional[int] = None,
                  stride: Optional[int] = None) -> None:
        if window_size is not None:
            if int(window_size) <= 0:
                raise ValueError("window_size deve ser maior que zero.")
            self.window_size = int(window_size)
            self.eeg_buffer = deque(maxlen=self.window_size)
        if channels is not None:
            if int(channels) <= 0:
                raise ValueError("channels deve ser maior que zero.")
            self.channels = int(channels)
        if stride is not None:
            if int(stride) <= 0:
                raise ValueError("stride deve ser maior que zero.")
            self.stride = int(stride)
        self.start()

    def add_sample(self, sample: Sequence[float]) -> FreeRunWindowResult:
        if not self.running:
            return FreeRunWindowResult(self.STATUS_INACTIVE, self.total_samples,
                                       None, self.windows_emitted)
        self.eeg_buffer.append(self._normalize_sample(sample))
        self.total_samples += 1
        self.samples_since_last_emit += 1
        if len(self.eeg_buffer) < self.window_size:
            return FreeRunWindowResult(self.STATUS_COLLECTING, self.total_samples,
                                       None, self.windows_emitted)
        if self.samples_since_last_emit >= self.stride:
            self.samples_since_last_emit = 0
            self.windows_emitted += 1
            window = list(self.eeg_buffer)[-self.window_size:]
            return FreeRunWindowResult(self.STATUS_READY, self.total_samples,
                                       window, self.windows_emitted)
        return FreeRunWindowResult(self.STATUS_COLLECTING, self.total_samples,
                                   None, self.windows_emitted)

    def _normalize_sample(self, sample: Sequence[float]) -> list[float]:
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        if len(values) >= self.channels:
            return [float(v) for v in values[:self.channels]]
        return [float(v) for v in values] + [0.0] * (self.channels - len(values))
