"""One prediction on the application pool, with no widget ownership."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


@dataclass(frozen=True)
class InferenceOutcome:
    generation: int
    prediction: object = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RLUpdateOutcome:
    n: int = 0
    loss: float | None = None
    error: str | None = None
    updates_applied: int = 0
    detail: dict = field(default_factory=dict)


class InferenceSignals(QObject):
    finished = pyqtSignal(object)


class InferenceWorker(QRunnable):
    def __init__(self, controller, window, generation):
        super().__init__()
        self.controller = controller
        self.window = window
        self.generation = generation
        # No parent: the runnable retains this emitter until run() finishes.
        self.signals = InferenceSignals()

    def run(self):
        started = perf_counter()
        prediction, error = None, None
        try:
            prediction = self.controller.predict(self.window)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.signals.finished.emit(InferenceOutcome(
            self.generation, prediction, error, (perf_counter() - started) * 1000,
        ))


class RLUpdateWorker(QRunnable):
    """Aplica 1 passo de RL (clone-fit-swap) fora da thread da UI."""

    def __init__(self, controller, windows, labels, weights, *, epochs, lr,
                 freeze_backbone=True):
        super().__init__()
        self.controller = controller
        self.windows = windows
        self.labels = labels
        self.weights = weights
        self.epochs = epochs
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.signals = InferenceSignals()

    def run(self):
        try:
            result = self.controller.rl_online_update(
                self.windows, self.labels, sample_weights=self.weights,
                epochs=self.epochs, lr=self.lr,
                freeze_backbone=self.freeze_backbone)
            self.signals.finished.emit(RLUpdateOutcome(
                n=int(result.get("n", 0)), loss=result.get("loss"),
                updates_applied=self.controller.rl_updates_count(),
                detail=dict(result)))
        except BaseException as exc:
            self.signals.finished.emit(RLUpdateOutcome(
                error=f"{type(exc).__name__}: {exc}"))
