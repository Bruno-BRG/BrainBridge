import numpy as np
from collections import deque

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from brainbridge_v2.presentation.gui.styles import Theme

try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

CHANNEL_COLORS = [
    "#63b3ed", "#48bb78", "#f6ad55", "#fc8181", "#b794f4",
    "#4fd1c5", "#f687b3", "#90cdf4", "#68d391", "#fbd38d",
    "#9ae6b4", "#feb2b2", "#d6bcfa", "#81e6d9", "#fbb6ce", "#bee3f8",
]


class _BufferedBackend:
    CHANNEL_COUNT = 16
    CHANNEL_OFFSET = 100.0
    WINDOW_SECONDS = 8.0
    SAMPLE_RATE = 125
    MAX_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)

    def add_data(self, eeg_data: np.ndarray):
        sample = np.asarray(eeg_data, dtype=np.float32)
        if sample.shape != (self.CHANNEL_COUNT,) or not np.isfinite(sample).all():
            return
        # Acquisition/filter callers may reuse their arrays after this call.
        self.data_buffer.append(sample.copy())
        self.time_buffer.append(self.current_time)
        self.current_time += 1.0 / self.SAMPLE_RATE
        self._dirty = True


class _PyQtGraphBackend(_BufferedBackend):

    def __init__(self, parent_widget: QWidget):
        # Raster rendering also works on remote desktops and without a GL context.
        pg.setConfigOptions(antialias=False, foreground=Theme.WHITE)
        self.data_buffer = deque(maxlen=self.MAX_SAMPLES)
        self.time_buffer = deque(maxlen=self.MAX_SAMPLES)
        self.current_time = 0.0
        self._dirty = False

        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(useOpenGL=False)
        self.plot_widget.setBackground(Theme.PANEL_BG)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "Tempo (s)", color=Theme.WHITE)
        self.plot_widget.setLabel("left", "Canais / intervalo 100 µV", color=Theme.WHITE)
        self.plot_widget.setXRange(0, self.WINDOW_SECONDS, padding=0)
        self.plot_widget.setYRange(-self.CHANNEL_OFFSET, self.CHANNEL_OFFSET * self.CHANNEL_COUNT)
        self.plot_widget.getPlotItem().hideButtons()
        self.plot_widget.getPlotItem().disableAutoRange()
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getAxis("left").setTicks([[
            (i * self.CHANNEL_OFFSET, f"CH {i + 1:02d}")
            for i in range(self.CHANNEL_COUNT)
        ]])

        axis_pen = pg.mkPen(Theme.BTN_BORDER)
        for axis_name in ("bottom", "left"):
            axis = self.plot_widget.getPlotItem().getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(Theme.WHITE)

        layout.addWidget(self.plot_widget, 1)

        self.curves = []
        for i in range(self.CHANNEL_COUNT):
            pen = pg.mkPen(color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)], width=1)
            self.curves.append(self.plot_widget.plot(pen=pen))

        self.timer = QTimer(parent_widget)
        self.timer.timeout.connect(self._flush_plot)
        self.timer.start(33)

    def _flush_plot(self):
        if not self._dirty or len(self.data_buffer) < 2:
            return
        self._dirty = False

        times = np.asarray(self.time_buffer, dtype=np.float64)
        data = np.stack(self.data_buffer, axis=0)
        current_time = times[-1]
        window_start = max(0.0, current_time - self.WINDOW_SECONDS)
        mask = times >= window_start
        windowed_times = times[mask] - window_start
        windowed_data = data[mask]
        if len(windowed_times) < 2:
            return

        for i, curve in enumerate(self.curves):
            curve.setData(
                windowed_times,
                windowed_data[:, i] + i * self.CHANNEL_OFFSET,
                skipFiniteCheck=True,
            )


class _MatplotlibBackend(_BufferedBackend):

    def __init__(self, parent_widget: QWidget):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self.data_buffer = deque(maxlen=self.MAX_SAMPLES)
        self.time_buffer = deque(maxlen=self.MAX_SAMPLES)
        self.current_time = 0.0
        self._dirty = False

        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(12, 6), dpi=100, facecolor=Theme.BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {Theme.BG_DARK};")
        layout.addWidget(self.canvas, 1)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(Theme.PANEL_BG)
        self.ax.set_xlim(0, self.WINDOW_SECONDS)
        self.ax.set_ylim(-self.CHANNEL_OFFSET, self.CHANNEL_OFFSET * self.CHANNEL_COUNT)
        for spine in self.ax.spines.values():
            spine.set_color(Theme.BTN_BORDER)
        self.ax.tick_params(colors=Theme.GRAY, labelcolor=Theme.WHITE)
        self.ax.set_xlabel("Tempo (s)", color=Theme.WHITE)
        self.ax.set_ylabel("Canais / intervalo 100 µV", color=Theme.WHITE)
        self.ax.set_yticks(np.arange(self.CHANNEL_COUNT) * self.CHANNEL_OFFSET)
        self.ax.set_yticklabels([f"CH {i + 1:02d}" for i in range(self.CHANNEL_COUNT)], fontsize=8)
        self.figure.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.12)
        self.ax.grid(True, alpha=0.25, color=Theme.BTN_BORDER)

        self.lines = []
        for i in range(self.CHANNEL_COUNT):
            line, = self.ax.plot([], [], color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)], linewidth=0.8, alpha=0.8)
            self.lines.append(line)

        self.timer = QTimer(parent_widget)
        self.timer.timeout.connect(self._flush_plot)
        self.timer.start(50)

    def _flush_plot(self):
        if not self._dirty or len(self.data_buffer) < 2:
            return
        self._dirty = False

        times = np.asarray(self.time_buffer, dtype=np.float64)
        data = np.stack(self.data_buffer, axis=0)
        current_time = times[-1]
        window_start = max(0.0, current_time - self.WINDOW_SECONDS)
        mask = times >= window_start
        windowed_times = times[mask] - window_start
        windowed_data = data[mask]
        if len(windowed_times) < 2:
            return

        for i, line in enumerate(self.lines):
            line.set_data(windowed_times, windowed_data[:, i] + i * self.CHANNEL_OFFSET)

        self.ax.set_xlim(0, self.WINDOW_SECONDS)
        self.canvas.draw_idle()


class EEGPlotWidget(QWidget):
    """Plot EEG em tempo real — PyQtGraph (rápido) com fallback Matplotlib."""

    def __init__(self, parent=None):
        super().__init__(parent)
        backend_cls = _PyQtGraphBackend if _HAS_PYQTGRAPH else _MatplotlibBackend
        self._backend = backend_cls(self)

    def add_data(self, eeg_data: np.ndarray):
        self._backend.add_data(eeg_data)
