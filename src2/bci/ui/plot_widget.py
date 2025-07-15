import numpy as np
from collections import deque
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

class EEGPlotWidget(QWidget):
    """Widget para visualização de dados EEG em tempo real"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = 16
        self.window_size = 500  # 4 segundos @ 125Hz
        self.data_buffers = [deque(maxlen=self.window_size) for _ in range(self.channels)]
        self.selected_channel = -1  # -1 = todos os canais
        self.scale = 100  # ±100µV
        
        # Initialize time-related buffers
        self.time_buffer = deque(maxlen=self.window_size)
        self.data_buffer = deque(maxlen=self.window_size) 
        self.current_time = 0.0
        
        # Timer para atualização do plot
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(50)  # Atualizar a cada 50ms
        
        self._build_ui()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(50)  # Atualiza a 20FPS

    def _build_ui(self):
        layout = QVBoxLayout()
        
        # Controles
        controls = QHBoxLayout()
        
        # Seleção de canal
        channel_label = QLabel("Canal:")
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Todos")
        for i in range(self.channels):
            self.channel_combo.addItem(f"Canal {i+1}")
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        controls.addWidget(channel_label)
        controls.addWidget(self.channel_combo)
        
        # Escala
        scale_label = QLabel("Escala:")
        self.scale_combo = QComboBox()
        scales = ["Auto", "±50µV", "±100µV", "±200µV", "±500µV"]
        self.scale_combo.addItems(scales)
        self.scale_combo.setCurrentText("±100µV")
        self.scale_combo.currentTextChanged.connect(self._scale_changed)
        controls.addWidget(scale_label)
        controls.addWidget(self.scale_combo)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Amplitude', 'µV')
        self.plot_widget.setLabel('bottom', 'Tempo', 's')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setXRange(0, 4)  # Initial 4-second window
        self.plot_curves = []
        
        # Criar curvas para cada canal
        for i in range(self.channels):
            color = pg.intColor(i, self.channels)
            curve = self.plot_widget.plot(pen=color, name=f'Canal {i+1}')
            self.plot_curves.append(curve)
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def add_data(self, data: np.ndarray):
        """Adiciona novos dados ao plot"""
        if len(data) == self.channels:
            # Atualizar buffers de tempo e dados
            self.time_buffer.append(self.current_time)
            self.data_buffer.append(data)
            self.current_time += 1/125.0  # Assumindo frequência de 125Hz
            
            # Atualizar buffers individuais dos canais
            for i, value in enumerate(data):
                self.data_buffers[i].append(value)


    def _update_plot(self):
        """Atualiza o plot periodicamente"""
        if len(self.time_buffer) == 0:
            return

        # Get time array for x-axis
        time_array = np.array(self.time_buffer)
        
        # Calculate x-axis limits for sliding window
        current_time = time_array[-1]
        window_start = max(0, current_time - 4.0)  # Show last 4 seconds
        
        if self.selected_channel == -1:
            # Plotar todos os canais
            for i, curve in enumerate(self.plot_curves):
                curve.setData(time_array, list(self.data_buffers[i]))
                curve.show()
        else:
            # Plotar apenas o canal selecionado
            for i, curve in enumerate(self.plot_curves):
                if i == self.selected_channel:
                    curve.setData(time_array, list(self.data_buffers[i]))
                    curve.show()
                else:
                    curve.hide()
        
        # Set x-axis range for sliding window effect
        self.plot_widget.setXRange(window_start, current_time)
        
        # Aplicar escala no eixo Y
        if self.scale_combo.currentText() != "Auto":
            self.plot_widget.setYRange(-self.scale, self.scale)

    def _channel_changed(self, index):
        """Callback quando o canal selecionado muda"""
        self.selected_channel = index - 1  # -1 = todos
        
    def _scale_changed(self, text):
        """Callback quando a escala muda"""
        if text == "Auto":
            self.plot_widget.enableAutoRange()
        else:
            self.scale = int(text.strip("±µV"))
            self.plot_widget.setYRange(-self.scale, self.scale)




