import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                           QComboBox, QLabel)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class EEGPlotWidget(QWidget):
    """Widget para plotar dados EEG em tempo real"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_plot()
        
        # Buffer para dados
        self.data_buffer = deque(maxlen=1000)  # 8 segundos a 125 Hz
        self.time_buffer = deque(maxlen=1000)
        self.current_time = 0
        
        # Timer para atualizar plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 FPS
        
    def setup_ui(self):
        """Configura a interface do widget"""
        layout = QVBoxLayout()
        
        # Controles
        controls_layout = QHBoxLayout()
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([f"Canal {i}" for i in range(16)])
        self.channel_combo.addItem("Todos os Canais")
        self.channel_combo.currentTextChanged.connect(self.change_channel)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Auto", "±50µV", "±100µV", "±200µV", "±500µV"])
        self.scale_combo.currentTextChanged.connect(self.change_scale)
        
        controls_layout.addWidget(QLabel("Canal:"))
        controls_layout.addWidget(self.channel_combo)
        controls_layout.addWidget(QLabel("Escala:"))
        controls_layout.addWidget(self.scale_combo)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Área do plot
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def setup_plot(self):
        """Configura o plot inicial"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 8)  # 8 segundos
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('Tempo (s)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title('Dados EEG em Tempo Real')
        self.ax.grid(True, alpha=0.3)
        
        # Linhas para cada canal
        self.lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 16))
        
        for i in range(16):
            line, = self.ax.plot([], [], color=colors[i], linewidth=0.8, 
                               label=f'Canal {i}', alpha=0.7)
            self.lines.append(line)
            
        self.canvas.draw()
        
    def add_data(self, eeg_data: np.ndarray):
        """Adiciona novos dados EEG"""
        if len(eeg_data) == 16:  # 16 canais
            self.data_buffer.append(eeg_data)
            self.time_buffer.append(self.current_time)
            self.current_time += 1/125  # 125 Hz
            
    def update_plot(self):
        """Atualiza o plot com novos dados"""
        if len(self.data_buffer) == 0:
            return
            
        # Converter buffer para arrays numpy
        times = np.array(self.time_buffer)
        data = np.array(self.data_buffer)
        
        if len(times) < 2:
            return
            
        # Atualizar janela de tempo
        current_time = times[-1]
        window_start = max(0, current_time - 8)
        
        # Filtrar dados da janela
        mask = times >= window_start
        windowed_times = times[mask] - window_start
        windowed_data = data[mask]
        
        # Atualizar cada linha
        selected_channel = self.channel_combo.currentText()
        
        if selected_channel == "Todos os Canais":
            # Mostrar todos os canais com offset
            for i in range(16):
                if len(windowed_data) > 0:
                    y_data = windowed_data[:, i] + i * 100  # Offset vertical
                    self.lines[i].set_data(windowed_times, y_data)
                    self.lines[i].set_visible(True)
            
            self.ax.set_ylim(-100, 1600)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Mostrar apenas um canal
            channel_idx = int(selected_channel.split()[1])
            
            for i in range(16):
                if i == channel_idx and len(windowed_data) > 0:
                    self.lines[i].set_data(windowed_times, windowed_data[:, i])
                    self.lines[i].set_visible(True)
                else:
                    self.lines[i].set_visible(False)
            
            # Ajustar escala
            scale_text = self.scale_combo.currentText()
            if scale_text == "Auto":
                if len(windowed_data) > 0:
                    y_data = windowed_data[:, channel_idx]
                    if len(y_data) > 0:
                        y_min, y_max = np.min(y_data), np.max(y_data)
                        margin = (y_max - y_min) * 0.1
                        self.ax.set_ylim(y_min - margin, y_max + margin)
            else:
                scale_val = int(scale_text.replace("±", "").replace("µV", ""))
                self.ax.set_ylim(-scale_val, scale_val)
            
            self.ax.legend().set_visible(False)
        
        self.ax.set_xlim(0, 8)
        self.canvas.draw()
        
    def change_channel(self):
        """Callback para mudança de canal"""
        self.setup_plot()
        
    def change_scale(self):
        """Callback para mudança de escala"""
        pass