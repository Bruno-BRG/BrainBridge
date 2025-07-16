import csv
import threading
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

class SimpleCSVLogger:
    """Logger CSV simples para dados EEG com suporte a marcadores"""
    
    def __init__(self, filename):
        self.filename = filename
        self.is_logging = False
        self.data_buffer = []
        self.lock = threading.Lock()
        self.sample_count = 0
        self.pending_t0_marker = None
        self.t0_samples_remaining = 0
        
    def start_logging(self):
        """Inicia a gravação"""
        self.is_logging = True
        self.sample_count = 0
        # Criar cabeçalho
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Cabeçalho com marcador
            header = ['Timestamp'] + [f'EXG Channel {i}' for i in range(16)] + ['Marker']
            writer.writerow(header)
        print(f"Gravação iniciada: {self.filename}")
        
    def log_data(self, eeg_data, marker=None):
        """Adiciona dados ao log com marcador opcional"""
        if self.is_logging:
            with self.lock:
                timestamp = datetime.now().isoformat()
                
                # Verificar se deve inserir T0 automaticamente
                if self.t0_samples_remaining > 0:
                    self.t0_samples_remaining -= 1
                    if self.t0_samples_remaining == 0 and self.pending_t0_marker:
                        marker = self.pending_t0_marker
                        self.pending_t0_marker = None
                
                row = [timestamp] + list(eeg_data) + [marker if marker else '']
                self.data_buffer.append(row)
                self.sample_count += 1
                
                # Salvar a cada 10 amostras
                if len(self.data_buffer) >= 10:
                    self._flush_buffer()
    
    def add_marker(self, marker_type):
        """Adiciona um marcador e programa T0 se necessário"""
        if marker_type in ['T1', 'T2']:
            # Programar T0 para 400 amostras depois
            self.pending_t0_marker = 'T0'
            self.t0_samples_remaining = 400
        return marker_type
    
    def _flush_buffer(self):
        """Salva o buffer no arquivo"""
        if self.data_buffer:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_buffer)
            self.data_buffer.clear()
    
    def stop_logging(self):
        """Para a gravação"""
        self.is_logging = False
        with self.lock:
            if self.data_buffer:
                self._flush_buffer()
        print(f"Gravação finalizada: {self.filename}")
