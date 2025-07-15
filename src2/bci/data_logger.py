import csv
from datetime import datetime
from pathlib import Path
import numpy as np

class CSVDataLogger:
    """Logger para gravar dados EEG em formato CSV"""
    def __init__(self, filepath: str, patient_id: str = None, task: str = None):
        self.filepath = Path(filepath)
        self.patient_id = patient_id
        self.task = task
        self.file = None
        self.writer = None
        self.samples_written = 0
        self.start_time = None

    def start_logging(self):
        """Inicia a gravação"""
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.start_time = datetime.now()
        
        # Cabeçalho OpenBCI
        self.writer.writerow(['%OpenBCI Raw EXG Data'])
        self.writer.writerow(['%Number of channels = 16'])
        self.writer.writerow(['%Sample Rate = 125 Hz'])
        if self.patient_id:
            self.writer.writerow([f'%Patient ID = {self.patient_id}'])
        if self.task:
            self.writer.writerow([f'%Task = {self.task}'])
        
        # Colunas
        columns = ['Sample Index'] + [f'EXG Channel {i}' for i in range(16)] + ['Annotations']
        self.writer.writerow(columns)

    def log_data(self, data: np.ndarray, marker: str = ''):
        """Grava uma amostra de dados"""
        if self.writer:
            row = [self.samples_written] + data.tolist() + [marker]
            self.writer.writerow(row)
            self.samples_written += 1

    def stop_logging(self):
        """Para a gravação"""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
