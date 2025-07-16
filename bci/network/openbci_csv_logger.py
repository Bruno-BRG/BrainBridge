"""
OpenBCI CSV Logger - Formato compatível com OpenBCI
"""
import csv
import os
from datetime import datetime
from typing import List, Optional

class OpenBCICSVLogger:
    """Logger que gera CSVs no formato exato do OpenBCI"""
    
    def __init__(self, patient_id: str, task: str, patient_name: str = None, base_path: str = None):
        self.patient_id = patient_id
        self.task = task
        self.patient_name = patient_name or "Unknown"
        self.base_path = base_path or "data/recordings"
        self.sample_index = 0
        self.csv_file = None
        self.csv_writer = None
        self.filename = None
        self.patient_folder = None
        
        # Estado dos marcadores
        self.last_marker = None
        self.samples_since_marker = 0
        self.baseline_active = False
        self.baseline_start_time = None
        
        self._create_csv_file()
    
    def _create_csv_file(self):
        """Cria o arquivo CSV com headers no formato OpenBCI"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar nome da pasta do paciente (sanitizar nome para filesystem)
        safe_patient_name = self._sanitize_filename(self.patient_name)
        self.patient_folder = f"{self.patient_id}_{safe_patient_name}"
        
        # Caminho completo para a pasta do paciente
        patient_dir = os.path.join(self.base_path, self.patient_folder)
        
        # Garantir que o diretório do paciente existe
        os.makedirs(patient_dir, exist_ok=True)
        
        # Nome do arquivo
        self.filename = f"{self.patient_id}_{self.task}_{timestamp}.csv"
        filepath = os.path.join(patient_dir, self.filename)
        
        self.csv_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Headers no formato OpenBCI exato
        self.csv_writer.writerow(["%OpenBCI Raw EXG Data"])
        self.csv_writer.writerow(["%Number of channels = 16"])
        self.csv_writer.writerow(["%Sample Rate = 125 Hz"])
        self.csv_writer.writerow(["%Board = OpenBCI_GUI$BoardCytonSerialDaisy"])
        
        # Cabeçalho das colunas (exatamente como no arquivo de exemplo)
        header = [
            "Sample Index",
            "EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3",
            "EXG Channel 4", "EXG Channel 5", "EXG Channel 6", "EXG Channel 7",
            "EXG Channel 8", "EXG Channel 9", "EXG Channel 10", "EXG Channel 11",
            "EXG Channel 12", "EXG Channel 13", "EXG Channel 14", "EXG Channel 15",
            "Accel Channel 0", "Accel Channel 1", "Accel Channel 2",
            "Other", "Other.1", "Other.2", "Other.3", "Other.4", "Other.5", "Other.6",
            "Analog Channel 0", "Analog Channel 1", "Analog Channel 2",
            "Timestamp", "Other.7", "Timestamp (Formatted)", "Annotations"
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()
    
    def _sanitize_filename(self, name):
        """Sanitiza o nome para ser usado como nome de pasta/arquivo"""
        import re
        # Remover caracteres especiais e espaços
        safe_name = re.sub(r'[<>:"/\\|?*]', '', name)
        safe_name = safe_name.replace(' ', '_')
        # Limitar o tamanho
        return safe_name[:50] if len(safe_name) > 50 else safe_name
    
    def log_sample(self, eeg_data: List[float], marker: Optional[str] = None):
        """
        Registra uma amostra EEG no formato OpenBCI
        
        Args:
            eeg_data: Lista com 16 valores EEG (channels 0-15)
            marker: Marcador opcional (T0, T1, T2)
        """
        if len(eeg_data) != 16:
            raise ValueError(f"Esperado 16 canais EEG, recebido {len(eeg_data)}")
        
        # Determinar marcador atual
        current_marker = ""
        
        # Se um marcador foi explicitamente fornecido, usar ele
        if marker:
            current_marker = marker
            self.last_marker = marker
            self.samples_since_marker = 0
        else:
            # Verificar se devemos adicionar T0 automaticamente
            if self.last_marker in ['T1', 'T2']:
                self.samples_since_marker += 1
                # Após 400 amostras (~3.2s a 125Hz), adicionar T0 automaticamente
                if self.samples_since_marker >= 400:
                    current_marker = 'T0'
                    self.last_marker = 'T0'
                    self.samples_since_marker = 0
        
        # Preparar linha de dados no formato OpenBCI exato
        row = [self.sample_index]  # Sample Index
        
        # EEG Channels (0-15) - manter exatamente como vem
        row.extend(eeg_data)
        
        # Accel Channels (3 canais, zeros)
        row.extend([0, 0, 0])
        
        # Other columns (7 colunas, zeros)
        row.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Analog Channels (3 canais, zeros)
        row.extend([0, 0, 0])
        
        # Timestamp, Other.7, Timestamp (Formatted) (zeros)
        row.extend([0, 0, 0])
        
        # Annotations (marker ou vazio)
        row.append(current_marker)
        
        # Escrever linha
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        self.sample_index += 1
    
    def add_marker(self, marker: str):
        """Adiciona um marcador que será incluído na próxima amostra"""
        if not self.baseline_active:
            return marker
        return None
    
    def start_baseline(self):
        """Inicia o período de baseline (5 minutos)"""
        self.baseline_active = True
        self.baseline_start_time = datetime.now()
        
    def is_baseline_active(self):
        """Verifica se o baseline ainda está ativo"""
        if not self.baseline_active or not self.baseline_start_time:
            return False
            
        elapsed = (datetime.now() - self.baseline_start_time).total_seconds()
        if elapsed >= 300:  # 5 minutos
            self.baseline_active = False
            self.baseline_start_time = None
            return False
        return True
    
    def get_baseline_remaining(self):
        """Retorna o tempo restante do baseline em segundos"""
        if not self.is_baseline_active():
            return 0
        elapsed = (datetime.now() - self.baseline_start_time).total_seconds()
        return max(0, 300 - elapsed)
    
    def get_full_path(self):
        """Retorna o caminho completo do arquivo"""
        if self.patient_folder and self.filename:
            return os.path.join(self.base_path, self.patient_folder, self.filename)
        return os.path.join(self.base_path, self.filename) if self.filename else None
    
    def stop_logging(self):
        """Para o logging (método esperado pela interface)"""
        self.close()
        
    def close(self):
        """Fecha o arquivo CSV"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
    
    def __del__(self):
        self.close()