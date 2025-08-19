"""
Modelo de Gravação - Seguindo SRP

Esta classe representa uma gravação de dados e contém apenas as informações
e operações relacionadas à gravação.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path
import os


@dataclass
class Recording:
    """
    Modelo de gravação seguindo SRP
    
    Responsabilidade única: Representar e gerenciar informações de gravação
    """
    session_id: int
    filename: str
    file_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    sample_count: Optional[int] = None
    recording_id: Optional[int] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validação após inicialização"""
        self.validate()
    
    def validate(self) -> None:
        """
        Valida os dados da gravação
        
        Raises:
            ValueError: Se algum dado for inválido
        """
        if not isinstance(self.session_id, int) or self.session_id <= 0:
            raise ValueError("ID da sessão deve ser um inteiro positivo")
        
        if not self.filename or not self.filename.strip():
            raise ValueError("Nome do arquivo é obrigatório")
        
        if not self.file_path or not self.file_path.strip():
            raise ValueError("Caminho do arquivo é obrigatório")
        
        if self.end_time and self.end_time <= self.start_time:
            raise ValueError("Horário de fim deve ser posterior ao de início")
        
        if self.file_size_bytes is not None and self.file_size_bytes < 0:
            raise ValueError("Tamanho do arquivo deve ser não-negativo")
        
        if self.sample_count is not None and self.sample_count < 0:
            raise ValueError("Contagem de amostras deve ser não-negativa")
    
    def complete_recording(self) -> None:
        """Completa a gravação, atualizando informações do arquivo"""
        self.end_time = datetime.now()
        
        # Atualizar informações do arquivo se existir
        if os.path.exists(self.file_path):
            self.file_size_bytes = os.path.getsize(self.file_path)
    
    def get_duration_seconds(self) -> Optional[int]:
        """Retorna duração da gravação em segundos"""
        if not self.end_time:
            return None
        
        return int((self.end_time - self.start_time).total_seconds())
    
    def get_duration_string(self) -> str:
        """Retorna duração formatada como string"""
        duration = self.get_duration_seconds()
        if duration is None:
            return "Em andamento..."
        
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_file_size_string(self) -> str:
        """Retorna tamanho do arquivo formatado"""
        if self.file_size_bytes is None:
            return "Desconhecido"
        
        # Converter para unidades legíveis
        if self.file_size_bytes < 1024:
            return f"{self.file_size_bytes} B"
        elif self.file_size_bytes < 1024 ** 2:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        elif self.file_size_bytes < 1024 ** 3:
            return f"{self.file_size_bytes / (1024 ** 2):.1f} MB"
        else:
            return f"{self.file_size_bytes / (1024 ** 3):.1f} GB"
    
    def file_exists(self) -> bool:
        """Verifica se o arquivo de gravação existe"""
        return os.path.exists(self.file_path)
    
    def is_active(self) -> bool:
        """Verifica se a gravação está ativa (não finalizada)"""
        return self.end_time is None
    
    def get_relative_path(self, base_path: str) -> str:
        """Retorna caminho relativo a partir de um diretório base"""
        try:
            return str(Path(self.file_path).relative_to(Path(base_path)))
        except ValueError:
            return self.file_path
    
    def to_dict(self) -> dict:
        """Converte a gravação para dicionário"""
        return {
            'id': self.recording_id,
            'session_id': self.session_id,
            'filename': self.filename,
            'file_path': self.file_path,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'file_size_bytes': self.file_size_bytes,
            'sample_count': self.sample_count,
            'notes': self.notes,
            'duration_seconds': self.get_duration_seconds(),
            'file_exists': self.file_exists()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Recording':
        """Cria uma gravação a partir de dicionário"""
        return cls(
            session_id=data['session_id'],
            filename=data['filename'],
            file_path=data['file_path'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            file_size_bytes=data.get('file_size_bytes'),
            sample_count=data.get('sample_count'),
            recording_id=data.get('id'),
            notes=data.get('notes')
        )
    
    def __str__(self) -> str:
        """Representação string da gravação"""
        status = "ativa" if self.is_active() else "finalizada"
        return f"Recording(id={self.recording_id}, session_id={self.session_id}, status={status})"
    
    def __repr__(self) -> str:
        """Representação detalhada da gravação"""
        return (f"Recording(recording_id={self.recording_id}, session_id={self.session_id}, "
                f"filename='{self.filename}', active={self.is_active()})")
