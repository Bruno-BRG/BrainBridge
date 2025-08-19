"""
Modelo de Dados EEG - Seguindo SRP

Esta classe representa dados EEG e contém apenas as informações
e operações relacionadas aos dados de EEG.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class MarkerType(Enum):
    """Tipos de marcadores"""
    T1 = "T1"  # Movimento real
    T2 = "T2"  # Movimento imaginado
    T0 = "T0"  # Marcador automático
    BASELINE = "BASELINE"  # Baseline


@dataclass
class EEGChannel:
    """
    Representa um canal EEG
    
    Responsabilidade única: Dados de um canal específico
    """
    channel_id: int
    name: str
    data: np.ndarray
    sampling_rate: float
    unit: str = "µV"
    
    def __post_init__(self):
        """Validação após inicialização"""
        self.validate()
    
    def validate(self) -> None:
        """Valida os dados do canal"""
        if self.channel_id < 0:
            raise ValueError("ID do canal deve ser não-negativo")
        
        if not self.name.strip():
            raise ValueError("Nome do canal é obrigatório")
        
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Dados devem ser um numpy array")
        
        if self.sampling_rate <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
    
    def get_statistics(self) -> Dict[str, float]:
        """Retorna estatísticas do canal"""
        return {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'samples': len(self.data)
        }


@dataclass
class EEGMarker:
    """
    Representa um marcador nos dados EEG
    
    Responsabilidade única: Informações de um marcador
    """
    marker_type: MarkerType
    timestamp: datetime
    sample_index: int
    notes: Optional[str] = None
    
    def validate(self) -> None:
        """Valida os dados do marcador"""
        if not isinstance(self.marker_type, MarkerType):
            raise ValueError("Tipo de marcador deve ser válido")
        
        if self.sample_index < 0:
            raise ValueError("Índice da amostra deve ser não-negativo")
    
    def to_dict(self) -> dict:
        """Converte marcador para dicionário"""
        return {
            'type': self.marker_type.value,
            'timestamp': self.timestamp.isoformat(),
            'sample_index': self.sample_index,
            'notes': self.notes
        }


@dataclass
class EEGData:
    """
    Modelo de dados EEG seguindo SRP
    
    Responsabilidade única: Representar e validar dados EEG completos
    """
    channels: List[EEGChannel]
    markers: List[EEGMarker]
    sampling_rate: float
    start_timestamp: datetime
    session_id: Optional[int] = None
    accelerometer_data: Optional[np.ndarray] = None  # 3 canais (X, Y, Z)
    
    def __post_init__(self):
        """Validação após inicialização"""
        self.validate()
    
    def validate(self) -> None:
        """Valida os dados EEG"""
        if not self.channels:
            raise ValueError("Deve haver pelo menos um canal")
        
        if self.sampling_rate <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        
        # Verificar consistência entre canais
        if len(set(len(ch.data) for ch in self.channels)) > 1:
            raise ValueError("Todos os canais devem ter o mesmo número de amostras")
        
        # Verificar marcadores
        for marker in self.markers:
            marker.validate()
            if marker.sample_index >= self.get_sample_count():
                raise ValueError(f"Marcador em índice {marker.sample_index} está fora do range")
    
    def get_channel_count(self) -> int:
        """Retorna número de canais"""
        return len(self.channels)
    
    def get_sample_count(self) -> int:
        """Retorna número de amostras"""
        return len(self.channels[0].data) if self.channels else 0
    
    def get_duration_seconds(self) -> float:
        """Retorna duração em segundos"""
        return self.get_sample_count() / self.sampling_rate
    
    def get_channel_by_id(self, channel_id: int) -> Optional[EEGChannel]:
        """Busca canal por ID"""
        return next((ch for ch in self.channels if ch.channel_id == channel_id), None)
    
    def get_channel_by_name(self, name: str) -> Optional[EEGChannel]:
        """Busca canal por nome"""
        return next((ch for ch in self.channels if ch.name == name), None)
    
    def get_markers_by_type(self, marker_type: MarkerType) -> List[EEGMarker]:
        """Retorna marcadores de um tipo específico"""
        return [m for m in self.markers if m.marker_type == marker_type]
    
    def add_marker(self, marker_type: MarkerType, sample_index: int, 
                   timestamp: Optional[datetime] = None, notes: Optional[str] = None) -> None:
        """Adiciona um marcador"""
        if timestamp is None:
            # Calcular timestamp baseado no sample_index
            seconds_offset = sample_index / self.sampling_rate
            timestamp = self.start_timestamp + np.timedelta64(int(seconds_offset * 1000), 'ms')
        
        marker = EEGMarker(
            marker_type=marker_type,
            timestamp=timestamp,
            sample_index=sample_index,
            notes=notes
        )
        
        self.markers.append(marker)
        # Manter marcadores ordenados por sample_index
        self.markers.sort(key=lambda m: m.sample_index)
    
    def get_data_matrix(self) -> np.ndarray:
        """Retorna dados como matriz (channels x samples)"""
        return np.array([ch.data for ch in self.channels])
    
    def get_data_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Retorna janela de dados"""
        if start_sample < 0 or end_sample > self.get_sample_count():
            raise ValueError("Índices da janela estão fora do range")
        
        return np.array([ch.data[start_sample:end_sample] for ch in self.channels])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos dados"""
        data_matrix = self.get_data_matrix()
        return {
            'channels': self.get_channel_count(),
            'samples': self.get_sample_count(),
            'duration_seconds': self.get_duration_seconds(),
            'sampling_rate': self.sampling_rate,
            'marker_count': len(self.markers),
            'markers_by_type': {
                marker_type.value: len(self.get_markers_by_type(marker_type))
                for marker_type in MarkerType
            },
            'data_stats': {
                'mean': float(np.mean(data_matrix)),
                'std': float(np.std(data_matrix)),
                'min': float(np.min(data_matrix)),
                'max': float(np.max(data_matrix))
            }
        }
    
    def to_csv_format(self) -> List[List[str]]:
        """Converte dados para formato CSV compatível com OpenBCI"""
        header = ['Sample Index']
        
        # Adicionar colunas de canais EEG
        for i in range(self.get_channel_count()):
            header.append(f'EXG Channel {i}')
        
        # Adicionar colunas de acelerômetro
        header.extend(['Accel X', 'Accel Y', 'Accel Z'])
        
        # Adicionar outras colunas
        header.extend(['Other', 'Analog', 'Timestamp', 'Annotations'])
        
        rows = [header]
        
        # Criar mapa de marcadores por sample_index
        marker_map = {m.sample_index: m.marker_type.value for m in self.markers}
        
        # Converter dados
        sample_count = self.get_sample_count()
        for i in range(sample_count):
            row = [str(i)]  # Sample Index
            
            # Dados dos canais EEG
            for channel in self.channels:
                row.append(f"{channel.data[i]:.6f}")
            
            # Dados do acelerômetro (zeros se não disponível)
            if self.accelerometer_data is not None and i < len(self.accelerometer_data[0]):
                row.extend([f"{self.accelerometer_data[j][i]:.3f}" for j in range(3)])
            else:
                row.extend(['0.0', '0.0', '0.0'])
            
            # Other e Analog (sempre zeros)
            row.extend(['0', '0'])
            
            # Timestamp
            seconds_offset = i / self.sampling_rate
            timestamp = self.start_timestamp + np.timedelta64(int(seconds_offset * 1000), 'ms')
            row.append(f"{timestamp.timestamp():.3f}")
            
            # Annotations (marcador se existir)
            row.append(marker_map.get(i, ''))
            
            rows.append(row)
        
        return rows
    
    def __str__(self) -> str:
        """Representação string dos dados EEG"""
        return (f"EEGData(channels={self.get_channel_count()}, "
                f"samples={self.get_sample_count()}, "
                f"duration={self.get_duration_seconds():.2f}s)")
    
    def __repr__(self) -> str:
        """Representação detalhada dos dados EEG"""
        return (f"EEGData(session_id={self.session_id}, channels={self.get_channel_count()}, "
                f"samples={self.get_sample_count()}, sampling_rate={self.sampling_rate}, "
                f"markers={len(self.markers)})")
