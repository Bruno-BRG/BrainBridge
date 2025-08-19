"""
Interfaces de Serviços - Seguindo ISP (Interface Segregation Principle)

Estas interfaces definem contratos específicos para serviços,
permitindo implementações flexíveis e testáveis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import numpy as np


class IDataStreamingService(ABC):
    """Interface para serviço de streaming de dados"""
    
    @abstractmethod
    def start_streaming(self, config: Dict[str, Any]) -> bool:
        """Inicia o streaming de dados"""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> bool:
        """Para o streaming de dados"""
        pass
    
    @abstractmethod
    def is_streaming(self) -> bool:
        """Verifica se está fazendo streaming"""
        pass
    
    @abstractmethod
    def get_latest_data(self) -> Optional[np.ndarray]:
        """Retorna os dados mais recentes"""
        pass
    
    @abstractmethod
    def subscribe_to_data(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Subscreve a callback para novos dados"""
        pass


class IDataProcessingService(ABC):
    """Interface para serviço de processamento de dados"""
    
    @abstractmethod
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Pré-processa dados EEG"""
        pass
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extrai características dos dados"""
        pass
    
    @abstractmethod
    def filter_data(self, data: np.ndarray, filter_params: Dict[str, Any]) -> np.ndarray:
        """Aplica filtros aos dados"""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Valida a qualidade dos dados"""
        pass


class IModelService(ABC):
    """Interface para serviço de modelos de ML"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Carrega modelo treinado"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Faz predição com o modelo"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Verifica se modelo está carregado"""
        pass


class IValidationService(ABC):
    """Interface para serviço de validação"""
    
    @abstractmethod
    def validate_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valida dados do paciente"""
        pass
    
    @abstractmethod
    def validate_session_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valida configuração da sessão"""
        pass
