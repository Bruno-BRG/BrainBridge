"""
Module:   data_normalizer
Purpose:  Provides data normalization tools for EEG signal processing
Author:   Bruno Rocha
Created:  2025-01-15
Notes:    Contains classes for different normalization strategies
          Used by both training and real-time inference
"""

import numpy as np
import os
import json
from typing import Dict, Optional, Tuple, Union, List


class UniversalEEGNormalizer:
    """Normalizador universal para dados EEG"""

    def __init__(self, method: str = 'zscore', mode: str = 'training', stats_path: Optional[str] = None):
        self.method = method
        self.mode = mode
        self.stats_path = stats_path
        self.global_stats = {}
        self.is_fitted = False

        # Carregar estatísticas pré-calculadas se caminho fornecido
        if stats_path and os.path.exists(stats_path):
            self.load_stats(stats_path)

    def _ensure_3d(self, data: np.ndarray) -> np.ndarray:
        """Garantir que dados estejam em 3D (n_samples, n_channels, n_timepoints)"""
        if len(data.shape) == 2:
            n_samples, n_features = data.shape
            if n_features % 16 == 0:  # Assumir 16 canais
                n_channels = 16
                n_timepoints = n_features // n_channels
                data = data.reshape(n_samples, n_channels, n_timepoints)
            else:
                data = data[:, np.newaxis, :]
        elif len(data.shape) == 1:
            data = data[np.newaxis, np.newaxis, :]
        return data

    def fit(self, data: np.ndarray):
        """Ajustar normalizador aos dados"""
        # Se já estiver ajustado e não estivermos no modo de treinamento, retorna
        if self.is_fitted and self.mode != 'training':
            return self
            
        data_3d = self._ensure_3d(data)

        if self.method == 'zscore':
            self.global_stats['mean'] = np.mean(data_3d, axis=(0, 2), keepdims=True)
            self.global_stats['std'] = np.std(data_3d, axis=(0, 2), keepdims=True)
            # Evitar divisão por zero
            self.global_stats['std'] = np.where(self.global_stats['std'] == 0, 1.0, self.global_stats['std'])
        elif self.method == 'robust_zscore':
            self.global_stats['median'] = np.median(data_3d, axis=(0, 2), keepdims=True)
            q75 = np.percentile(data_3d, 75, axis=(0, 2), keepdims=True)
            q25 = np.percentile(data_3d, 25, axis=(0, 2), keepdims=True)
            self.global_stats['iqr'] = q75 - q25
            # Evitar divisão por zero
            self.global_stats['iqr'] = np.where(self.global_stats['iqr'] == 0, 1.0, self.global_stats['iqr'])
        elif self.method == 'minmax':
            self.global_stats['min'] = np.min(data_3d, axis=(0, 2), keepdims=True)
            self.global_stats['max'] = np.max(data_3d, axis=(0, 2), keepdims=True)
            # Evitar divisão por zero
            range_value = self.global_stats['max'] - self.global_stats['min']
            self.global_stats['range'] = np.where(range_value == 0, 1.0, range_value)

        self.is_fitted = True
        
        # Salvar estatísticas se no modo de treinamento e caminho fornecido
        if self.mode == 'training' and self.stats_path:
            self.save_stats(self.stats_path)
            
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transformar dados"""
        if not self.is_fitted:
            raise ValueError("Normalizador deve ser ajustado antes da transformação")

        original_shape = data.shape
        original_dtype = data.dtype  # CRITICAL: Remember original dtype
        data_3d = self._ensure_3d(data)

        if self.method == 'zscore':
            normalized = (data_3d - self.global_stats['mean']) / self.global_stats['std']
        elif self.method == 'robust_zscore':
            normalized = (data_3d - self.global_stats['median']) / self.global_stats['iqr']
        elif self.method == 'minmax':
            normalized = (data_3d - self.global_stats['min']) / self.global_stats['range']

        # Restaurar forma original
        if len(original_shape) != len(normalized.shape):
            if len(original_shape) == 2:
                normalized = normalized.reshape(original_shape[0], -1)
            elif len(original_shape) == 1:
                normalized = normalized.flatten()
        
        # CRITICAL: Maintain original dtype (usually float32)
        return normalized.astype(original_dtype)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Ajustar e transformar em um passo"""
        return self.fit(data).transform(data)
        
    def save_stats(self, filepath: str) -> None:
        """Salvar estatísticas de normalização em arquivo"""
        # Converter valores do dicionário para formato serializável
        serializable_stats = {}
        for key, value in self.global_stats.items():
            serializable_stats[key] = value.tolist()
        
        # Adicionar metadados
        metadata = {
            'method': self.method,
            'mode': self.mode,
            'created': str(np.datetime64('now')),
            'format_version': '1.0'
        }
        
        # Combinar dados
        save_data = {
            'stats': serializable_stats,
            'metadata': metadata
        }
        
        # Garantir que o diretório exista
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Salvar para arquivo
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_stats(self, filepath: str) -> bool:
        """Carregar estatísticas de normalização de arquivo"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extrair estatísticas
            loaded_stats = data.get('stats', {})
            
            # Converter listas de volta para arrays numpy com dimensões adequadas
            for key, value in loaded_stats.items():
                if isinstance(value, list):
                    arr = np.array(value)
                    # Reshape para adicionar keepdims se necessário
                    if len(arr.shape) == 3:
                        self.global_stats[key] = arr
                    else:
                        self.global_stats[key] = arr.reshape(1, -1, 1)
            
            self.is_fitted = True
            return True
            
        except Exception as e:
            print(f"Erro ao carregar estatísticas de normalização: {e}")
            return False


class ImprovedEEGNormalizer:
    """Improved EEG normalizer with multiple strategies and validation"""

    def __init__(self, method: str = 'robust_zscore', scope: str = 'channel',
                 outlier_threshold: float = 3.0):
        """
        Args:
            method: 'robust_zscore', 'minmax', or 'raw_zscore'
            scope: 'channel', 'trial', or 'global'
            outlier_threshold: number of deviations to consider outlier
        """
        self.method = method
        self.scope = scope
        self.outlier_threshold = outlier_threshold
        self.stats: Dict = {}
        self.is_fitted = False

    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Detect and handle outliers using IQR or standard deviation"""
        if self.method == 'robust_zscore':
            Q1 = np.percentile(X, 25, axis=(0, 2), keepdims=True)
            Q3 = np.percentile(X, 75, axis=(0, 2), keepdims=True)
            IQR = Q3 - Q1
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
        else:
            mean = np.mean(X, axis=(0, 2), keepdims=True)
            std = np.std(X, axis=(0, 2), keepdims=True)
            lower = mean - self.outlier_threshold * std
            upper = mean + self.outlier_threshold * std

        # Clip extreme values
        return np.clip(X, lower, upper)

    def fit(self, X: np.ndarray) -> 'ImprovedEEGNormalizer':
        """Ajusta o normalizador aos dados EXATAMENTE como no notebook"""
        
        # Garantir formato 3D
        if len(X.shape) == 2:
            if X.shape[1] % 16 == 0:  # Assumir 16 canais
                n_channels = 16
                X = X.reshape(X.shape[0], n_channels, -1)
            else:
                X = X[:, np.newaxis, :]

        # CRITICAL: Handle outliers ANTES da normalização (como no notebook)
        X = self._handle_outliers(X)

        if self.scope == 'channel':
            if self.method == 'robust_zscore':
                # CRITICAL: Usar exatamente as mesmas estatísticas do notebook
                self.stats['median'] = np.median(X, axis=(0, 2), keepdims=True)
                q75, q25 = np.percentile(X, [75, 25], axis=(0, 2))
                self.stats['iqr'] = (q75 - q25)[None, :, None] + 1e-8

            elif self.method == 'minmax':
                self.stats['min'] = X.min(axis=(0, 2), keepdims=True)
                self.stats['max'] = X.max(axis=(0, 2), keepdims=True)

            else:  # raw_zscore
                self.stats['mean'] = np.mean(X, axis=(0, 2), keepdims=True)
                self.stats['std'] = np.std(X, axis=(0, 2), keepdims=True) + 1e-8

        elif self.scope == 'trial':
            if self.method == 'robust_zscore':
                self.stats['median'] = np.median(X, axis=2, keepdims=True)
                q75, q25 = np.percentile(X, [75, 25], axis=2)
                self.stats['iqr'] = (q75 - q25)[:, :, None] + 1e-8

            elif self.method == 'minmax':
                self.stats['min'] = X.min(axis=2, keepdims=True)
                self.stats['max'] = X.max(axis=2, keepdims=True)

            else:  # raw_zscore
                self.stats['mean'] = np.mean(X, axis=2, keepdims=True)
                self.stats['std'] = np.std(X, axis=2, keepdims=True) + 1e-8

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma dados EXATAMENTE como no notebook"""
        if not self.is_fitted:
            raise ValueError("Normalize.fit() deve ser chamado antes de transform()")

        # CRITICAL: Manter dtype original
        original_dtype = X.dtype
        
        # Garantir formato 3D
        original_shape = X.shape
        if len(X.shape) == 2:
            if X.shape[1] % 16 == 0:
                n_channels = 16
                X = X.reshape(X.shape[0], n_channels, -1)
            else:
                X = X[:, np.newaxis, :]

        # Transformação
        if self.method == 'robust_zscore':
            X_norm = (X - self.stats['median']) / self.stats['iqr']
        elif self.method == 'minmax':
            X_norm = (X - self.stats['min']) / (self.stats['max'] - self.stats['min'] + 1e-8)
        else:  # raw_zscore
            X_norm = (X - self.stats['mean']) / self.stats['std']

        # Restaurar forma original
        if len(original_shape) == 2:
            X_norm = X_norm.reshape(original_shape)

        # CRITICAL: Manter dtype original
        return X_norm.astype(original_dtype)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data and transform in one step"""
        return self.fit(X).transform(X)

    def get_stats(self) -> Dict:
        """Return normalization statistics"""
        return self.stats.copy()


def validate_normalization(X: np.ndarray, normalized_X: np.ndarray) -> Tuple[bool, Dict]:
    """Validate normalization quality

    Args:
        X: Original data
        normalized_X: Normalized data

    Returns:
        (is_valid, stats): Boolean indicating if checks passed and statistics
    """
    stats = {}

    # 1. Check mean and std
    stats['mean'] = float(np.mean(normalized_X))
    stats['std'] = float(np.std(normalized_X))

    # 2. Calculate percentiles
    stats['percentiles'] = {
        '1%': float(np.percentile(normalized_X, 1)),
        '99%': float(np.percentile(normalized_X, 99))
    }

    # 3. Check preservation of relative order
    original_order = np.argsort(np.mean(X.reshape(X.shape[0], -1), axis=1))
    normalized_order = np.argsort(np.mean(normalized_X.reshape(normalized_X.shape[0], -1), axis=1))
    stats['order_correlation'] = float(np.corrcoef(original_order, normalized_order)[0, 1])

    # 4. Check for outliers
    z_scores = np.abs((normalized_X - stats['mean']) / stats['std'])
    stats['outliers_ratio'] = float(np.mean(z_scores > 3))

    # Validation criteria
    is_valid = (
        abs(stats['mean']) < 0.1 and            # Mean close to zero
        abs(stats['std'] - 1.0) < 0.5 and       # Std close to one
        stats['order_correlation'] > 0.7 and    # Order preserved
        stats['outliers_ratio'] < 0.01          # Few outliers
    )

    return is_valid, stats


def validate_normalization(normalized_data: np.ndarray) -> bool:
    # Simple validation: check that the data mean is near 0 and std is near 1
    if normalized_data.size == 0:
        return False
    mean_val = np.mean(normalized_data)
    std_val = np.std(normalized_data)
    return np.abs(mean_val) < 0.1 and np.abs(std_val - 1.0) < 0.1

def create_inference_normalizer(stats_path: Optional[str] = None) -> UniversalEEGNormalizer:
    # Use zscore method in inference mode
    return UniversalEEGNormalizer(method='zscore', mode='inference', stats_path=stats_path)

def create_universal_normalizer(method: str = 'zscore', mode: str = 'training') -> UniversalEEGNormalizer:
    # Returns a universal normalizer instance
    return UniversalEEGNormalizer(method=method, mode=mode)


def create_finetuning_normalizer(
    method: str = 'robust_zscore',
    stats_path: Optional[str] = None
) -> ImprovedEEGNormalizer:
    """
    Create normalizer specifically for fine-tuning.
    
    This is a convenience function that creates an improved normalizer for fine-tuning.
    
    Args:
        method: Normalization method ('robust_zscore', 'raw_zscore', 'minmax')
        stats_path: Optional path to statistics file
        
    Returns:
        Normalizer configured for fine-tuning
    """
    return ImprovedEEGNormalizer(method=method, scope='channel', outlier_threshold=3.0)


# Factory function for universal use
def create_universal_normalizer(method: str = 'zscore', mode: str = 'training', stats_path: Optional[str] = None) -> UniversalEEGNormalizer:
    return UniversalEEGNormalizer(method=method, mode=mode, stats_path=stats_path)


def create_finetuning_normalizer(
    method: str = 'robust_zscore',
    stats_path: Optional[str] = None
) -> ImprovedEEGNormalizer:
    """
    Create normalizer specifically for fine-tuning.
    
    This is a convenience function that creates an improved normalizer for fine-tuning.
    
    Args:
        method: Normalization method ('robust_zscore', 'raw_zscore', 'minmax')
        stats_path: Optional path to statistics file
        
    Returns:
        Normalizer configured for fine-tuning
    """
    return ImprovedEEGNormalizer(method=method, scope='channel', outlier_threshold=3.0)
