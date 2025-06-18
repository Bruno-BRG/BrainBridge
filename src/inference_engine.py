"""
Real-time inference module for BCI system
Handles loading pre-trained models and making real-time predictions
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import queue
import threading
import time
from scipy import signal
from collections import deque

logger = logging.getLogger(__name__)

# Import model from braindecode if available
try:
    from braindecode.models import EEGInceptionERP
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    logger.warning("Braindecode not available, using fallback CNN")

class EEGInceptionERPModel(nn.Module):
    """Wrapper para EEGInceptionERP com fallback para CNN simples - mesma arquitetura do notebook"""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, sfreq: float = 125.0, **kwargs):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.is_trained = False

        if BRAINDECODE_AVAILABLE:
            try:
                # Import here to avoid scoping issues
                from braindecode.models import EEGInceptionERP
                self.model = EEGInceptionERP(
                    n_chans=n_chans,
                    n_outputs=n_outputs,
                    n_times=n_times,
                    sfreq=sfreq
                )
                self.model_type = "EEGInceptionERP"
                logger.info(f"✅ Usando EEGInceptionERP: {n_chans} canais, {n_times} pontos temporais")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao criar EEGInceptionERP: {e}")
                self.model = self._create_fallback_cnn(n_chans, n_outputs, n_times)
                self.model_type = "FallbackCNN"
        else:
            self.model = self._create_fallback_cnn(n_chans, n_outputs, n_times)
            self.model_type = "FallbackCNN"

    def _create_fallback_cnn(self, n_chans: int, n_outputs: int, n_times: int):
        """Criar CNN simples como fallback"""
        logger.info(f"🔧 Criando CNN fallback: {n_chans} canais, {n_times} pontos temporais")
        return nn.Sequential(
            # Convolução temporal
            nn.Conv1d(n_chans, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Redução de dimensionalidade
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            # Convolução final
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Classificador
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        if self.model_type == "EEGInceptionERP":
            return self.model(x)
        else:
            # Para o CNN fallback, precisamos transpor de (batch, channels, time) para (batch, channels, time)
            return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        """Carregar state dict do modelo"""
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self):
        """Retornar state dict do modelo"""
        return self.model.state_dict()

    def to(self, device):
        """Mover modelo para device"""
        self.model = self.model.to(device)
        return self

    def eval(self):
        """Colocar modelo em modo de avaliação"""
        self.model.eval()
        return self

class EEGInferenceEngine:
    """Real-time EEG inference engine"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.normalizer_stats = None
        self.is_loaded = False
        
        # Model parameters (will be loaded from checkpoint)
        self.n_chans = 16
        self.n_times = 400  # Updated default to match the trained models
        self.sample_rate = 125.0
        self.n_outputs = 2  # Left vs Right hand
        self.drop_prob = 0.5
        self.n_filters = 8
        
        # Sliding window for real-time inference
        self.window_size = self.n_times
        self.data_buffer = deque(maxlen=self.window_size * 2)  # Double buffer
          # Class labels with uncertainty threshold
        self.class_labels = ["Left Hand", "Right Hand", "Uncertain/Rest"]
        self.confidence_threshold = 0.6  # Threshold para considerar predição confiável
        
        # Debug counters
        self.samples_received = 0
        self.predictions_made = 0
        self.last_log_time = time.time()
        
        logger.info(f"🧠 Inicializando InferenceEngine: buffer_size={self.window_size}, device={device}")
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model parameters from checkpoint
            if 'constructor_args' in checkpoint:
                params = checkpoint['constructor_args']
                self.n_chans = params.get('n_chans', 16)
                self.n_times = params.get('n_times', 400)
                self.sample_rate = params.get('sfreq', 125.0)
                self.n_outputs = params.get('n_outputs', 2)
                self.drop_prob = params.get('drop_prob', 0.5)
                self.n_filters = params.get('n_filters', 8)
                logger.info(f"Loaded model parameters: n_chans={self.n_chans}, n_times={self.n_times}, sfreq={self.sample_rate}")
            elif 'model_params' in checkpoint:
                params = checkpoint['model_params']
                self.n_chans = params.get('n_chans', 16)
                self.n_times = params.get('n_times', 400)
                self.sample_rate = params.get('sfreq', 125.0)
                self.n_outputs = params.get('n_outputs', 2)
                self.drop_prob = params.get('drop_prob', 0.5)
                self.n_filters = params.get('n_filters', 8)
                logger.info(f"Loaded model parameters: n_chans={self.n_chans}, n_times={self.n_times}, sfreq={self.sample_rate}")
            
            # Create model
            self.model = EEGInceptionERPModel(
                n_chans=self.n_chans,
                n_outputs=self.n_outputs,
                n_times=self.n_times,
                sfreq=self.sample_rate
            )
            logger.info(f"Created model with {self.n_chans} channels, {self.n_times} time points")
            
            # Load state dict with compatibility handling
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume checkpoint is the state dict itself
                state_dict = checkpoint

            # Try to load state dict, handling size mismatches
            try:
                self.model.load_state_dict(state_dict, strict=True)
                logger.info("Model loaded with strict=True")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    logger.warning(f"Size mismatch detected: {e}")
                    logger.info("Attempting to load with strict=False")
                    
                    # Load with strict=False to ignore mismatched layers
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                else:
                    raise e
            
            self.model.to(self.device)
            self.model.eval()
              # Load normalization statistics
            if 'normalization_stats' in checkpoint:
                self.normalizer_stats = checkpoint['normalization_stats']
                logger.info("Loaded normalization statistics")
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        """
        Preprocess EEG data for inference using EXACTLY the same pipeline as the training notebook
        Implements ImprovedEEGNormalizer with method='robust_zscore', scope='channel'
        """
        # Ensure data is 2D (channels x time)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Garantir formato (channels, samples)
        if data.shape[0] != self.n_chans:
            data = data.T
        
        logger.debug(f"📊 INFERENCE preprocessing - shape inicial: {data.shape}")
        
        # === PREPROCESSING EXATO DO NOTEBOOK ===
        # Criar formato 3D (trials, channels, time) como no notebook
        data_3d = data[np.newaxis, :, :]  # (1, channels, time)
        
        # === HANDLE OUTLIERS (método robust) ===
        Q1 = np.percentile(data_3d, 25, axis=(0, 2), keepdims=True)  # (1, channels, 1)
        Q3 = np.percentile(data_3d, 75, axis=(0, 2), keepdims=True)  # (1, channels, 1) 
        IQR = Q3 - Q1
        outlier_threshold = 3.0
        lower = Q1 - outlier_threshold * IQR
        upper = Q3 + outlier_threshold * IQR
        
        # Clip outliers
        data_clipped = np.clip(data_3d, lower, upper)
        
        # === ROBUST Z-SCORE por canal (scope='channel') ===
        median = np.median(data_clipped, axis=(0, 2), keepdims=True)  # (1, channels, 1)
        q75, q25 = np.percentile(data_clipped, [75, 25], axis=(0, 2))
        iqr = (q75 - q25)[None, :, None] + 1e-8  # (1, channels, 1)
        
        # Normalize
        normalized_3d = (data_clipped - median) / iqr
        
        # Converter para formato final (channels, time)
        normalized_data = normalized_3d.squeeze(0)  # (channels, time)
        
        logger.debug(f"📊 INFERENCE preprocessing - shape final: {normalized_data.shape}")
        logger.debug(f"📊 INFERENCE preprocessing - range: [{np.min(normalized_data):.3f}, {np.max(normalized_data):.3f}]")
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(normalized_data).unsqueeze(0)  # (1, channels, time)
        return tensor.to(self.device)
    
    def predict(self, data: np.ndarray) -> Tuple[int, float, List[float]]:
        """Make prediction on EEG data
        
        Returns:
            predicted_class: 0 or 1
            confidence: confidence score (0-1)
            probabilities: list of class probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess data
        tensor = self.preprocess_data(data)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            probabilities = probabilities.squeeze().cpu().numpy().tolist()
        
        return predicted_class, confidence, probabilities
    
    def add_sample(self, sample: List[float]):
        """Add new sample to buffer for sliding window inference"""
        self.samples_received += 1
        current_time = time.time()
        
        if len(sample) != self.n_chans:
            logger.warning(f"⚠️ Canal mismatch: esperado {self.n_chans}, recebido {len(sample)}")
            return
        
        self.data_buffer.append(sample)
        
        # Log detalhado ocasionalmente
        if self.samples_received % 50 == 0 or current_time - self.last_log_time > 5.0:
            sample_range = f"[{min(sample):.3f}, {max(sample):.3f}]"
            logger.info(f"🧠 InferenceEngine recebeu sample #{self.samples_received}: "
                       f"buffer={len(self.data_buffer)}/{self.window_size}, range={sample_range}")
            self.last_log_time = current_time
            
        # Verifica se pode fazer predição
        if self.can_predict() and self.samples_received % 25 == 0:  # Predições mais frequentes
            logger.info(f"🎯 Buffer cheio - fazendo predição (sample #{self.samples_received})")
    
    def can_predict(self) -> bool:
        """Check if we have enough data for prediction"""
        return len(self.data_buffer) >= self.window_size
    def predict_from_buffer(self) -> Optional[Tuple[int, float, List[float]]]:
        """Make prediction using current buffer"""
        if not self.can_predict():
            logger.debug(f"❌ Não pode predizer - buffer insuficiente: {len(self.data_buffer)}/{self.window_size}")
            return None
        
        self.predictions_made += 1
        
        # Get last window_size samples
        window_data = list(self.data_buffer)[-self.window_size:]
        data = np.array(window_data).T  # (channels, time)
        
        logger.info(f"🔮 Fazendo predição #{self.predictions_made}: shape={data.shape}")
        
        try:
            result = self.predict(data)
            class_idx, confidence, probabilities = result
            class_name = self.get_class_label(class_idx)
            
            logger.info(f"✅ Predição #{self.predictions_made}: {class_name} (conf={confidence:.3f}, "
                       f"probs={[f'{p:.3f}' for p in probabilities]})")
            
            return result
        except Exception as e:
            logger.error(f"❌ Erro na predição #{self.predictions_made}: {e}")
            return None
    
    def get_class_label(self, class_idx: int) -> str:
        """Get human-readable class label"""
        if 0 <= class_idx < len(self.class_labels):
            return self.class_labels[class_idx]
        return f"Unknown_{class_idx}"
    
    def reset_buffer(self):
        """Reset the data buffer"""
        self.data_buffer.clear()
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.data_buffer)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'n_channels': self.n_chans,
            'n_times': self.n_times,
            'sample_rate': self.sample_rate,
            'n_outputs': self.n_outputs,
            'class_labels': self.class_labels,
            'is_loaded': self.is_loaded,
            'device': str(self.device)
        }

class RealTimeInferenceManager:
    """Manager for real-time inference with multiple models"""
    
    def __init__(self):
        logger.info("🔧 Inicializando RealTimeInferenceManager...")
        self.engines: Dict[str, EEGInferenceEngine] = {}
        self.active_engine: Optional[str] = None
        self.prediction_callback = None
        self.is_running = False
        self.inference_thread = None
        self.data_queue = queue.Queue()
        logger.info("✅ RealTimeInferenceManager inicializado com sucesso")
        
    def load_model(self, model_name: str, model_path: str, device: str = "cpu") -> bool:
        """Load a model"""
        try:
            engine = EEGInferenceEngine(model_path, device)
            if engine.is_loaded:
                self.engines[model_name] = engine
                logger.info(f"Model '{model_name}' loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def set_active_model(self, model_name: str) -> bool:
        """Set active model for inference"""
        if model_name in self.engines:
            self.active_engine = model_name
            logger.info(f"Active model set to '{model_name}'")
            return True
        else:
            logger.error(f"Model '{model_name}' not found")
            return False
    
    def start_inference(self, prediction_callback=None):
        """Start real-time inference"""
        if not self.active_engine:
            raise ValueError("No active model set")
        
        self.prediction_callback = prediction_callback
        self.is_running = True
        
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        logger.info("Real-time inference started")
    
    def stop_inference(self):
        """Stop real-time inference"""
        self.is_running = False
        
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        logger.info("Real-time inference stopped")
    
    def add_sample(self, sample: List[float]):
        """Add new sample for inference"""
        if self.is_running and self.active_engine:
            self.data_queue.put(sample)
            
            # Log ocasionalmente para verificar se amostras estão chegando ao manager
            if self.data_queue.qsize() % 50 == 0:
                logger.info(f"🔄 Manager recebeu sample - queue_size={self.data_queue.qsize()}, "
                           f"active_engine={self.active_engine}")
        else:
            if not self.is_running:
                logger.warning(f"⚠️ Manager não está rodando - sample ignorado")
            if not self.active_engine:
                logger.warning(f"⚠️ Nenhum engine ativo - sample ignorado")
    
    def _inference_loop(self):
        """Main inference loop"""
        engine = self.engines[self.active_engine]
        sample_count = 0
        prediction_count = 0
        last_log_time = time.time()
        
        logger.info(f"🚀 Iniciando loop de inferência com engine: {self.active_engine}")
        
        while self.is_running:
            try:
                # Get sample from queue
                sample = self.data_queue.get(timeout=1.0)
                sample_count += 1
                
                # Add to engine buffer
                engine.add_sample(sample)
                
                # Log de debug ocasional
                current_time = time.time()
                if sample_count % 50 == 0 or current_time - last_log_time > 5.0:
                    buffer_info = f"{len(engine.data_buffer)}/{engine.window_size}"
                    logger.info(f"🔄 Loop processou {sample_count} samples, buffer: {buffer_info}")
                    last_log_time = current_time
                
                # Make prediction if possible
                if engine.can_predict():
                    result = engine.predict_from_buffer()
                    
                    if result and self.prediction_callback:
                        prediction_count += 1
                        predicted_class, confidence, probabilities = result
                        
                        prediction_data = {
                            'predicted_class': predicted_class,
                            'class_label': engine.get_class_label(predicted_class),
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"🎯 Predição #{prediction_count}: {prediction_data['class_label']} "
                                   f"(conf={confidence:.3f}) - enviando para GUI...")
                        
                        try:
                            self.prediction_callback(prediction_data)
                            logger.info(f"✅ Callback da predição #{prediction_count} executado com sucesso")
                        except Exception as callback_error:
                            logger.error(f"❌ Erro no callback da predição #{prediction_count}: {callback_error}")
                    elif not self.prediction_callback:
                        logger.warning(f"⚠️ Predição feita mas nenhum callback configurado")
                
            except queue.Empty:
                # Log ocasional quando não há dados na queue
                if sample_count % 500 == 0:
                    logger.debug(f"🕐 Queue vazia no loop de inferência (timeout)")
                continue
            except Exception as e:
                logger.error(f"❌ Erro no loop de inferência: {e}")
                time.sleep(0.1)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.engines.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        if model_name in self.engines:
            return self.engines[model_name].get_model_info()
        return None
