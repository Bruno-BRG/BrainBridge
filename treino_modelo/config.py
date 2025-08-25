"""
Configurações centralizadas do sistema BrainBridge
"""

import os
from pathlib import Path

# Configurações de Dados
DATA_CONFIG = {
    'sample_rate': 125,  # Hz
    'channels': 16,      # Canais EEG (0-15)
    'window_size': 250,  # 2 segundos a 125Hz
    'overlap': 125,      # 50% sobreposição
    'classes': ['left', 'right'],
    'markers': {
        'left_start': 'T1',
        'right_start': 'T2', 
        'end': 'T0'
    }
}

# Configurações de Modelo
MODEL_CONFIG = {
    'architecture': 'CNN_1D',
    'conv_filters': [64, 64, 128],
    'conv_kernels': [3, 3, 3],
    'dense_units': 512,
    'dropout_rates': [0.25, 0.25, 0.5],
    'activation': 'relu',
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

# Configurações de Treinamento
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'test_size': 0.2,
    'random_state': 42
}

# Configurações de Predição
PREDICTION_CONFIG = {
    'min_confidence': 0.65,
    'min_consensus': 0.7,
    'prediction_interval': 25,  # amostras (200ms a 125Hz)
    'history_length': 10,       # últimas N predições
    'buffer_size': 250          # tamanho do buffer circular
}

# Diretórios
BASE_DIR = Path(__file__).parent
DIRECTORIES = {
    'base': BASE_DIR,
    'data': Path("c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"),
    'models': BASE_DIR / "models",
    'logs': BASE_DIR / "logs",
    'results': BASE_DIR / "results",
    'temp': BASE_DIR / "temp"
}

# Configurações de CLI
CLI_CONFIG = {
    'clear_screen': True,
    'show_progress': True,
    'verbose': True,
    'colors': True,
    'banner': True
}

# Configurações de Log
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_log': True,
    'console_log': True,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Configurações de Performance
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'memory_growth': True,
    'mixed_precision': False,
    'parallel_processing': True,
    'max_workers': 4
}

def create_directories():
    """Cria os diretórios necessários se não existirem"""
    for name, path in DIRECTORIES.items():
        if name != 'data':  # Não cria o diretório de dados automaticamente
            path.mkdir(parents=True, exist_ok=True)

def get_model_save_path(prefix: str = "", suffix: str = ""):
    """
    Gera caminho para salvar modelo
    
    Args:
        prefix: Prefixo do nome do arquivo
        suffix: Sufixo do nome do arquivo
    
    Returns:
        Path: Caminho completo do arquivo
    """
    import time
    timestamp = int(time.time())
    
    if prefix:
        prefix = f"{prefix}_"
    if suffix:
        suffix = f"_{suffix}"
    
    filename = f"{prefix}eeg_motor_imagery{suffix}_{timestamp}.keras"
    return DIRECTORIES['models'] / filename

def validate_data_directory():
    """Valida se o diretório de dados existe e contém dados"""
    data_dir = DIRECTORIES['data']
    
    if not data_dir.exists():
        return False, f"Diretório de dados não encontrado: {data_dir}"
    
    # Verifica se há sujeitos
    subjects = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('S')]
    
    if not subjects:
        return False, f"Nenhum sujeito encontrado em: {data_dir}"
    
    # Verifica se há arquivos CSV
    csv_count = 0
    for subject_dir in subjects:
        csv_files = list(subject_dir.glob("*.csv"))
        csv_count += len(csv_files)
    
    if csv_count == 0:
        return False, "Nenhum arquivo CSV encontrado nos diretórios de sujeitos"
    
    return True, f"Diretório válido: {len(subjects)} sujeitos, {csv_count} arquivos CSV"

def get_system_info():
    """Retorna informações do sistema"""
    import platform
    import psutil
    
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        gpus = len(tf.config.list_physical_devices('GPU'))
    except ImportError:
        tf_version = "Não instalado"
        gpus = 0
    
    return {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf_version,
        'gpus_available': gpus,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2)
    }

# Inicialização
create_directories()
