"""
Configuração de caminhos para o sistema BCI
"""

import os
from pathlib import Path

# Pasta raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Configuração de pastas
FOLDERS = {
    'recordings': PROJECT_ROOT / 'data' / 'recordings',
    'database': PROJECT_ROOT / 'data' / 'database', 
    'models': PROJECT_ROOT / 'models',
    'docs': PROJECT_ROOT / 'docs',
    'tests': PROJECT_ROOT / 'tests',
    'legacy': PROJECT_ROOT / 'legacy'
}

# Criar pastas se não existirem
for folder_name, folder_path in FOLDERS.items():
    folder_path.mkdir(parents=True, exist_ok=True)

# Caminhos específicos
DATABASE_PATH = FOLDERS['database'] / 'bci_patients.db'
RECORDINGS_PATH = FOLDERS['recordings']

def get_recording_path(filename):
    """Retorna o caminho completo para um arquivo de gravação"""
    return RECORDINGS_PATH / filename

def get_database_path():
    """Retorna o caminho do banco de dados"""
    return DATABASE_PATH

def ensure_folders_exist():
    """Garante que todas as pastas necessárias existem"""
    for folder_path in FOLDERS.values():
        folder_path.mkdir(parents=True, exist_ok=True)

# Executar ao importar
ensure_folders_exist()