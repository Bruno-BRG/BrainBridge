import os
from pathlib import Path

# Pasta raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

FOLDERS = {
    'recordings': PROJECT_ROOT / 'data' / 'recordings',
    'database': PROJECT_ROOT / 'data' / 'database', 
    'models': PROJECT_ROOT / 'models',
    'docs': PROJECT_ROOT / 'docs',
    'tests': PROJECT_ROOT / 'tests',
    'legacy': PROJECT_ROOT / 'legacy'
}

def ensure_folders_exist():
    for folder in FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = FOLDERS['database'] / 'bci_patients.db'
RECORDINGS_PATH = FOLDERS['recordings']

def get_database_path():
    return DATABASE_PATH

def get_recording_path(filename: str):
    return RECORDINGS_PATH / filename
