"""
Teste da interface BCI com marcadores T1, T2 e Baseline
"""

import sys
import os
from pathlib import Path

# Adicionar pasta src ao path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

try:
    from bci_interface import main
    print("✓ Interface BCI carregada com sucesso")
    print("✓ Funcionalidades disponíveis:")
    print("  - Cadastro de pacientes")
    print("  - Streaming de dados EEG")
    print("  - Gravação com marcadores T1, T2 e Baseline")
    print("  - Timer de 5 minutos para Baseline")
    print("  - Auto-inserção de T0 após 400 amostras")
    print("  - Estrutura de pastas organizada")
    print("\n🚀 Iniciando interface...")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"✗ Erro ao importar interface: {e}")
    print("Verifique se todas as dependências estão instaladas:")
    print("pip install PyQt5 numpy pandas matplotlib")
except Exception as e:
    print(f"✗ Erro inesperado: {e}")
