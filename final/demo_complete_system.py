"""
Demonstração Completa do Sistema BCI
Script que inicia o simulador EEG e o sistema BCI para uma demonstração completa
"""

import sys
import time
import threading
import subprocess
from pathlib import Path
import argparse

# Adicionar pasta final ao path
sys.path.append(str(Path(__file__).parent))

from eeg_simulator import EEGDataSimulator

def run_bci_system(model_path: str, duration: float = 60.0):
    """Executar o sistema BCI em um processo separado"""
    try:
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / "complete_bci_system.py"),
            "--model", model_path,
            "--host", "localhost",
            "--port", "12345"
        ]
        
        print(f"🚀 Iniciando sistema BCI...")
        print(f"📂 Comando: {' '.join(cmd)}")
        
        # Executar como processo separado
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Aguardar um pouco para o sistema inicializar
        time.sleep(3)
        
        return process
        
    except Exception as e:
        print(f"❌ Erro ao iniciar sistema BCI: {e}")
        return None

def main():
    """Função principal da demonstração"""
    parser = argparse.ArgumentParser(description='Demonstração Completa do Sistema BCI')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo .pt')
    parser.add_argument('--duration', type=float, default=60.0, help='Duração da demo (segundos)')
    parser.add_argument('--sim-only', action='store_true', help='Apenas simulador, sem BCI')
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print(f"💡 Modelos disponíveis:")
        models_dir = Path.cwd() / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                print(f"   - {model_file}")
        return
    
    print(f"🎭 DEMONSTRAÇÃO COMPLETA DO SISTEMA BCI")
    print(f"=" * 60)
    print(f"📂 Modelo: {model_path}")
    print(f"⏱️ Duração: {args.duration} segundos")
    print(f"🎮 Simulador: Dados EEG sintéticos")
    print(f"🧠 Sistema BCI: Predições em tempo real")
    print(f"=" * 60)
    
    bci_process = None
    
    try:
        # Iniciar sistema BCI se não for apenas simulador
        if not args.sim_only:
            bci_process = run_bci_system(str(model_path), args.duration)
            if bci_process is None:
                print(f"❌ Falha ao iniciar sistema BCI")
                return
            
            print(f"✅ Sistema BCI iniciado (PID: {bci_process.pid})")
        
        # Aguardar um pouco para o sistema BCI inicializar
        if bci_process:
            print(f"⏳ Aguardando sistema BCI inicializar...")
            time.sleep(5)
        
        # Criar e iniciar simulador EEG
        print(f"🎮 Iniciando simulador EEG...")
        simulator = EEGDataSimulator(
            host='localhost',
            port=12345,
            sample_rate=125.0,
            n_channels=16
        )
        
        # Iniciar simulação
        print(f"📡 Começando transmissão de dados simulados...")
        simulator.start_simulation(duration=args.duration)
        
        # Aguardar conclusão
        if bci_process:
            print(f"⏳ Aguardando sistema BCI finalizar...")
            bci_process.wait(timeout=10)
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Demonstração interrompida pelo usuário")
    
    except Exception as e:
        print(f"❌ Erro durante demonstração: {e}")
    
    finally:
        # Limpar recursos
        if bci_process:
            try:
                bci_process.terminate()
                bci_process.wait(timeout=5)
                print(f"✅ Sistema BCI finalizado")
            except:
                bci_process.kill()
                print(f"🔪 Sistema BCI forçado a parar")
        
        print(f"🏁 Demonstração concluída!")

if __name__ == "__main__":
    main()
