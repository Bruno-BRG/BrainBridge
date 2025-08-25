#!/usr/bin/env python3
"""
BrainBridge - CLI Simplificada para Treinamento EEG
"""

import os
import sys

def print_banner():
    print("\n" + "="*70)
    print(" 🧠 BRAINBRIDGE - SISTEMA DE TREINAMENTO EEG MOTOR IMAGERY 🧠")
    print("="*70)

def print_menu():
    print("\n📋 MENU PRINCIPAL:")
    print("  1. 📊 Analisar Dataset EEG")
    print("  2. 🤖 Treinar Modelo - Todos os Pacientes")
    print("  3. 👤 Treinar Modelo - Paciente Específico")
    print("  4. 📚 Treinar Modelo - Cross-Validation (50/50)")
    print("  5. 🔄 Treinar Modelo - Leave-One-Out")
    print("  6. 🧪 Testar Modelo Existente")
    print("  7. 📈 Comparar Modelos")
    print("  8. ⚙️  Configurações")
    print("  9. 🚪 Sair")

def get_user_choice():
    try:
        choice = input("\n🎯 Escolha uma opção (1-9): ")
        return choice
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Sistema finalizado pelo usuário.")
        sys.exit(0)

def run_analysis():
    print("\n📊 EXECUTANDO ANÁLISE DO DATASET...")
    try:
        exec(open('analyze_eeg_data.py').read())
    except Exception as e:
        print(f"❌ Erro: {e}")
    input("\nPressione ENTER para continuar...")

def run_full_training():
    print("\n🤖 EXECUTANDO TREINAMENTO COMPLETO...")
    try:
        exec(open('train_eeg_motor_imagery.py').read())
    except Exception as e:
        print(f"❌ Erro: {e}")
    input("\nPressione ENTER para continuar...")

def run_single_patient():
    print("\n👤 TREINAMENTO DE PACIENTE ESPECÍFICO")
    
    # Lista sujeitos disponíveis
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    if os.path.exists(data_dir):
        subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('S')]
        subjects.sort()
        
        print(f"\n📋 Sujeitos disponíveis ({len(subjects)}):")
        for i, subject in enumerate(subjects, 1):
            print(f"  {i:2d}. {subject}")
        
        try:
            choice = int(input(f"\nEscolha o sujeito (1-{len(subjects)}): "))
            if 1 <= choice <= len(subjects):
                selected = subjects[choice-1]
                print(f"\n🚀 Treinando modelo para {selected}...")
                # Aqui você implementaria o treinamento específico
                print(f"✅ Treinamento para {selected} seria executado aqui")
            else:
                print("❌ Escolha inválida!")
        except ValueError:
            print("❌ Entrada inválida!")
    else:
        print("❌ Diretório de dados não encontrado!")
    
    input("\nPressione ENTER para continuar...")

def run_cross_validation():
    print("\n📚 CROSS-VALIDATION 50/50")
    confirm = input("Confirmar treinamento cross-validation? (s/N): ")
    if confirm.lower() == 's':
        print("🚀 Cross-validation seria executado aqui...")
        # Implementar cross-validation
    input("\nPressione ENTER para continuar...")

def run_leave_one_out():
    print("\n🔄 LEAVE-ONE-OUT VALIDATION")
    confirm = input("Confirmar leave-one-out validation? (s/N): ")
    if confirm.lower() == 's':
        print("🚀 Leave-one-out seria executado aqui...")
        # Implementar leave-one-out
    input("\nPressione ENTER para continuar...")

def test_models():
    print("\n🧪 TESTE DE MODELOS")
    models_dir = "models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5', '.model'))]
        if models:
            print(f"\n📁 Modelos disponíveis ({len(models)}):")
            for i, model in enumerate(models, 1):
                print(f"  {i:2d}. {model}")
            
            try:
                choice = int(input(f"\nEscolha o modelo (1-{len(models)}): "))
                if 1 <= choice <= len(models):
                    selected = models[choice-1]
                    print(f"\n🧪 Testando modelo: {selected}")
                    try:
                        # Executa teste
                        exec(open('test_eeg_model.py').read())
                    except Exception as e:
                        print(f"❌ Erro no teste: {e}")
                else:
                    print("❌ Escolha inválida!")
            except ValueError:
                print("❌ Entrada inválida!")
        else:
            print("❌ Nenhum modelo encontrado!")
    else:
        print("❌ Diretório de modelos não encontrado!")
    
    input("\nPressione ENTER para continuar...")

def compare_models():
    print("\n📈 COMPARAÇÃO DE MODELOS")
    models_dir = "models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5', '.model'))]
        if len(models) >= 2:
            print(f"📊 Comparando {len(models)} modelos...")
            # Implementar comparação
            print("✅ Comparação seria executada aqui")
        else:
            print("❌ Pelo menos 2 modelos são necessários!")
    else:
        print("❌ Diretório de modelos não encontrado!")
    
    input("\nPressione ENTER para continuar...")

def show_settings():
    print("\n⚙️  CONFIGURAÇÕES DO SISTEMA")
    print("-" * 50)
    
    settings = {
        "Taxa de amostragem": "125 Hz",
        "Canais EEG": "16 (0-15)",
        "Janela temporal": "2 segundos (250 amostras)",
        "Sobreposição": "50% (125 amostras)",
        "Confiança mínima": "60-70%",
        "Arquitetura": "CNN 1D com BatchNorm",
        "Otimizador": "Adam",
    }
    
    for key, value in settings.items():
        print(f"  {key:20} | {value}")
    
    print(f"\n📁 Diretórios:")
    dirs = {
        "Dataset": "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data",
        "Modelos": "models/",
        "Logs": "logs/"
    }
    
    for name, path in dirs.items():
        status = "✓" if os.path.exists(path) else "❌"
        print(f"  {name:20} | {path} {status}")
    
    # Informações do sistema
    print(f"\n💻 Sistema:")
    import platform
    print(f"  Plataforma          | {platform.system()}")
    print(f"  Python              | {platform.python_version()}")
    
    try:
        import tensorflow as tf
        print(f"  TensorFlow          | {tf.__version__}")
        gpus = len(tf.config.list_physical_devices('GPU'))
        print(f"  GPUs disponíveis    | {gpus}")
    except ImportError:
        print(f"  TensorFlow          | Não instalado")
    
    input("\nPressione ENTER para continuar...")

def main():
    """Função principal da CLI"""
    while True:
        try:
            # Limpa a tela (funciona no Windows e Linux)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print_banner()
            print_menu()
            
            choice = get_user_choice()
            
            if choice == '1':
                run_analysis()
            elif choice == '2':
                run_full_training()
            elif choice == '3':
                run_single_patient()
            elif choice == '4':
                run_cross_validation()
            elif choice == '5':
                run_leave_one_out()
            elif choice == '6':
                test_models()
            elif choice == '7':
                compare_models()
            elif choice == '8':
                show_settings()
            elif choice == '9':
                print("\n👋 Obrigado por usar o BrainBridge!")
                print("🧠 Sistema de treinamento EEG finalizado.")
                break
            else:
                print("❌ Opção inválida! Escolha entre 1-9.")
                input("Pressione ENTER para continuar...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Operação interrompida pelo usuário.")
            choice = input("Deseja sair? (s/N): ")
            if choice.lower() == 's':
                break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {str(e)}")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    main()
