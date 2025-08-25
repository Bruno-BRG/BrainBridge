#!/usr/bin/env python3
"""
BrainBridge - Sistema de Treinamento EEG Motor Imagery
CLI Principal com Interface Interativa
"""

import os
import sys
import time
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_eeg_motor_imagery import main as train_main, load_all_data, preprocess_data
from analyze_eeg_data import analyze_all_data, generate_analysis_report
from test_eeg_model import batch_test_model
from eeg_classifier_integration import create_classifier_from_models_dir
from training_utils import (
    train_single_subject, 
    train_cross_validation_split, 
    train_leave_one_out_validation, 
    compare_all_models
)
from config import DIRECTORIES, CLI_CONFIG, validate_data_directory, get_system_info

# Cores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Exibe o banner do sistema"""
    banner = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  {Colors.BOLD}██████╗ ██████╗  █████╗ ██╗███╗   ██╗██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗{Colors.ENDC}{Colors.CYAN}  ║
║  {Colors.BOLD}██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝{Colors.ENDC}{Colors.CYAN}  ║
║  {Colors.BOLD}██████╔╝██████╔╝███████║██║██╔██╗ ██║██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗{Colors.ENDC}{Colors.CYAN}    ║
║  {Colors.BOLD}██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝{Colors.ENDC}{Colors.CYAN}    ║
║  {Colors.BOLD}██████╔╝██║  ██║██║  ██║██║██║ ╚████║██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗{Colors.ENDC}{Colors.CYAN}  ║
║  {Colors.BOLD}╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝{Colors.ENDC}{Colors.CYAN}  ║
║                                                                               ║
║              {Colors.YELLOW}🧠 Sistema de Treinamento EEG Motor Imagery 🧠{Colors.ENDC}{Colors.CYAN}                ║
║                        {Colors.GREEN}Versão 2.0 - Interface CLI Avançada{Colors.ENDC}{Colors.CYAN}                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def print_menu():
    """Exibe o menu principal"""
    menu = f"""
{Colors.HEADER}┌─────────────────────────────────────────────────────────────────┐
│                        {Colors.BOLD}MENU PRINCIPAL{Colors.ENDC}{Colors.HEADER}                           │
├─────────────────────────────────────────────────────────────────┤{Colors.ENDC}
{Colors.BLUE}│  1. {Colors.GREEN}📊 Analisar Dataset EEG{Colors.ENDC}                                  {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  2. {Colors.YELLOW}🤖 Treinar Modelo - Todos os Pacientes{Colors.ENDC}                    {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  3. {Colors.YELLOW}👤 Treinar Modelo - Paciente Específico{Colors.ENDC}                   {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  4. {Colors.YELLOW}📚 Treinar Modelo - Cross-Validation (50/50){Colors.ENDC}               {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  5. {Colors.YELLOW}🔄 Treinar Modelo - Leave-One-Out{Colors.ENDC}                         {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  6. {Colors.CYAN}🧪 Testar Modelo Existente{Colors.ENDC}                                {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  7. {Colors.CYAN}📈 Comparar Modelos{Colors.ENDC}                                       {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  8. {Colors.GREEN}⚙️  Configurações{Colors.ENDC}                                          {Colors.BLUE}│{Colors.ENDC}
{Colors.BLUE}│  9. {Colors.RED}🚪 Sair{Colors.ENDC}                                                   {Colors.BLUE}│{Colors.ENDC}
{Colors.HEADER}└─────────────────────────────────────────────────────────────────┘{Colors.ENDC}
"""
    print(menu)

def get_available_subjects(data_dir: str = None) -> List[str]:
    """Retorna lista de sujeitos disponíveis"""
    if data_dir is None:
        data_dir = str(DIRECTORIES['data'])
    
    subjects = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)) and item.startswith('S'):
                subjects.append(item)
    return sorted(subjects)

def print_subjects_list(subjects: List[str]):
    """Exibe lista de sujeitos em formato bonito"""
    print(f"\n{Colors.CYAN}📋 Sujeitos Disponíveis:{Colors.ENDC}")
    print(f"{Colors.BLUE}┌─────────────────────────────────────────────────────────────────┐{Colors.ENDC}")
    
    # Exibe em colunas
    cols = 4
    for i in range(0, len(subjects), cols):
        row_subjects = subjects[i:i+cols]
        row_str = "│ "
        for j, subject in enumerate(row_subjects):
            row_str += f"{Colors.GREEN}{subject:>6}{Colors.ENDC}"
            if j < len(row_subjects) - 1:
                row_str += " │ "
        
        # Preenche espaços vazios
        while len(row_subjects) < cols:
            row_str += "      │ "
            row_subjects.append("")
        
        row_str += " │"
        print(f"{Colors.BLUE}{row_str}{Colors.ENDC}")
    
    print(f"{Colors.BLUE}└─────────────────────────────────────────────────────────────────┘{Colors.ENDC}")
    print(f"{Colors.YELLOW}Total: {len(subjects)} sujeitos{Colors.ENDC}")

def progress_bar(current: int, total: int, prefix: str = "Progresso", length: int = 50):
    """Exibe barra de progresso"""
    percent = current / total
    bar_length = int(length * percent)
    bar = "█" * bar_length + "▒" * (length - bar_length)
    
    print(f"\r{Colors.CYAN}{prefix}: {Colors.GREEN}[{bar}] {percent:.1%} ({current}/{total}){Colors.ENDC}", end="", flush=True)
    if current == total:
        print()

def analyze_dataset():
    """Opção 1: Analisar dataset"""
    print(f"\n{Colors.HEADER}📊 ANÁLISE DO DATASET EEG{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    # Valida diretório de dados
    is_valid, message = validate_data_directory()
    if not is_valid:
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
        return
    
    data_dir = str(DIRECTORIES['data'])
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
    print(f"{Colors.YELLOW}🔍 Analisando dados em: {data_dir}{Colors.ENDC}")
    
    try:
        results = analyze_all_data(data_dir)
        summary = generate_analysis_report(results)
        
        print(f"\n{Colors.GREEN}✅ Análise concluída com sucesso!{Colors.ENDC}")
        print(f"{Colors.CYAN}📄 Relatório salvo em: eeg_data_analysis.txt{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante análise: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def train_all_patients():
    """Opção 2: Treinar com todos os pacientes"""
    print(f"\n{Colors.HEADER}🤖 TREINAMENTO - TODOS OS PACIENTES{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    confirm = input(f"{Colors.YELLOW}Deseja treinar usando todos os pacientes? (s/N): {Colors.ENDC}")
    if confirm.lower() != 's':
        return
    
    print(f"{Colors.GREEN}🚀 Iniciando treinamento com todos os pacientes...{Colors.ENDC}")
    
    try:
        # Chama a função de treinamento original
        train_main()
        print(f"\n{Colors.GREEN}✅ Treinamento concluído com sucesso!{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante treinamento: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def train_specific_patient():
    """Opção 3: Treinar com paciente específico"""
    print(f"\n{Colors.HEADER}👤 TREINAMENTO - PACIENTE ESPECÍFICO{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    data_dir = str(DIRECTORIES['data'])
    subjects = get_available_subjects()
    
    if not subjects:
        print(f"{Colors.RED}❌ Nenhum sujeito encontrado no dataset{Colors.ENDC}")
        return
    
    print_subjects_list(subjects)
    
    subject_choice = input(f"\n{Colors.YELLOW}Digite o código do sujeito (ex: S001): {Colors.ENDC}").upper()
    
    if subject_choice not in subjects:
        print(f"{Colors.RED}❌ Sujeito inválido!{Colors.ENDC}")
        return
    
    print(f"{Colors.GREEN}🚀 Iniciando treinamento para sujeito: {subject_choice}{Colors.ENDC}")
    
    try:
        # Implementa treinamento para um sujeito específico
        train_single_subject(subject_choice, data_dir)
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante treinamento: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def train_cross_validation():
    """Opção 4: Treinar com cross-validation 50/50"""
    print(f"\n{Colors.HEADER}📚 TREINAMENTO - CROSS-VALIDATION (50/50){Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    subjects = get_available_subjects(data_dir)
    
    if len(subjects) < 4:
        print(f"{Colors.RED}❌ Número insuficiente de sujeitos para cross-validation{Colors.ENDC}")
        return
    
    print(f"{Colors.YELLOW}📊 Total de sujeitos: {len(subjects)}{Colors.ENDC}")
    print(f"{Colors.YELLOW}📈 Treinamento: {len(subjects)//2} sujeitos{Colors.ENDC}")
    print(f"{Colors.YELLOW}🧪 Teste: {len(subjects) - len(subjects)//2} sujeitos{Colors.ENDC}")
    
    confirm = input(f"\n{Colors.YELLOW}Confirmar treinamento cross-validation? (s/N): {Colors.ENDC}")
    if confirm.lower() != 's':
        return
    
    print(f"{Colors.GREEN}🚀 Iniciando cross-validation...{Colors.ENDC}")
    
    try:
        train_cross_validation_split(subjects, data_dir)
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante cross-validation: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def train_leave_one_out():
    """Opção 5: Treinar com leave-one-out"""
    print(f"\n{Colors.HEADER}🔄 TREINAMENTO - LEAVE-ONE-OUT{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    subjects = get_available_subjects(data_dir)
    
    if len(subjects) < 3:
        print(f"{Colors.RED}❌ Número insuficiente de sujeitos para leave-one-out{Colors.ENDC}")
        return
    
    print(f"{Colors.YELLOW}📊 Total de sujeitos: {len(subjects)}{Colors.ENDC}")
    print(f"{Colors.YELLOW}🔄 Serão treinados {len(subjects)} modelos (um para cada sujeito teste){Colors.ENDC}")
    
    confirm = input(f"\n{Colors.YELLOW}Confirmar treinamento leave-one-out? (s/N): {Colors.ENDC}")
    if confirm.lower() != 's':
        return
    
    print(f"{Colors.GREEN}🚀 Iniciando leave-one-out validation...{Colors.ENDC}")
    
    try:
        train_leave_one_out_validation(subjects, data_dir)
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante leave-one-out: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def test_existing_model():
    """Opção 6: Testar modelo existente"""
    print(f"\n{Colors.HEADER}🧪 TESTE DE MODELO EXISTENTE{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"{Colors.RED}❌ Diretório de modelos não encontrado: {models_dir}{Colors.ENDC}")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5', '.model'))]
    
    if not model_files:
        print(f"{Colors.RED}❌ Nenhum modelo encontrado no diretório{Colors.ENDC}")
        return
    
    print(f"{Colors.GREEN}📁 Modelos disponíveis:{Colors.ENDC}")
    for i, model in enumerate(model_files, 1):
        print(f"{Colors.YELLOW}  {i}. {model}{Colors.ENDC}")
    
    try:
        choice = int(input(f"\n{Colors.YELLOW}Escolha o modelo (1-{len(model_files)}): {Colors.ENDC}"))
        if 1 <= choice <= len(model_files):
            selected_model = os.path.join(models_dir, model_files[choice-1])
            
            num_tests = int(input(f"{Colors.YELLOW}Número de arquivos para teste (padrão 10): {Colors.ENDC}") or "10")
            
            print(f"{Colors.GREEN}🧪 Testando modelo: {selected_model}{Colors.ENDC}")
            
            data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
            batch_test_model(selected_model, data_dir, num_tests)
            
        else:
            print(f"{Colors.RED}❌ Escolha inválida{Colors.ENDC}")
            
    except ValueError:
        print(f"{Colors.RED}❌ Entrada inválida{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante teste: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def compare_models():
    """Opção 7: Comparar modelos"""
    print(f"\n{Colors.HEADER}📈 COMPARAÇÃO DE MODELOS{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"{Colors.RED}❌ Diretório de modelos não encontrado{Colors.ENDC}")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5', '.model'))]
    
    if len(model_files) < 2:
        print(f"{Colors.RED}❌ Pelo menos 2 modelos são necessários para comparação{Colors.ENDC}")
        return
    
    print(f"{Colors.GREEN}📊 Comparando {len(model_files)} modelos...{Colors.ENDC}")
    
    try:
        compare_all_models(models_dir, model_files)
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante comparação: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def show_settings():
    """Opção 8: Configurações"""
    print(f"\n{Colors.HEADER}⚙️  CONFIGURAÇÕES DO SISTEMA{Colors.ENDC}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    # Configurações do modelo
    from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PREDICTION_CONFIG
    
    settings = {
        "Taxa de amostragem": f"{DATA_CONFIG['sample_rate']} Hz",
        "Canais EEG": f"{DATA_CONFIG['channels']} (0-{DATA_CONFIG['channels']-1})",
        "Janela temporal": f"{DATA_CONFIG['window_size']/DATA_CONFIG['sample_rate']:.1f}s ({DATA_CONFIG['window_size']} amostras)",
        "Sobreposição": f"{DATA_CONFIG['overlap']/DATA_CONFIG['sample_rate']:.1f}s ({DATA_CONFIG['overlap']} amostras)",
        "Confiança mínima": f"{PREDICTION_CONFIG['min_confidence']*100:.0f}%",
        "Arquitetura": MODEL_CONFIG['architecture'],
        "Otimizador": MODEL_CONFIG['optimizer'],
        "Função de perda": MODEL_CONFIG['loss'],
        "Batch size": str(TRAINING_CONFIG['batch_size']),
        "Epochs máximos": str(TRAINING_CONFIG['epochs'])
    }
    
    print(f"{Colors.BLUE}┌─────────────────────────────────────────────────────────────────┐{Colors.ENDC}")
    for key, value in settings.items():
        print(f"{Colors.BLUE}│ {Colors.YELLOW}{key:20}{Colors.ENDC} {Colors.BLUE}│{Colors.ENDC} {Colors.GREEN}{value:35}{Colors.ENDC} {Colors.BLUE}│{Colors.ENDC}")
    print(f"{Colors.BLUE}└─────────────────────────────────────────────────────────────────┘{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}📁 Diretórios:{Colors.ENDC}")
    for name, path in DIRECTORIES.items():
        status = "✓" if path.exists() else "❌"
        print(f"{Colors.YELLOW}  • {name.capitalize()}: {Colors.GREEN}{path} {status}{Colors.ENDC}")
    
    # Informações do sistema
    print(f"\n{Colors.CYAN}💻 Sistema:{Colors.ENDC}")
    try:
        sys_info = get_system_info()
        print(f"{Colors.YELLOW}  • Plataforma: {Colors.GREEN}{sys_info['platform']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • Python: {Colors.GREEN}{sys_info['python_version']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • TensorFlow: {Colors.GREEN}{sys_info['tensorflow_version']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • GPUs: {Colors.GREEN}{sys_info['gpus_available']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • CPUs: {Colors.GREEN}{sys_info['cpu_count']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • Memória: {Colors.GREEN}{sys_info['memory_gb']} GB{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}  ❌ Erro ao obter informações do sistema: {str(e)}{Colors.ENDC}")
    
    # Status do dataset
    print(f"\n{Colors.CYAN}📊 Dataset:{Colors.ENDC}")
    is_valid, message = validate_data_directory()
    if is_valid:
        print(f"{Colors.GREEN}  ✓ {message}{Colors.ENDC}")
    else:
        print(f"{Colors.RED}  ❌ {message}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")

def main():
    """Função principal da CLI"""
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
            print_banner()
            print_menu()
            
            try:
                choice = input(f"\n{Colors.YELLOW}🎯 Escolha uma opção (1-9): {Colors.ENDC}")
                
                if choice == '1':
                    analyze_dataset()
                elif choice == '2':
                    train_all_patients()
                elif choice == '3':
                    train_specific_patient()
                elif choice == '4':
                    train_cross_validation()
                elif choice == '5':
                    train_leave_one_out()
                elif choice == '6':
                    test_existing_model()
                elif choice == '7':
                    compare_models()
                elif choice == '8':
                    show_settings()
                elif choice == '9':
                    print(f"\n{Colors.GREEN}👋 Obrigado por usar o BrainBridge!{Colors.ENDC}")
                    print(f"{Colors.CYAN}🧠 Sistema de treinamento EEG finalizado.{Colors.ENDC}")
                    break
                else:
                    print(f"{Colors.RED}❌ Opção inválida! Escolha entre 1-9.{Colors.ENDC}")
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}⚠️  Operação interrompida pelo usuário.{Colors.ENDC}")
                choice = input(f"{Colors.YELLOW}Deseja sair? (s/N): {Colors.ENDC}")
                if choice.lower() == 's':
                    break
            except Exception as e:
                print(f"\n{Colors.RED}❌ Erro inesperado: {str(e)}{Colors.ENDC}")
                input(f"{Colors.YELLOW}Pressione ENTER para continuar...{Colors.ENDC}")
                
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}👋 Sistema finalizado pelo usuário.{Colors.ENDC}")

if __name__ == "__main__":
    main()
