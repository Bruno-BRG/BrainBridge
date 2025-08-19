import os
import numpy as np

def quick_analysis():
    """Análise rápida do sistema BCI"""
    print("🧠 Análise Rápida - Sistema BCI")
    print("="*40)
    
    # Verificar modelos existentes
    print("\n📋 Modelos existentes:")
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith('.keras')]
        if models:
            print(f"  Encontrados {len(models)} modelos:")
            best_acc = 0
            for model in models:
                try:
                    parts = model.split('-')
                    if len(parts) >= 2:
                        patient = parts[0]
                        acc = float(parts[1])
                        status = "✅" if acc >= 70 else "⚠️" if acc >= 60 else "❌"
                        print(f"    {status} {patient}: {acc}%")
                        best_acc = max(best_acc, acc)
                except:
                    print(f"    ❓ {model}")
            print(f"\n  🏆 Melhor acurácia atual: {best_acc}%")
        else:
            print("  ❌ Nenhum modelo .keras encontrado")
    else:
        print("  ❌ Pasta models não existe")
    
    # Verificar dados processados
    print("\n📊 Dados processados:")
    if os.path.exists("processed_eeg_data"):
        actions = ["T0", "T1", "T2"]
        action_names = {"T0": "Repouso", "T1": "Mão Esquerda", "T2": "Mão Direita"}
        
        total_patients = 0
        for action in actions:
            action_path = f"processed_eeg_data/{action}"
            if os.path.exists(action_path):
                patients = len([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
                total_patients = max(total_patients, patients)
                print(f"  {action_names[action]}: {patients} pacientes")
        
        print(f"\n  📈 Total de pacientes: {total_patients}")
        
        if total_patients >= 10:
            print("  ✅ Dados suficientes para treinamento")
        elif total_patients >= 5:
            print("  ⚠️ Dados limitados - considere data augmentation")
        else:
            print("  ❌ Dados insuficientes - precisa de mais dados")
    else:
        print("  ❌ Pasta processed_eeg_data não existe")
    
    # Verificar dados brutos
    print("\n📁 Dados brutos:")
    if os.path.exists("eeg_data"):
        subjects = len([d for d in os.listdir("eeg_data") if os.path.isdir(os.path.join("eeg_data", d))])
        print(f"  ✅ {subjects} sujeitos disponíveis")
    else:
        print("  ❌ Pasta eeg_data não existe")
    
    # Recomendações
    print("\n💡 Próximos passos para 70% de acurácia:")
    print("  1. 🔄 Execute o training_improved.py (já em execução)")
    print("  2. 🎯 Use ensemble de múltiplos modelos")
    print("  3. 🧹 Aplique pré-processamento avançado")
    print("  4. 🎲 Aumente dados com augmentation")
    print("  5. ⚙️ Otimize hiperparâmetros")

if __name__ == "__main__":
    quick_analysis()
