import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

def analyze_existing_models():
    """Analisa os modelos existentes e suas performances"""
    print("🔍 Analisando modelos existentes...")
    
    if not os.path.exists("models"):
        print("❌ Pasta 'models' não encontrada")
        return
    
    model_files = [f for f in os.listdir("models") if f.endswith('.keras')]
    
    if not model_files:
        print("❌ Nenhum modelo .keras encontrado")
        return
    
    print(f"📁 Encontrados {len(model_files)} modelos:")
    
    model_info = []
    
    for model_file in model_files:
        try:
            # Extrair informações do nome do arquivo
            parts = model_file.split('-')
            if len(parts) >= 3:
                patient = parts[0]
                accuracy = float(parts[1])
                loss = float(parts[2].split('.')[0])
                
                model_info.append({
                    'arquivo': model_file,
                    'paciente': patient,
                    'acuracia': accuracy,
                    'loss': loss,
                    'status': '✅' if accuracy >= 70 else '⚠️' if accuracy >= 60 else '❌'
                })
                
                print(f"  {model_info[-1]['status']} {patient}: {accuracy}% (loss: {loss})")
            else:
                print(f"  ❓ {model_file}: formato não reconhecido")
                
        except Exception as e:
            print(f"  ❌ Erro ao analisar {model_file}: {e}")
    
    if model_info:
        # Estatísticas
        accuracies = [m['acuracia'] for m in model_info]
        print(f"\n📊 Estatísticas:")
        print(f"  Acurácia média: {np.mean(accuracies):.2f}%")
        print(f"  Melhor acurácia: {np.max(accuracies):.2f}%")
        print(f"  Modelos ≥70%: {sum(1 for acc in accuracies if acc >= 70)}")
        print(f"  Modelos ≥60%: {sum(1 for acc in accuracies if acc >= 60)}")
        
        # Encontrar o melhor modelo
        best_model = max(model_info, key=lambda x: x['acuracia'])
        print(f"\n🏆 Melhor modelo: {best_model['arquivo']}")
        print(f"  Paciente: {best_model['paciente']}")
        print(f"  Acurácia: {best_model['acuracia']}%")
        
        return best_model
    
    return None

def test_model_on_sample_data(model_path):
    """Testa um modelo em dados de amostra"""
    try:
        print(f"\n🧪 Testando modelo: {os.path.basename(model_path)}")
        
        # Carregar modelo
        model = load_model(model_path)
        
        print("📋 Resumo do modelo:")
        model.summary()
        
        # Criar dados de teste sintéticos
        print("\n🔬 Testando com dados sintéticos...")
        test_data = np.random.randn(10, 16, 125)  # 10 amostras de teste
        
        predictions = model.predict(test_data, verbose=0)
        
        print("Predições de exemplo:")
        for i, pred in enumerate(predictions):
            class_pred = np.argmax(pred)
            confidence = np.max(pred)
            action = ["T0 (repouso)", "T1 (mão esquerda)", "T2 (mão direita)"][class_pred]
            print(f"  Amostra {i+1}: {action} (confiança: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar modelo: {e}")
        return False

def check_data_distribution():
    """Verifica a distribuição dos dados processados"""
    print("\n📈 Verificando distribuição dos dados...")
    
    if not os.path.exists("processed_eeg_data"):
        print("❌ Pasta 'processed_eeg_data' não encontrada")
        return
    
    actions = ["T0", "T1", "T2"]
    action_names = {"T0": "Repouso", "T1": "Mão Esquerda", "T2": "Mão Direita"}
    
    total_samples = 0
    
    for action in actions:
        action_path = f"processed_eeg_data/{action}"
        if os.path.exists(action_path):
            patients = os.listdir(action_path)
            samples_count = 0
            
            for patient in patients:
                patient_path = os.path.join(action_path, patient)
                if os.path.isdir(patient_path):
                    files = [f for f in os.listdir(patient_path) if f.endswith('.npy')]
                    samples_count += len(files)
            
            total_samples += samples_count
            print(f"  {action_names[action]}: {len(patients)} pacientes, {samples_count} amostras")
        else:
            print(f"  {action_names[action]}: não encontrado")
    
    print(f"\n📊 Total: {total_samples} amostras")
    return total_samples

def generate_recommendations():
    """Gera recomendações para melhorar a acurácia"""
    print("\n💡 Recomendações para atingir 70% de acurácia:")
    print("="*50)
    
    # Verificar dados
    total_samples = check_data_distribution()
    
    recommendations = []
    
    if total_samples < 1000:
        recommendations.append("🔄 Aumentar o dataset - considere data augmentation")
    
    if total_samples < 500:
        recommendations.append("📊 Dataset muito pequeno - colete mais dados")
    
    # Verificar modelos existentes
    best_model = analyze_existing_models()
    
    if best_model and best_model['acuracia'] < 70:
        if best_model['acuracia'] < 50:
            recommendations.append("🏗️ Arquitetura do modelo - considere redes mais profundas")
        elif best_model['acuracia'] < 60:
            recommendations.append("⚙️ Hiperparâmetros - ajuste learning rate e regularização")
        else:
            recommendations.append("🎯 Quase lá! Tente ensemble de modelos ou fine-tuning")
    
    recommendations.extend([
        "🧹 Pré-processamento - normalize e filtre os dados EEG",
        "🎲 Data augmentation - ruído, shift temporal, rotação",
        "🏋️ Transfer learning - use modelos pré-treinados",
        "🎭 Ensemble - combine múltiplos modelos",
        "📊 Cross-validation - validação cruzada k-fold"
    ])
    
    print("\n🎯 Estratégias recomendadas:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return recommendations

def main():
    print("🧠 BCI Model Analyzer - Análise Rápida do Sistema")
    print("="*60)
    
    # Verificar estrutura de arquivos
    print("📁 Verificando estrutura de arquivos...")
    required_dirs = ["processed_eeg_data", "models"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (não encontrado)")
    
    # Analisar modelos existentes
    best_model = analyze_existing_models()
    
    # Verificar distribuição de dados
    check_data_distribution()
    
    # Testar melhor modelo se disponível
    if best_model:
        model_path = f"models/{best_model['arquivo']}"
        test_model_on_sample_data(model_path)
    
    # Gerar recomendações
    generate_recommendations()
    
    print("\n" + "="*60)
    print("✨ Análise concluída!")

if __name__ == "__main__":
    main()
