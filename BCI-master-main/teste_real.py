import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test_model_real(model_path, test_subjects=None):
    """Teste real do modelo com dados de validação"""
    print(f"🧠 TESTE REAL: {os.path.basename(model_path)}")
    print("=" * 60)
    
    # Carregar modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None
    
    # Definir sujeitos de teste (não usados no treinamento)
    if test_subjects is None:
        test_subjects = ['S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015']
    
    print(f"🎯 Testando com sujeitos: {test_subjects}")
    
    # Carregar dados de teste
    X_test = []
    y_test = []
    actions = ['T0', 'T1', 'T2']
    action_names = ['Repouso', 'Mão Esquerda', 'Mão Direita']
    
    samples_per_class = {0: 0, 1: 0, 2: 0}
    
    for action_idx, action in enumerate(actions):
        action_dir = f"processed_eeg_data/{action}"
        
        for subject in test_subjects:
            subject_path = os.path.join(action_dir, subject)
            if os.path.exists(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.npy'):
                        try:
                            filepath = os.path.join(subject_path, file)
                            data = np.load(filepath)
                            
                            # Verificar dimensões
                            if data.shape == (16, 125):
                                X_test.append(data)
                                y_test.append(action_idx)
                                samples_per_class[action_idx] += 1
                        except Exception as e:
                            print(f"⚠️ Erro ao carregar {file}: {e}")
                            continue
    
    if len(X_test) == 0:
        print("❌ Nenhum dado de teste encontrado!")
        return None
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n📊 Dados de teste carregados:")
    print(f"   Total de amostras: {len(X_test)}")
    print(f"   Dimensões: {X_test.shape}")
    for i, name in enumerate(action_names):
        print(f"   {name}: {samples_per_class[i]} amostras")
    
    # Fazer predições
    print("\n🔄 Fazendo predições...")
    try:
        predictions = model.predict(X_test, batch_size=32, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
    except Exception as e:
        print(f"❌ Erro ao fazer predições: {e}")
        return None
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   Acurácia Geral: {accuracy:.2%}")
    
    # Verificar se atingiu 70%
    if accuracy >= 0.70:
        print("✅ SUCESSO! Modelo atingiu 70%+ de acurácia!")
        status = "APROVADO"
    elif accuracy >= 0.60:
        print("⚠️ Próximo do objetivo (60-70%)")
        status = "QUASE"
    else:
        print("❌ Abaixo de 60% de acurácia")
        status = "REPROVADO"
    
    # Relatório detalhado por classe
    print(f"\n📋 Relatório por classe:")
    report = classification_report(y_test, y_pred, target_names=action_names, output_dict=True)
    
    for i, name in enumerate(action_names):
        precision = report[name]['precision']
        recall = report[name]['recall']
        f1 = report[name]['f1-score']
        print(f"   {name}:")
        print(f"     Precisão: {precision:.2%}")
        print(f"     Recall: {recall:.2%}")
        print(f"     F1-Score: {f1:.2%}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=action_names,
                yticklabels=action_names)
    plt.title(f'Matriz de Confusão - {os.path.basename(model_path)}\nAcurácia: {accuracy:.2%}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{os.path.basename(model_path)}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'status': status,
        'report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'y_true': y_test,
        'y_pred': y_pred
    }

def test_all_best_models():
    """Testa todos os melhores modelos"""
    print("🚀 TESTE REAL DE TODOS OS MELHORES MODELOS")
    print("=" * 80)
    
    # Modelos com 70%+ segundo o nome
    best_models = [
        "models/S007-86.67-acc-loss-0.37.keras",
        "models/S004-76.67-acc-loss-0.56.keras",
        "models/S006-75.56-acc-loss-0.54.keras",
        "models/S001-72.22-acc-loss-0.73.keras",
        "models/S003-71.11-acc-loss-0.65.keras",
        "models/S008-71.11-acc-loss-0.7.keras"
    ]
    
    results = {}
    approved_models = []
    
    for model_path in best_models:
        if os.path.exists(model_path):
            result = test_model_real(model_path)
            if result:
                results[model_path] = result
                if result['status'] == "APROVADO":
                    approved_models.append((model_path, result['accuracy']))
            print("\n" + "-" * 80 + "\n")
    
    # Resumo final
    print("🏆 RESUMO FINAL DOS TESTES REAIS")
    print("=" * 80)
    
    if approved_models:
        print(f"✅ {len(approved_models)} modelo(s) APROVADO(S) com 70%+ de acurácia:")
        
        # Ordenar por acurácia
        approved_models.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_path, acc) in enumerate(approved_models, 1):
            model_name = os.path.basename(model_path)
            print(f"   {i}. {model_name}: {acc:.2%}")
        
        best_model = approved_models[0]
        print(f"\n🥇 MELHOR MODELO: {os.path.basename(best_model[0])}")
        print(f"🎯 ACURÁCIA: {best_model[1]:.2%}")
        print(f"\n🎉 SEU SISTEMA BCI ESTÁ FUNCIONANDO DE VERDADE!")
        print(f"✅ OBJETIVO DE 70% DE ACURÁCIA: ATINGIDO!")
        
    else:
        print("❌ Nenhum modelo atingiu 70% nos testes reais")
        print("💡 Possíveis causas:")
        print("   - Overfitting nos dados de treinamento")
        print("   - Dados de teste muito diferentes")
        print("   - Necessário mais pré-processamento")
    
    return results

if __name__ == "__main__":
    # Executar teste real
    results = test_all_best_models()
