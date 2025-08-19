import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report

def test_single_model(model_path):
    """Testa um modelo específico com dados de validação"""
    print(f"\n🧠 Testando: {os.path.basename(model_path)}")
    print("-" * 40)
    
    try:
        # Carregar modelo
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo carregado")
        
        # Carregar alguns dados de teste
        X_test = []
        y_test = []
        
        # Usar apenas alguns sujeitos para teste rápido
        test_subjects = ['S009', 'S011', 'S012']  # Sujeitos não usados no treinamento
        actions = ['T0', 'T1', 'T2']
        
        for action_idx, action in enumerate(actions):
            action_dir = f"processed_eeg_data/{action}"
            count = 0
            
            for subject in test_subjects:
                subject_path = os.path.join(action_dir, subject)
                if os.path.exists(subject_path):
                    for file in os.listdir(subject_path):
                        if file.endswith('.npy') and count < 20:  # Máximo 20 amostras por classe
                            try:
                                data = np.load(os.path.join(subject_path, file))
                                if data.shape == (16, 400):  # Verificar dimensões
                                    X_test.append(data)
                                    y_test.append(action_idx)
                                    count += 1
                            except:
                                continue
        
        if len(X_test) == 0:
            print("❌ Nenhum dado de teste encontrado!")
            return None
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"📊 Dados de teste: {X_test.shape}")
        print(f"📈 Distribuição: T0={np.sum(y_test==0)}, T1={np.sum(y_test==1)}, T2={np.sum(y_test==2)}")
        
        # Fazer predições
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calcular acurácia
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Acurácia: {accuracy:.2%}")
        
        # Status
        if accuracy >= 0.70:
            print("✅ SUCESSO! Modelo atinge 70%+ de acurácia")
        elif accuracy >= 0.60:
            print("⚠️ Modelo próximo do objetivo (60-70%)")
        else:
            print("❌ Modelo abaixo de 60% de acurácia")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def main():
    print("🚀 TESTE RÁPIDO DOS MELHORES MODELOS BCI")
    print("=" * 50)
    
    # Testar os melhores modelos baseado no nome
    best_models = [
        "models/S007-86.67-acc-loss-0.37.keras",
        "models/S004-76.67-acc-loss-0.56.keras", 
        "models/S006-75.56-acc-loss-0.54.keras",
        "models/S001-72.22-acc-loss-0.73.keras",
        "models/S003-71.11-acc-loss-0.65.keras",
        "models/S008-71.11-acc-loss-0.7.keras"
    ]
    
    successful_models = []
    
    for model_path in best_models:
        if os.path.exists(model_path):
            accuracy = test_single_model(model_path)
            if accuracy and accuracy >= 0.70:
                successful_models.append((model_path, accuracy))
    
    print("\n" + "=" * 50)
    print("📋 RESUMO FINAL")
    print("=" * 50)
    
    if successful_models:
        print(f"🎉 {len(successful_models)} modelo(s) atingiu(ram) 70%+ de acurácia:")
        for model_path, acc in successful_models:
            model_name = os.path.basename(model_path)
            print(f"  ✅ {model_name}: {acc:.2%}")
        
        print(f"\n🚀 SEU SISTEMA BCI ESTÁ FUNCIONANDO!")
        print(f"🎯 Melhor modelo: {os.path.basename(successful_models[0][0])}")
        print(f"📊 Acurácia: {successful_models[0][1]:.2%}")
        
    else:
        print("⚠️ Nenhum modelo atingiu 70% nos dados de teste")
        print("💡 Recomendação: Execute o treinamento novamente")

if __name__ == "__main__":
    main()
