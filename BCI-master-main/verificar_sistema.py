import os
import numpy as np

def verificar_sistema():
    print("🧠 VERIFICAÇÃO RÁPIDA DO SISTEMA BCI")
    print("=" * 50)
    
    # 1. Verificar modelos
    print("\n📋 Modelos disponíveis:")
    models_dir = "models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        models_70plus = []
        
        for model in sorted(models):
            try:
                # Extrair acurácia do nome do arquivo
                parts = model.split('-')
                if len(parts) >= 2:
                    acc = float(parts[1])
                    status = "✅" if acc >= 70 else "⚠️" if acc >= 60 else "❌"
                    print(f"  {status} {model}: {acc:.1f}%")
                    if acc >= 70:
                        models_70plus.append((model, acc))
            except:
                print(f"  ❓ {model}")
        
        print(f"\n🎯 Modelos com 70%+ de acurácia: {len(models_70plus)}")
        for model, acc in models_70plus:
            print(f"  ✅ {model}: {acc:.1f}%")
    
    # 2. Verificar dados
    print("\n📊 Dados processados:")
    data_dir = "processed_eeg_data"
    if os.path.exists(data_dir):
        actions = ["T0", "T1", "T2"]
        for action in actions:
            action_path = os.path.join(data_dir, action)
            if os.path.exists(action_path):
                count = len([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
                print(f"  {action}: {count} sujeitos")
    
    # 3. Verificar uma amostra de dados
    print("\n🔍 Verificando formato dos dados:")
    try:
        sample_file = None
        for action in ["T0", "T1", "T2"]:
            action_path = os.path.join("processed_eeg_data", action)
            if os.path.exists(action_path):
                for subject in os.listdir(action_path):
                    subject_path = os.path.join(action_path, subject)
                    if os.path.isdir(subject_path):
                        for file in os.listdir(subject_path):
                            if file.endswith('.npy'):
                                sample_file = os.path.join(subject_path, file)
                                break
                        if sample_file:
                            break
                if sample_file:
                    break
        
        if sample_file:
            data = np.load(sample_file)
            print(f"  ✅ Arquivo de exemplo: {os.path.basename(sample_file)}")
            print(f"  📐 Dimensões: {data.shape}")
            print(f"  📊 Tipo: {data.dtype}")
            
            if data.shape == (16, 400):
                print("  ✅ Formato correto: 16 canais x 400 amostras")
            else:
                print("  ⚠️ Formato não padrão")
        
    except Exception as e:
        print(f"  ❌ Erro ao verificar dados: {e}")
    
    # 4. Conclusão
    print("\n" + "=" * 50)
    print("📋 CONCLUSÃO")
    print("=" * 50)
    
    if len(models_70plus) > 0:
        print("🎉 SEU SISTEMA ESTÁ FUNCIONANDO!")
        print(f"✅ {len(models_70plus)} modelo(s) com 70%+ de acurácia")
        print(f"🏆 Melhor modelo: {models_70plus[0][0]} ({models_70plus[0][1]:.1f}%)")
        print("\n💡 Para testar com dados reais:")
        print("  1. Use o arquivo analysis.py")
        print("  2. Modifique MODEL_NAME para apontar para seu melhor modelo")
        print("  3. Execute para ver matriz de confusão")
        
        return True
    else:
        print("⚠️ Nenhum modelo com 70%+ de acurácia encontrado")
        print("💡 Recomendações:")
        print("  - Execute training.py para treinar novos modelos")
        print("  - Verifique se os dados estão processados corretamente")
        
        return False

if __name__ == "__main__":
    verificar_sistema()
