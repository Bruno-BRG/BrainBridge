"""
Teste de Carregamento e Validação do Modelo BCI
Script para verificar se o modelo EEGNet pode ser carregado e usado para predições
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json

# Adicionar pasta src e final ao path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from realtime_bci_system import EEGNet, RobustEEGNormalizer

def test_model_loading(model_path):
    """Testar carregamento do modelo"""
    print(f"🧪 Testando carregamento do modelo: {model_path}")
    
    try:
        # Configurar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {device}")
        
        # Carregar checkpoint (PyTorch 2.6 compatibility)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ Checkpoint carregado")
        
        # Verificar conteúdo do checkpoint
        print(f"📋 Conteúdo do checkpoint:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                print(f"   - {key}: {len(checkpoint[key])} parâmetros")
            elif key == 'normalization_stats':
                print(f"   - {key}: {list(checkpoint[key].keys())}")
            else:
                print(f"   - {key}: {checkpoint[key]}")
        
        # Extrair parâmetros do modelo
        if 'model_params' in checkpoint:
            model_params = checkpoint['model_params']
            print(f"📊 Parâmetros do modelo: {model_params}")
        else:
            # Parâmetros padrão
            model_params = {
                'n_channels': 16,
                'n_classes': 2,
                'n_samples': 400,
                'dropout_rate': 0.5,
                'F1': 4,
                'D': 2,
                'F2': 8
            }
            print(f"⚠️ Usando parâmetros padrão: {model_params}")
        
        # Criar modelo
        model = EEGNet(**model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Modelo criado e carregado com sucesso")
        print(f"📊 Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        
        # Testar predição com dados simulados
        print(f"\n🧪 Testando predição com dados simulados...")
        
        # Criar dados de teste
        n_channels = model_params['n_channels']
        n_samples = model_params['n_samples']
        batch_size = 1
        
        # Dados simulados (ruído gaussiano)
        test_data = torch.randn(batch_size, n_channels, n_samples, device=device)
        
        # Predição
        with torch.no_grad():
            outputs = model(test_data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"📊 Resultado da predição:")
        print(f"   Shape de entrada: {test_data.shape}")
        print(f"   Shape de saída: {outputs.shape}")
        print(f"   Classe predita: {predicted_class}")
        print(f"   Confiança: {confidence:.3f}")
        print(f"   Probabilidades: {probabilities[0].cpu().numpy()}")
        
        # Testar normalizador se disponível
        if 'normalization_stats' in checkpoint:
            print(f"\n🧪 Testando normalizador...")
            
            normalizer = RobustEEGNormalizer()
            normalizer.load_stats(checkpoint['normalization_stats'])
            
            # Dados de teste para normalização
            test_data_norm = np.random.randn(n_channels, n_samples) * 100  # Dados com escala maior
            
            # Normalizar
            normalized_data = normalizer.transform_single(test_data_norm)
            
            print(f"   Dados originais: média={np.mean(test_data_norm):.3f}, std={np.std(test_data_norm):.3f}")
            print(f"   Dados normalizados: média={np.mean(normalized_data):.3f}, std={np.std(normalized_data):.3f}")
            print(f"   ✅ Normalização funcionando")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_available_models():
    """Listar modelos disponíveis"""
    print(f"📂 Modelos disponíveis:")
    
    # Procurar na pasta models
    models_dir = Path.cwd() / "models"
    if models_dir.exists():
        models = list(models_dir.glob("*.pt"))
        if models:
            for i, model in enumerate(models):
                print(f"   {i+1}. {model.name}")
            return models
        else:
            print(f"   Nenhum modelo encontrado em {models_dir}")
    else:
        print(f"   Pasta models não encontrada: {models_dir}")
    
    return []

def main():
    """Função principal"""
    print(f"🧪 TESTE DE MODELO BCI")
    print(f"=" * 50)
    
    # Listar modelos disponíveis
    models = list_available_models()
    
    if not models:
        print(f"❌ Nenhum modelo encontrado para testar")
        return
    
    # Testar cada modelo
    for model_path in models:
        print(f"\n" + "="*50)
        success = test_model_loading(model_path)
        
        if success:
            print(f"✅ Modelo {model_path.name} passou em todos os testes")
        else:
            print(f"❌ Modelo {model_path.name} falhou nos testes")
    
    print(f"\n🏁 Teste concluído!")

if __name__ == "__main__":
    main()
