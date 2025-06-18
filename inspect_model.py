import torch
import numpy as np
from pathlib import Path

def inspect_model(model_path):
    """Inspeciona um modelo salvo"""
    print(f"\n=== Inspecionando modelo: {model_path} ===")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Tipo do checkpoint: {type(checkpoint)}")
        print(f"Chaves do checkpoint: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\nChaves do model_state_dict:")
            for key in list(state_dict.keys())[:10]:  # Primeiros 10
                print(f"  {key}: {state_dict[key].shape}")
            if len(state_dict.keys()) > 10:
                print(f"  ... e mais {len(state_dict.keys()) - 10} camadas")
                
        # Verificar se tem final_layer
        has_final_layer = any('final_layer' in key for key in state_dict.keys())
        print(f"\nTem 'final_layer': {has_final_layer}")
        
        # Verificar dimensões de entrada
        for key in state_dict.keys():
            if 'weight' in key and 'conv' in key.lower():
                print(f"Primeira camada convolucional: {key} -> {state_dict[key].shape}")
                break
                
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

# Inspecionar alguns modelos
models_dir = Path("models/teste")
if models_dir.exists():
    for model_file in list(models_dir.glob("*.pt"))[:2]:  # Apenas os primeiros 2
        inspect_model(model_file)
        
print("\n=== Teste de criação do modelo EEGInceptionERP ===")
try:
    from braindecode.models import EEGInceptionERP
    
    # Criar modelo com parâmetros padrão
    model = EEGInceptionERP(
        n_chans=16,
        n_outputs=2, 
        n_times=400,
        sfreq=125.0
    )
    
    print("✅ EEGInceptionERP criado com sucesso!")
    print(f"Modelo: {model}")
    
    # Verificar arquitetura
    print("\nCamadas do modelo:")
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module}")
            
except Exception as e:
    print(f"❌ Erro ao criar EEGInceptionERP: {e}")
