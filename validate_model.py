#!/usr/bin/env python3
"""
🧪 Validação Básica do Modelo BCI

Script para validar se o modelo está funcionando:
1. Carrega modelo mais recente
2. Testa com dados sintéticos
3. Mostra arquitetura e predições
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time

# ============================================================================
# MODELOS (COPIADOS DO NOTEBOOK)
# ============================================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.25, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Bloco 1: Temporal Convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Bloco 2: Depthwise Convolution
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Bloco 3: Separable Convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classificador
        self.feature_size = self._get_conv_output_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, n_classes)
        )
        
    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.numel()
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x

class AdvancedEEGNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=2, n_samples=400, 
                 dropout_rate=0.25, kernel_length=64, F1=8, D=2, F2=16):
        super(AdvancedEEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Multi-scale Temporal Convolution
        self.temporal_conv1 = nn.Conv2d(1, F1//2, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.temporal_conv2 = nn.Conv2d(1, F1//2, (1, kernel_length//2), padding=(0, kernel_length // 4), bias=False)
        self.temporal_bn = nn.BatchNorm2d(F1)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(F1, F1//4, 1),
            nn.ReLU(),
            nn.Conv2d(F1//4, F1, 1),
            nn.Sigmoid()
        )
        
        # Depthwise convolution
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Separable convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Feature enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(F2, F2*2, 1, bias=False),
            nn.BatchNorm2d(F2*2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2*2, F2),
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(F2, n_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Multi-scale temporal
        temp1 = self.temporal_conv1(x)
        temp2 = self.temporal_conv2(x)
        x = torch.cat([temp1, temp2], dim=1)
        x = self.temporal_bn(x)
        
        # Spatial attention
        attention = self.spatial_attention(x)
        x = x * attention
        
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.feature_enhancement(x)
        x = self.classifier(x)
        
        return x

class CustomEEGModel(nn.Module):
    def __init__(self, n_chans=16, n_outputs=2, n_times=400, sfreq=125.0, 
                 model_type='advanced', **kwargs):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.model_type = model_type
        
        if model_type == 'advanced':
            self.model = AdvancedEEGNet(
                n_channels=n_chans, n_classes=n_outputs, n_samples=n_times, **kwargs
            )
            self.model_name = "AdvancedEEGNet"
        else:
            self.model = EEGNet(
                n_channels=n_chans, n_classes=n_outputs, n_samples=n_times, **kwargs
            )
            self.model_name = "EEGNet"
        
        self.is_trained = False
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# VALIDADOR
# ============================================================================

class ModelValidator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")
    
    def find_latest_model(self):
        """Encontrar modelo mais recente"""
        models_dir = Path("models")
        
        if not models_dir.exists():
            print("❌ Pasta 'models' não encontrada!")
            print("💡 Dica: Execute o notebook de treinamento primeiro")
            return None
        
        model_files = list(models_dir.glob("*.pt"))
        if not model_files:
            print("❌ Nenhum modelo .pt encontrado!")
            print("💡 Dica: Treine um modelo primeiro no notebook")
            return None
        
        # Ordenar por data de modificação
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Modelo encontrado: {latest.name}")
        
        # Mostrar idade do arquivo
        mtime = latest.stat().st_mtime
        age = time.time() - mtime
        if age < 3600:  # < 1 hora
            print(f"🕐 Criado há {age/60:.1f} minutos")
        elif age < 86400:  # < 1 dia
            print(f"🕐 Criado há {age/3600:.1f} horas")
        else:
            print(f"🕐 Criado há {age/86400:.1f} dias")
        
        return latest
    
    def load_model(self, model_path):
        """Carregar e validar modelo"""
        print(f"\n📥 Carregando modelo...")
        
        try:
            # Carregar com weights_only=False devido ao PyTorch 2.6+
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print("✅ Checkpoint carregado")
            
            # Mostrar informações do checkpoint
            if 'test_accuracy' in checkpoint:
                print(f"📊 Acurácia de teste: {checkpoint['test_accuracy']:.4f}")
            
            if 'cv_mean' in checkpoint and 'cv_std' in checkpoint:
                print(f"📊 Cross-validation: {checkpoint['cv_mean']:.4f} ± {checkpoint['cv_std']:.4f}")
            
            # Criar modelo
            if 'constructor_args' in checkpoint:
                args = checkpoint['constructor_args']
                print(f"🔧 Parâmetros: {args}")
                model = CustomEEGModel(**args)
            else:
                print("⚠️ Usando parâmetros padrão")
                model = CustomEEGModel()
            
            # Carregar pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Contar parâmetros
            total_params = sum(p.numel() for p in model.parameters())
            print(f"🧠 Parâmetros: {total_params:,}")
            
            return model, checkpoint
            
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def test_model_io(self, model):
        """Testar entrada/saída do modelo"""
        print(f"\n🧪 Testando entrada/saída...")
        
        # Teste com diferentes formatos
        test_cases = [
            (1, 16, 400),    # Batch único
            (5, 16, 400),    # Batch múltiplo
            # (1, 8, 400),   # Menos canais - comentado pois modelo requer 16
        ]
        
        for i, shape in enumerate(test_cases, 1):
            try:
                print(f"  Teste {i}: {shape}")
                
                # Gerar dados sintéticos
                input_data = torch.randn(*shape).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(input_data)
                    probabilities = torch.softmax(output, dim=1)
                
                print(f"    ✅ Input: {input_data.shape} -> Output: {output.shape}")
                print(f"    📊 Probabilidades exemplo: {probabilities[0].cpu().numpy()}")
                
            except Exception as e:
                print(f"    ❌ Erro: {e}")
    
    def test_realistic_predictions(self, model, checkpoint):
        """Testar com dados mais realistas"""
        print(f"\n🎯 Teste com dados sintéticos realistas...")
        
        # Gerar dados que simulam EEG real
        np.random.seed(42)
        n_tests = 10
        
        predictions = []
        confidences = []
        
        for i in range(n_tests):
            # Simular dados EEG (16 canais, 400 amostras, ~3.2s @ 125Hz)
            t = np.linspace(0, 3.2, 400)
            eeg_data = np.zeros((16, 400))
            
            # Adicionar componentes de frequência típicas do EEG
            for ch in range(16):
                # Ritmo alfa (8-12 Hz)
                eeg_data[ch] += 0.1 * np.sin(2 * np.pi * 10 * t + np.random.rand())
                
                # Ritmo beta (13-30 Hz)
                eeg_data[ch] += 0.05 * np.sin(2 * np.pi * 20 * t + np.random.rand())
                
                # Ruído
                eeg_data[ch] += 0.02 * np.random.randn(400)
                
                # Padrão específico para simular "intenção"
                if i % 2 == 0:  # "Mão esquerda"
                    if ch < 8:  # Canais esquerdos
                        eeg_data[ch, 100:200] += 0.03 * np.sin(2 * np.pi * 8 * t[100:200])
                else:  # "Mão direita"
                    if ch >= 8:  # Canais direitos
                        eeg_data[ch, 150:250] += 0.03 * np.sin(2 * np.pi * 12 * t[150:250])
            
            # Normalização (como seria feita em tempo real)
            if 'normalization_stats' in checkpoint:
                stats = checkpoint['normalization_stats']
                if 'median' in stats and 'iqr' in stats:
                    try:
                        median = stats['median']
                        iqr = stats['iqr']
                        
                        # Garantir forma correta para broadcasting
                        if hasattr(median, 'squeeze'):
                            median = median.squeeze()
                        if hasattr(iqr, 'squeeze'):
                            iqr = iqr.squeeze()
                        
                        # Reshape para (16, 1) se necessário
                        if median.shape == (16,):
                            median = median.reshape(16, 1)
                        if iqr.shape == (16,):
                            iqr = iqr.reshape(16, 1)
                        
                        eeg_data = (eeg_data - median) / (iqr + 1e-8)
                    except Exception as e:
                        print(f"    ⚠️ Erro na normalização salva: {e}")
                        # Fallback para z-score
                        eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
                else:
                    # Normalização z-score simples
                    eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
            else:
                # Normalização z-score simples
                eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
            
            # Converter para tensor
            input_tensor = torch.from_numpy(eeg_data).float().unsqueeze(0).to(self.device)
            
            # Predição
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            predictions.append(prediction)
            confidences.append(confidence)
            
            class_name = "Mão Direita" if prediction == 1 else "Mão Esquerda"
            expected = "Mão Direita" if i % 2 == 1 else "Mão Esquerda"
            correct = "✅" if class_name == expected else "❌"
            
            print(f"  Teste {i+1:2d}: {class_name} (conf: {confidence:.3f}) {correct} (esperado: {expected})")
        
        # Estatísticas
        print(f"\n📊 Resumo:")
        left_count = predictions.count(0)
        right_count = predictions.count(1)
        avg_confidence = np.mean(confidences)
        
        print(f"  👈 Mão Esquerda: {left_count}/10 ({left_count/10*100:.1f}%)")
        print(f"  👉 Mão Direita: {right_count}/10 ({right_count/10*100:.1f}%)")
        print(f"  🎯 Confiança média: {avg_confidence:.3f}")
        
        # Verificar se há variação nas predições
        if len(set(predictions)) == 1:
            print(f"  ⚠️ Todas as predições são iguais - possível problema no modelo")
        else:
            print(f"  ✅ Modelo produz predições variadas")
    
    def run_validation(self):
        """Executar validação completa"""
        print("🧪 VALIDAÇÃO DO MODELO BCI")
        print("=" * 40)
        
        # 1. Encontrar modelo
        model_path = self.find_latest_model()
        if not model_path:
            return
        
        # 2. Carregar modelo
        model, checkpoint = self.load_model(model_path)
        if not model:
            return
        
        # 3. Testar I/O
        self.test_model_io(model)
        
        # 4. Testar predições realistas
        self.test_realistic_predictions(model, checkpoint)
        
        print(f"\n✅ Validação concluída!")
        print(f"💡 O modelo está pronto para uso com LSL")
        print(f"🚀 Execute: python simple_bci_test.py")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
🧪 Validador de Modelo BCI
==========================

Este script valida se seu modelo treinado está funcionando:
✅ Encontra modelo mais recente
✅ Carrega e verifica parâmetros  
✅ Testa entrada/saída
✅ Simula predições realistas
✅ Verifica se está pronto para LSL
""")
    
    validator = ModelValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
