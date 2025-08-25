"""
Exemplo de integração do classificador EEG com o sistema BCI existente
Este arquivo mostra como modificar o sistema existente para usar o novo modelo
"""

import sys
import os
import numpy as np

# Adiciona o caminho do treino_modelo ao Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'treino_modelo'))

from eeg_classifier_integration import EEGMotorImageryClassifier, create_classifier_from_models_dir


class BCIMotorImageryIntegration:
    """
    Classe para integrar o classificador de imagética motora ao sistema BCI
    """
    
    def __init__(self):
        self.classifier = None
        self.is_active = False
        self.last_prediction = {'class': 'none', 'confidence': 0.0}
        
        # Configurações
        self.min_confidence = 0.65  # Confiança mínima para ação
        self.prediction_interval = 25  # Predição a cada N amostras (200ms a 125Hz)
        self.sample_count = 0
        
        # Inicializa o classificador
        self.setup_classifier()
    
    def setup_classifier(self):
        """Configura o classificador de imagética motora"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'treino_modelo', 'models')
            self.classifier = create_classifier_from_models_dir(models_dir)
            
            if self.classifier:
                print("✓ Classificador de imagética motora carregado com sucesso!")
                self.is_active = True
            else:
                print("⚠ Não foi possível carregar o classificador. Funcionando sem imagética motora.")
                self.is_active = False
                
        except Exception as e:
            print(f"⚠ Erro ao configurar classificador: {e}")
            self.is_active = False
    
    def process_eeg_sample(self, eeg_data):
        """
        Processa uma amostra de EEG para detecção de imagética motora
        
        Args:
            eeg_data: Array com os dados EEG (espera-se 16 canais)
        
        Returns:
            dict: Resultado da classificação ou None
        """
        if not self.is_active or self.classifier is None:
            return None
        
        try:
            # Extrai apenas os primeiros 16 canais (EEG)
            if len(eeg_data) >= 16:
                eeg_channels = eeg_data[:16]
            else:
                # Se não há dados suficientes, preenche com zeros
                eeg_channels = np.zeros(16)
                eeg_channels[:len(eeg_data)] = eeg_data
            
            # Adiciona a amostra ao classificador
            self.classifier.add_sample(eeg_channels)
            self.sample_count += 1
            
            # Faz predição em intervalos regulares
            if self.sample_count % self.prediction_interval == 0:
                # Predição instantânea
                instant_result = self.classifier.predict()
                
                # Predição estável (baseada no histórico)
                stable_result = self.classifier.get_stable_prediction(
                    min_confidence=self.min_confidence,
                    min_consensus=0.7
                )
                
                self.last_prediction = stable_result
                
                # Retorna resultado se há confiança suficiente
                if stable_result['confidence'] >= self.min_confidence:
                    return {
                        'type': 'motor_imagery',
                        'class': stable_result['class'],
                        'confidence': stable_result['confidence'],
                        'instant': instant_result,
                        'stable': stable_result
                    }
            
            return None
            
        except Exception as e:
            print(f"Erro no processamento EEG: {e}")
            return None
    
    def get_unity_command(self, classification_result):
        """
        Converte resultado da classificação em comando para Unity
        
        Args:
            classification_result: Resultado do process_eeg_sample
        
        Returns:
            str: Comando para enviar ao Unity
        """
        if not classification_result:
            return "NONE"
        
        movement_class = classification_result['class']
        confidence = classification_result['confidence']
        
        # Mapeia classes para comandos Unity
        command_map = {
            'left': 'MOVE_LEFT',
            'right': 'MOVE_RIGHT',
            'none': 'NONE'
        }
        
        command = command_map.get(movement_class, 'NONE')
        
        # Adiciona informação de confiança
        if command != 'NONE':
            command += f":{confidence:.2f}"
        
        return command
    
    def reset(self):
        """Reseta o estado do classificador"""
        if self.classifier:
            self.classifier.reset_buffer()
        self.sample_count = 0
        self.last_prediction = {'class': 'none', 'confidence': 0.0}
    
    def get_status(self):
        """Retorna status atual do sistema"""
        return {
            'active': self.is_active,
            'last_prediction': self.last_prediction,
            'sample_count': self.sample_count,
            'classifier_loaded': self.classifier is not None
        }


# Exemplo de como modificar o udp_receiver_BCI.py existente
def example_integration_with_udp_receiver():
    """
    Exemplo de como integrar com o udp_receiver_BCI.py existente
    """
    
    # Código que seria adicionado ao udp_receiver_BCI.py
    print("=== EXEMPLO DE INTEGRAÇÃO ===")
    print()
    
    print("1. No início do arquivo udp_receiver_BCI.py, adicione:")
    print("   from bci.integration.motor_imagery_integration import BCIMotorImageryIntegration")
    print()
    
    print("2. Na inicialização da classe, adicione:")
    print("   self.motor_imagery = BCIMotorImageryIntegration()")
    print()
    
    print("3. No método de processamento de dados EEG, adicione:")
    print("""
   def process_eeg_data(self, sample_data):
       # ... código existente ...
       
       # Adiciona processamento de imagética motora
       mi_result = self.motor_imagery.process_eeg_sample(sample_data)
       
       if mi_result:
           unity_command = self.motor_imagery.get_unity_command(mi_result)
           print(f"Imagética Motora detectada: {mi_result['class']} "
                 f"(conf: {mi_result['confidence']:.3f}) -> {unity_command}")
           
           # Envia comando para Unity
           self.send_to_unity(unity_command)
       
       # ... resto do código ...
    """)
    print()
    
    print("4. Para resetar entre sessões:")
    print("   self.motor_imagery.reset()")
    print()
    
    print("5. Para verificar status:")
    print("   status = self.motor_imagery.get_status()")
    print("   print(f'Motor Imagery Status: {status}')")


if __name__ == "__main__":
    example_integration_with_udp_receiver()
