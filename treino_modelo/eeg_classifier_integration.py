"""
Integração do modelo EEG Motor Imagery com o sistema BCI existente
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time
from collections import deque


class EEGMotorImageryClassifier:
    def __init__(self, model_path=None, window_size=250, channels=16):
        """
        Inicializa o classificador de imagética motora
        
        Args:
            model_path: Caminho para o modelo treinado (.keras ou .h5)
            window_size: Tamanho da janela em amostras (250 = 2 segundos a 125Hz)
            channels: Número de canais EEG
        """
        self.window_size = window_size
        self.channels = channels
        self.model = None
        self.data_buffer = deque(maxlen=window_size)
        self.predictions_history = deque(maxlen=10)  # Últimas 10 predições
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def load_model(self, model_path):
        """Carrega o modelo treinado"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Modelo carregado: {model_path}")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def add_sample(self, eeg_sample):
        """
        Adiciona uma nova amostra EEG ao buffer
        
        Args:
            eeg_sample: Array com 16 valores (canais EEG)
        """
        if len(eeg_sample) != self.channels:
            raise ValueError(f"Esperado {self.channels} canais, recebido {len(eeg_sample)}")
        
        self.data_buffer.append(eeg_sample)
    
    def predict(self):
        """
        Faz predição baseada nos dados atuais do buffer
        
        Returns:
            dict: {'class': 'left'|'right', 'confidence': float, 'probabilities': [left_prob, right_prob]}
        """
        if self.model is None:
            return {'class': 'none', 'confidence': 0.0, 'probabilities': [0.5, 0.5]}
        
        if len(self.data_buffer) < self.window_size:
            return {'class': 'none', 'confidence': 0.0, 'probabilities': [0.5, 0.5]}
        
        try:
            # Prepara os dados para predição
            data_window = np.array(list(self.data_buffer))
            data_window = data_window.reshape(1, self.window_size, self.channels)
            
            # Faz a predição
            prediction = self.model.predict(data_window, verbose=0)
            
            left_prob = float(prediction[0][0])
            right_prob = float(prediction[0][1])
            
            # Determina a classe
            if left_prob > right_prob:
                predicted_class = 'left'
                confidence = left_prob
            else:
                predicted_class = 'right'
                confidence = right_prob
            
            # Adiciona à história
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': [left_prob, right_prob]
            }
            
            self.predictions_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return {'class': 'none', 'confidence': 0.0, 'probabilities': [0.5, 0.5]}
    
    def get_stable_prediction(self, min_confidence=0.6, min_consensus=0.7):
        """
        Retorna uma predição estável baseada no histórico
        
        Args:
            min_confidence: Confiança mínima para considerar uma predição
            min_consensus: Porcentagem mínima de acordo nas últimas predições
        
        Returns:
            dict: Predição estável ou 'none'
        """
        if len(self.predictions_history) < 5:
            return {'class': 'none', 'confidence': 0.0}
        
        # Filtra predições com confiança suficiente
        confident_predictions = [p for p in self.predictions_history if p['confidence'] >= min_confidence]
        
        if len(confident_predictions) < 3:
            return {'class': 'none', 'confidence': 0.0}
        
        # Conta votos para cada classe
        left_votes = sum(1 for p in confident_predictions if p['class'] == 'left')
        right_votes = sum(1 for p in confident_predictions if p['class'] == 'right')
        total_votes = left_votes + right_votes
        
        # Verifica consenso
        if total_votes == 0:
            return {'class': 'none', 'confidence': 0.0}
        
        if left_votes / total_votes >= min_consensus:
            avg_confidence = np.mean([p['confidence'] for p in confident_predictions if p['class'] == 'left'])
            return {'class': 'left', 'confidence': avg_confidence}
        elif right_votes / total_votes >= min_consensus:
            avg_confidence = np.mean([p['confidence'] for p in confident_predictions if p['class'] == 'right'])
            return {'class': 'right', 'confidence': avg_confidence}
        else:
            return {'class': 'none', 'confidence': 0.0}
    
    def reset_buffer(self):
        """Limpa o buffer de dados"""
        self.data_buffer.clear()
        self.predictions_history.clear()


# Função auxiliar para integração com o sistema existente
def create_classifier_from_models_dir(models_dir="models"):
    """
    Cria um classificador usando o modelo mais recente do diretório
    
    Args:
        models_dir: Diretório onde estão os modelos
    
    Returns:
        EEGMotorImageryClassifier ou None se não encontrar modelo
    """
    if not os.path.exists(models_dir):
        print(f"Diretório {models_dir} não encontrado")
        return None
    
    # Encontra modelos
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5', '.model'))]
    
    if not model_files:
        print("Nenhum modelo encontrado")
        return None
    
    # Ordena por data de modificação (mais recente primeiro)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_model = os.path.join(models_dir, model_files[0])
    
    print(f"Usando modelo: {latest_model}")
    
    classifier = EEGMotorImageryClassifier()
    if classifier.load_model(latest_model):
        return classifier
    else:
        return None


# Exemplo de uso
def example_usage():
    """Exemplo de como usar o classificador"""
    
    # Cria o classificador
    classifier = create_classifier_from_models_dir()
    
    if classifier is None:
        print("Não foi possível criar o classificador")
        return
    
    print("Classificador criado com sucesso!")
    print("Simulando dados EEG...")
    
    # Simula chegada de dados EEG (normalmente viria do OpenBCI)
    for i in range(300):  # 300 amostras = ~2.4 segundos a 125Hz
        # Simula uma amostra EEG (16 canais)
        eeg_sample = np.random.randn(16) * 50  # Valores típicos de EEG em microvolts
        
        # Adiciona a amostra
        classifier.add_sample(eeg_sample)
        
        # Faz predição a cada 25 amostras (~0.2 segundos)
        if i % 25 == 0 and i > 0:
            result = classifier.predict()
            stable_result = classifier.get_stable_prediction()
            
            print(f"Amostra {i:3d}: {result['class']:>5} (conf: {result['confidence']:.3f}) | "
                  f"Estável: {stable_result['class']:>5} (conf: {stable_result['confidence']:.3f})")


if __name__ == "__main__":
    example_usage()
