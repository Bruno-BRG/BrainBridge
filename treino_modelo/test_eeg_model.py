import numpy as np
import pandas as pd
import tensorflow as tf
import os


def load_and_preprocess_realtime_data(csv_file_path, window_size=250):
    """
    Carrega e pré-processa dados EEG de um arquivo CSV para inferência em tempo real
    """
    try:
        # Lê o arquivo CSV
        df = pd.read_csv(csv_file_path, skiprows=4)
        
        # Extrai apenas os canais de EEG (colunas 1-16)
        eeg_data = df.iloc[:, 1:17].values.astype(float)
        
        # Pega a janela mais recente de dados
        if len(eeg_data) >= window_size:
            window = eeg_data[-window_size:]
            return window.reshape(1, window_size, 16)  # Shape para o modelo
        else:
            # Se não há dados suficientes, preenche com zeros
            window = np.zeros((window_size, 16))
            if len(eeg_data) > 0:
                window[:len(eeg_data)] = eeg_data
            return window.reshape(1, window_size, 16)
            
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return None

def predict_motor_imagery(model_path, data_window):
    """
    Faz predição de imagética motora usando o modelo treinado
    """
    try:
        # Carrega o modelo
        model = tf.keras.models.load_model(model_path)
        
        # Faz a predição
        prediction = model.predict(data_window, verbose=0)
        
        # Interpreta o resultado
        left_prob = prediction[0][0]
        right_prob = prediction[0][1]
        
        if left_prob > right_prob:
            predicted_class = "left"
            confidence = left_prob
        else:
            predicted_class = "right"
            confidence = right_prob
        
        return predicted_class, confidence, prediction[0]
        
    except Exception as e:
        print(f"Erro na predição: {str(e)}")
        return None, None, None

def test_model_on_file(model_path, csv_file_path):
    """
    Testa o modelo em um arquivo CSV específico
    """
    print(f"Testando modelo em: {csv_file_path}")
    
    # Carrega e pré-processa os dados
    data_window = load_and_preprocess_realtime_data(csv_file_path)
    
    if data_window is not None:
        # Faz a predição
        predicted_class, confidence, probabilities = predict_motor_imagery(model_path, data_window)
        
        if predicted_class is not None:
            print(f"Predição: {predicted_class}")
            print(f"Confiança: {confidence:.4f}")
            print(f"Probabilidades [esquerda, direita]: [{probabilities[0]:.4f}, {probabilities[1]:.4f}]")
            return predicted_class, confidence
        else:
            print("Erro na predição")
            return None, None
    else:
        print("Erro no carregamento dos dados")
        return None, None

def batch_test_model(model_path, data_dir, num_files=5):
    """
    Testa o modelo em múltiplos arquivos para avaliar a performance
    """
    print(f"Testando modelo em batch...")
    print(f"Modelo: {model_path}")
    print(f"Diretório de dados: {data_dir}")
    
    results = []
    
    # Lista todos os sujeitos
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    count = 0
    for subject in subjects[:num_files]:
        subject_path = os.path.join(data_dir, subject)
        
        # Pega o primeiro arquivo CSV do sujeito
        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
        
        if csv_files:
            csv_file = csv_files[0]
            csv_path = os.path.join(subject_path, csv_file)
            
            print(f"\n--- Testando {subject}/{csv_file} ---")
            predicted_class, confidence = test_model_on_file(model_path, csv_path)
            
            if predicted_class is not None:
                results.append({
                    'subject': subject,
                    'file': csv_file,
                    'prediction': predicted_class,
                    'confidence': confidence
                })
            
            count += 1
            if count >= num_files:
                break
    
    # Mostra resumo dos resultados
    if results:
        print(f"\n=== RESUMO DOS TESTES ===")
        left_predictions = sum(1 for r in results if r['prediction'] == 'left')
        right_predictions = sum(1 for r in results if r['prediction'] == 'right')
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"Total de testes: {len(results)}")
        print(f"Predições 'esquerda': {left_predictions}")
        print(f"Predições 'direita': {right_predictions}")
        print(f"Confiança média: {avg_confidence:.4f}")
        
        print(f"\nDetalhes:")
        for result in results:
            print(f"  {result['subject']}: {result['prediction']} (conf: {result['confidence']:.4f})")
    
    return results

def main():
    # Configurações
    model_path = "models/"  # Diretório onde estão os modelos
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    
    # Encontra o modelo mais recente
    if os.path.exists(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.keras') or f.endswith('.h5') or f.endswith('.model')]
        if model_files:
            # Ordena por data de modificação (mais recente primeiro)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
            latest_model = os.path.join(model_path, model_files[0])
            
            print(f"Usando modelo: {latest_model}")
            
            # Testa em batch
            batch_test_model(latest_model, data_dir, num_files=10)
            
        else:
            print("Nenhum modelo encontrado no diretório models/")
    else:
        print("Diretório models/ não encontrado")

if __name__ == "__main__":
    main()
