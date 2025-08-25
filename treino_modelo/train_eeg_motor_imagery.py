import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import random
import time
from collections import defaultdict


# Configurações
SAMPLE_RATE = 125  # Hz conforme especificado nos dados
CHANNELS = 16  # Canais de EEG (0-15)
WINDOW_SIZE = int(2.0 * SAMPLE_RATE)  # 2 segundos de dados por janela
OVERLAP = int(0.5 * SAMPLE_RATE)  # 50% de sobreposição
ACTIONS = ["left", "right"]

def extract_segments_from_csv(file_path):
    """
    Extrai segmentos de dados EEG baseados nos marcadores T1, T2, T0
    T1 -> T0: movimento à esquerda
    T2 -> T0: movimento à direita
    """
    try:
        # Lê o arquivo CSV pulando as linhas de cabeçalho
        df = pd.read_csv(file_path, skiprows=4)
        
        # Extrai apenas os canais de EEG (colunas 1-16)
        eeg_data = df.iloc[:, 1:17].values.astype(float)
        
        # Extrai as anotações (última coluna)
        annotations = df.iloc[:, -1].values
        
        segments = {"left": [], "right": []}
        
        # Encontra os índices dos marcadores
        marker_indices = {}
        for i, annotation in enumerate(annotations):
            if annotation in ['T0', 'T1', 'T2']:
                if annotation not in marker_indices:
                    marker_indices[annotation] = []
                marker_indices[annotation].append(i)
        
        print(f"Marcadores encontrados em {file_path}:")
        for marker, indices in marker_indices.items():
            print(f"  {marker}: {len(indices)} ocorrências")
        
        # Processa sequências T1->T0 (esquerda) e T2->T0 (direita)
        all_t0_indices = marker_indices.get('T0', [])
        
        # Para movimentos à esquerda (T1 -> T0)
        if 'T1' in marker_indices:
            for t1_idx in marker_indices['T1']:
                # Encontra o próximo T0 após T1
                next_t0 = None
                for t0_idx in all_t0_indices:
                    if t0_idx > t1_idx:
                        next_t0 = t0_idx
                        break
                
                if next_t0 is not None:
                    # Extrai o segmento entre T1 e T0
                    segment_data = eeg_data[t1_idx:next_t0]
                    
                    # Divide em janelas se o segmento for longo o suficiente
                    if len(segment_data) >= WINDOW_SIZE:
                        for start_idx in range(0, len(segment_data) - WINDOW_SIZE + 1, OVERLAP):
                            window = segment_data[start_idx:start_idx + WINDOW_SIZE]
                            segments["left"].append(window)
        
        # Para movimentos à direita (T2 -> T0)
        if 'T2' in marker_indices:
            for t2_idx in marker_indices['T2']:
                # Encontra o próximo T0 após T2
                next_t0 = None
                for t0_idx in all_t0_indices:
                    if t0_idx > t2_idx:
                        next_t0 = t0_idx
                        break
                
                if next_t0 is not None:
                    # Extrai o segmento entre T2 e T0
                    segment_data = eeg_data[t2_idx:next_t0]
                    
                    # Divide em janelas se o segmento for longo o suficiente
                    if len(segment_data) >= WINDOW_SIZE:
                        for start_idx in range(0, len(segment_data) - WINDOW_SIZE + 1, OVERLAP):
                            window = segment_data[start_idx:start_idx + WINDOW_SIZE]
                            segments["right"].append(window)
        
        print(f"Segmentos extraídos de {file_path}:")
        print(f"  Esquerda: {len(segments['left'])}")
        print(f"  Direita: {len(segments['right'])}")
        
        return segments
        
    except Exception as e:
        print(f"Erro ao processar {file_path}: {str(e)}")
        return {"left": [], "right": []}

def load_all_data(data_dir):
    """
    Carrega todos os dados de EEG do diretório especificado
    """
    all_segments = {"left": [], "right": []}
    
    # Percorre todos os sujeitos
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)
        
        if os.path.isdir(subject_path):
            print(f"\nProcessando sujeito: {subject_folder}")
            
            # Processa todos os arquivos CSV do sujeito
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.csv') and not file_name.endswith('_backup.csv'):
                    file_path = os.path.join(subject_path, file_name)
                    segments = extract_segments_from_csv(file_path)
                    
                    # Adiciona aos dados totais
                    all_segments["left"].extend(segments["left"])
                    all_segments["right"].extend(segments["right"])
    
    return all_segments

def preprocess_data(segments):
    """
    Pré-processa os dados: normalização e criação de X, y
    """
    X = []
    y = []
    
    # Adiciona dados de movimentos à esquerda
    for segment in segments["left"]:
        X.append(segment)
        y.append([1, 0])  # [esquerda, direita]
    
    # Adiciona dados de movimentos à direita
    for segment in segments["right"]:
        X.append(segment)
        y.append([0, 1])  # [esquerda, direita]
    
    # Converte para arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Embaralha os dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\nDados finais:")
    print(f"  Shape de X: {X.shape}")
    print(f"  Shape de y: {y.shape}")
    print(f"  Total de amostras: {len(X)}")
    print(f"  Distribuição de classes: Esquerda={np.sum(y[:, 0])}, Direita={np.sum(y[:, 1])}")
    
    return X, y

def create_model(input_shape):
    """
    Cria o modelo de rede neural convolucional
    """
    model = Sequential()
    
    # Primeira camada convolucional
    model.add(Conv1D(64, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # Segunda camada convolucional
    model.add(Conv1D(64, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    # Terceira camada convolucional
    model.add(Conv1D(128, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    # Camadas densas
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Camada de saída
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Caminho para os dados EEG
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    
    print("Carregando dados EEG...")
    segments = load_all_data(data_dir)
    
    # Verifica se temos dados suficientes
    total_left = len(segments["left"])
    total_right = len(segments["right"])
    
    if total_left == 0 or total_right == 0:
        print("Erro: Não foram encontrados dados suficientes para ambas as classes!")
        return
    
    print(f"\nTotal de segmentos carregados:")
    print(f"  Esquerda: {total_left}")
    print(f"  Direita: {total_right}")
    
    # Balanceia as classes
    min_samples = min(total_left, total_right)
    segments["left"] = segments["left"][:min_samples]
    segments["right"] = segments["right"][:min_samples]
    
    print(f"\nApós balanceamento:")
    print(f"  Esquerda: {len(segments['left'])}")
    print(f"  Direita: {len(segments['right'])}")
    
    # Pré-processa os dados
    X, y = preprocess_data(segments)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDivisão treino/teste:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Cria o modelo
    print("\nCriando modelo...")
    model = create_model(input_shape=(WINDOW_SIZE, CHANNELS))
    
    # Mostra o resumo do modelo
    model.summary()
    
    # Treinamento
    print("\nIniciando treinamento...")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5
        )
    ]
    
    # Treina o modelo
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Avalia o modelo
    print("\nAvaliando modelo...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    print(f"Perda no teste: {test_loss:.4f}")
    
    # Salva o modelo
    timestamp = int(time.time())
    model_name = f"models/eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.keras"
    
    # Cria o diretório se não existir
    os.makedirs("models", exist_ok=True)
    
    model.save(model_name)
    print(f"\nModelo salvo em: {model_name}")
    
    # Também salva um modelo no formato .h5 para compatibilidade
    h5_model_name = f"models/eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.h5"
    model.save(h5_model_name)
    print(f"Modelo também salvo em formato H5: {h5_model_name}")
    
    return model, history

if __name__ == "__main__":
    main()
