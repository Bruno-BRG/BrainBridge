import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Input
import os
import random
import time


ACTIONS = ["T0", "T1", "T2"]  # T0 = repouso, T1 = mão esquerda, T2 = mão direita
reshape = (-1, 16, 125)  # Ajustado para 125 amostras

def get_patient_from_filename(filename):
    # Extrair o ID do paciente do nome do arquivo (S001, S002, etc)
    return filename.split('_')[0]

def create_data_by_patient(starting_dir="processed_eeg_data"):
    data_by_patient = {}
    
    # Primeiro, vamos organizar os dados por paciente
    for action in ACTIONS:
        data_dir = os.path.join(starting_dir, action)
        for subject_dir in os.listdir(data_dir):
            if subject_dir not in data_by_patient:
                data_by_patient[subject_dir] = {action: [] for action in ACTIONS}
            
            subject_path = os.path.join(data_dir, subject_dir)
            if os.path.isdir(subject_path):
                for item in os.listdir(subject_path):
                    if item.endswith('.npy'):
                        data = np.load(os.path.join(subject_path, item))
                        data_by_patient[subject_dir][action].append(data)
    
    return data_by_patient

def create_model():
    model = Sequential([
        Input(shape=(16, 125)),
        
        # Primeiro bloco convolucional
        Conv1D(64, 10, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=(2)),
        
        # Segundo bloco convolucional
        Conv1D(128, 5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=(2)),
        
        # Terceiro bloco convolucional
        Conv1D(128, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=(2)),
        
        # Quarto bloco convolucional
        Conv1D(256, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=(2)),
        
        Flatten(),
        
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(3),
        Activation('softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    return model

# Carregando dados organizados por paciente
print("Carregando e preparando os dados...")
data_by_patient = create_data_by_patient()
patients = list(data_by_patient.keys())
print(f"Total de pacientes: {len(patients)}")

# Configurações de treinamento
epochs = 50
batch_size = 32
results = []

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Leave-one-out cross validation por paciente
for test_patient in patients:
    print(f"\n{'='*50}")
    print(f"Testando no paciente {test_patient}")
    print(f"{'='*50}")
    
    # Preparar dados de treino (todos os pacientes exceto o de teste)
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    
    # Dados de teste (apenas do paciente atual)
    for action_idx, action in enumerate(ACTIONS):
        for data in data_by_patient[test_patient][action]:
            test_X.append(data)
            y = [0] * len(ACTIONS)
            y[action_idx] = 1
            test_y.append(y)
    
    # Dados de treino (todos os outros pacientes)
    for patient in patients:
        if patient != test_patient:  # Excluir paciente de teste
            for action_idx, action in enumerate(ACTIONS):
                for data in data_by_patient[patient][action]:
                    train_X.append(data)
                    y = [0] * len(ACTIONS)
                    y[action_idx] = 1
                    train_y.append(y)
    
    # Converter para arrays e reshape
    train_X = np.array(train_X).reshape(reshape)
    test_X = np.array(test_X).reshape(reshape)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    
    print(f"Amostras de treino: {len(train_X)}")
    print(f"Amostras de teste: {len(test_X)}")
    
    # Criar e treinar o modelo
    model = create_model()
    
    print(f"\nTreinando modelo para o paciente {test_patient}...")
    history = model.fit(
        train_X, train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_X, test_y),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Avaliar o modelo
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    print(f"\nResultados para {test_patient}:")
    print(f"Acurácia: {score[1]*100:.2f}%")
    
    # Salvar o modelo
    if not os.path.exists("models"):
        os.makedirs("models")
    
    MODEL_NAME = f"models/{test_patient}-{round(score[1]*100,2)}-acc-loss-{round(score[0],2)}.keras"
    model.save(MODEL_NAME)
    print(f"Modelo salvo: {MODEL_NAME}")
    
    # Guardar resultados
    results.append({
        'patient': test_patient,
        'accuracy': score[1],
        'loss': score[0]
    })

# Imprimir resultados finais
print("\n" + "="*50)
print("RESULTADOS FINAIS")
print("="*50)
accuracies = [r['accuracy'] for r in results]
print(f"Acurácia média: {np.mean(accuracies)*100:.2f}% (±{np.std(accuracies)*100:.2f}%)")
print("\nResultados por paciente:")
for r in results:
    print(f"{r['patient']}: {r['accuracy']*100:.2f}%")
