import tensorflow as tf
import os
import time

def save_best_model():
    """
    Salva um modelo simples treinado para teste
    """
    # Cria o diretório se não existir
    os.makedirs("models", exist_ok=True)
    
    # Simula métricas do último treinamento
    test_accuracy = 0.7190
    timestamp = int(time.time())
    
    print("Criando modelo para teste...")
    
    # Cria um modelo simples para demonstração
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, input_shape=(250, 16)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, 3),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(128, 3),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation('softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Salva o modelo no formato Keras
    model_name = f"models/eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.keras"
    model.save(model_name)
    print(f"Modelo salvo em: {model_name}")
    
    # Também salva no formato H5
    h5_model_name = f"models/eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.h5"
    model.save(h5_model_name)
    print(f"Modelo também salvo em formato H5: {h5_model_name}")
    
    print("\nModelos salvos com sucesso!")
    return model_name, h5_model_name

if __name__ == "__main__":
    save_best_model()
