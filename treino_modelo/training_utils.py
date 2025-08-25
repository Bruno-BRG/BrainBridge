"""
Funções auxiliares para diferentes tipos de treinamento
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import os
import time
import random
from collections import defaultdict

# Importa as funções do módulo principal
from train_eeg_motor_imagery import (
    extract_segments_from_csv, 
    preprocess_data, 
    create_model,
    SAMPLE_RATE, CHANNELS, WINDOW_SIZE, OVERLAP, ACTIONS
)

def train_single_subject(subject_id: str, data_dir: str):
    """
    Treina modelo usando dados de um único sujeito
    """
    print(f"🎯 Treinamento para sujeito: {subject_id}")
    
    subject_path = os.path.join(data_dir, subject_id)
    if not os.path.exists(subject_path):
        raise ValueError(f"Sujeito {subject_id} não encontrado")
    
    # Carrega dados do sujeito específico
    all_segments = {"left": [], "right": []}
    
    csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
    
    print(f"📁 Processando {len(csv_files)} arquivos...")
    
    for i, file_name in enumerate(csv_files):
        file_path = os.path.join(subject_path, file_name)
        segments = extract_segments_from_csv(file_path)
        
        all_segments["left"].extend(segments["left"])
        all_segments["right"].extend(segments["right"])
        
        print(f"  ✓ {file_name}: {len(segments['left'])} esquerda, {len(segments['right'])} direita")
    
    total_left = len(all_segments["left"])
    total_right = len(all_segments["right"])
    
    if total_left == 0 or total_right == 0:
        raise ValueError(f"Dados insuficientes para {subject_id}")
    
    print(f"\n📊 Total de segmentos:")
    print(f"  Esquerda: {total_left}")
    print(f"  Direita: {total_right}")
    
    # Balanceia as classes
    min_samples = min(total_left, total_right)
    all_segments["left"] = all_segments["left"][:min_samples]
    all_segments["right"] = all_segments["right"][:min_samples]
    
    print(f"\n⚖️  Após balanceamento: {min_samples} amostras por classe")
    
    # Pré-processa os dados
    X, y = preprocess_data(all_segments)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔄 Divisão treino/teste:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Cria e treina o modelo
    model = create_model(input_shape=(WINDOW_SIZE, CHANNELS))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    print(f"\n🚀 Iniciando treinamento...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Avalia o modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Resultados finais:")
    print(f"  Acurácia: {test_accuracy:.4f}")
    print(f"  Perda: {test_loss:.4f}")
    
    # Salva o modelo
    timestamp = int(time.time())
    model_name = f"models/{subject_id}_eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.keras"
    
    os.makedirs("models", exist_ok=True)
    model.save(model_name)
    
    print(f"💾 Modelo salvo: {model_name}")
    
    return model, history, test_accuracy

def train_cross_validation_split(subjects: list, data_dir: str):
    """
    Treina modelo usando cross-validation 50/50
    """
    print(f"📚 Cross-validation com {len(subjects)} sujeitos")
    
    # Embaralha e divide os sujeitos
    subjects_copy = subjects.copy()
    random.shuffle(subjects_copy)
    
    split_point = len(subjects_copy) // 2
    train_subjects = subjects_copy[:split_point]
    test_subjects = subjects_copy[split_point:]
    
    print(f"\n🎯 Divisão dos sujeitos:")
    print(f"  Treino: {train_subjects}")
    print(f"  Teste: {test_subjects}")
    
    # Carrega dados de treino
    print(f"\n📊 Carregando dados de treino...")
    train_segments = {"left": [], "right": []}
    
    for subject in train_subjects:
        subject_path = os.path.join(data_dir, subject)
        if not os.path.exists(subject_path):
            continue
            
        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
        
        for file_name in csv_files:
            file_path = os.path.join(subject_path, file_name)
            segments = extract_segments_from_csv(file_path)
            
            train_segments["left"].extend(segments["left"])
            train_segments["right"].extend(segments["right"])
        
        print(f"  ✓ {subject}: {len([f for f in csv_files])} arquivos")
    
    # Carrega dados de teste
    print(f"\n🧪 Carregando dados de teste...")
    test_segments = {"left": [], "right": []}
    
    for subject in test_subjects:
        subject_path = os.path.join(data_dir, subject)
        if not os.path.exists(subject_path):
            continue
            
        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
        
        for file_name in csv_files:
            file_path = os.path.join(subject_path, file_name)
            segments = extract_segments_from_csv(file_path)
            
            test_segments["left"].extend(segments["left"])
            test_segments["right"].extend(segments["right"])
        
        print(f"  ✓ {subject}: {len([f for f in csv_files])} arquivos")
    
    # Balanceia as classes
    min_train = min(len(train_segments["left"]), len(train_segments["right"]))
    min_test = min(len(test_segments["left"]), len(test_segments["right"]))
    
    train_segments["left"] = train_segments["left"][:min_train]
    train_segments["right"] = train_segments["right"][:min_train]
    test_segments["left"] = test_segments["left"][:min_test]
    test_segments["right"] = test_segments["right"][:min_test]
    
    print(f"\n⚖️  Dados balanceados:")
    print(f"  Treino: {min_train} amostras por classe")
    print(f"  Teste: {min_test} amostras por classe")
    
    # Pré-processa os dados
    X_train, y_train = preprocess_data(train_segments)
    X_test, y_test = preprocess_data(test_segments)
    
    # Cria e treina o modelo
    model = create_model(input_shape=(WINDOW_SIZE, CHANNELS))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
    ]
    
    print(f"\n🚀 Iniciando treinamento cross-validation...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=150,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Avalia o modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Resultados Cross-Validation:")
    print(f"  Acurácia: {test_accuracy:.4f}")
    print(f"  Perda: {test_loss:.4f}")
    
    # Salva o modelo
    timestamp = int(time.time())
    model_name = f"models/cv_eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.keras"
    
    os.makedirs("models", exist_ok=True)
    model.save(model_name)
    
    print(f"💾 Modelo salvo: {model_name}")
    
    return model, history, test_accuracy

def train_leave_one_out_validation(subjects: list, data_dir: str):
    """
    Treina múltiplos modelos usando leave-one-out validation
    """
    print(f"🔄 Leave-One-Out Validation com {len(subjects)} sujeitos")
    
    results = []
    
    for i, test_subject in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"🎯 Iteração {i+1}/{len(subjects)} - Teste: {test_subject}")
        print(f"{'='*60}")
        
        # Define sujeitos de treino (todos exceto o de teste)
        train_subjects = [s for s in subjects if s != test_subject]
        
        print(f"📊 Sujeitos de treino: {train_subjects}")
        
        # Carrega dados de treino
        train_segments = {"left": [], "right": []}
        
        for subject in train_subjects:
            subject_path = os.path.join(data_dir, subject)
            if not os.path.exists(subject_path):
                continue
                
            csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
            
            for file_name in csv_files:
                file_path = os.path.join(subject_path, file_name)
                segments = extract_segments_from_csv(file_path)
                
                train_segments["left"].extend(segments["left"])
                train_segments["right"].extend(segments["right"])
        
        # Carrega dados de teste
        test_segments = {"left": [], "right": []}
        test_subject_path = os.path.join(data_dir, test_subject)
        
        if os.path.exists(test_subject_path):
            csv_files = [f for f in os.listdir(test_subject_path) if f.endswith('.csv') and not f.endswith('_backup.csv')]
            
            for file_name in csv_files:
                file_path = os.path.join(test_subject_path, file_name)
                segments = extract_segments_from_csv(file_path)
                
                test_segments["left"].extend(segments["left"])
                test_segments["right"].extend(segments["right"])
        
        # Verifica se há dados suficientes
        if (len(train_segments["left"]) == 0 or len(train_segments["right"]) == 0 or
            len(test_segments["left"]) == 0 or len(test_segments["right"]) == 0):
            print(f"⚠️  Dados insuficientes para {test_subject}, pulando...")
            continue
        
        # Balanceia as classes
        min_train = min(len(train_segments["left"]), len(train_segments["right"]))
        min_test = min(len(test_segments["left"]), len(test_segments["right"]))
        
        train_segments["left"] = train_segments["left"][:min_train]
        train_segments["right"] = train_segments["right"][:min_train]
        test_segments["left"] = test_segments["left"][:min_test]
        test_segments["right"] = test_segments["right"][:min_test]
        
        print(f"📊 Dados balanceados:")
        print(f"  Treino: {min_train} amostras por classe")
        print(f"  Teste: {min_test} amostras por classe")
        
        # Pré-processa os dados
        X_train, y_train = preprocess_data(train_segments)
        X_test, y_test = preprocess_data(test_segments)
        
        # Cria e treina o modelo
        model = create_model(input_shape=(WINDOW_SIZE, CHANNELS))
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        print(f"🚀 Treinando modelo para teste em {test_subject}...")
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.1,  # Usa 10% dos dados de treino para validação
            callbacks=callbacks,
            verbose=0  # Menos verboso para não poluir o output
        )
        
        # Avalia o modelo no sujeito de teste
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"📈 Resultado para {test_subject}:")
        print(f"  Acurácia: {test_accuracy:.4f}")
        print(f"  Perda: {test_loss:.4f}")
        
        # Salva resultado
        results.append({
            'test_subject': test_subject,
            'accuracy': test_accuracy,
            'loss': test_loss,
            'train_samples': min_train * 2,
            'test_samples': min_test * 2
        })
        
        # Salva o modelo
        timestamp = int(time.time())
        model_name = f"models/loo_{test_subject}_eeg_motor_imagery_{test_accuracy:.4f}acc_{timestamp}.keras"
        
        os.makedirs("models", exist_ok=True)
        model.save(model_name)
        
        print(f"💾 Modelo salvo: {model_name}")
    
    # Relatório final
    print(f"\n{'='*80}")
    print(f"📊 RELATÓRIO FINAL - LEAVE-ONE-OUT VALIDATION")
    print(f"{'='*80}")
    
    if results:
        accuracies = [r['accuracy'] for r in results]
        
        print(f"\n📈 Estatísticas de Acurácia:")
        print(f"  Média: {np.mean(accuracies):.4f}")
        print(f"  Desvio Padrão: {np.std(accuracies):.4f}")
        print(f"  Mínima: {np.min(accuracies):.4f}")
        print(f"  Máxima: {np.max(accuracies):.4f}")
        
        print(f"\n📋 Resultados por Sujeito:")
        for result in results:
            print(f"  {result['test_subject']}: {result['accuracy']:.4f}")
    
    return results

def compare_all_models(models_dir: str, model_files: list):
    """
    Compara performance de todos os modelos
    """
    print(f"📈 Comparando {len(model_files)} modelos...")
    
    data_dir = "c:/Users/Chari/Documents/CIMATEC/BrainBridge/resources/eeg_data"
    
    # Carrega dados de teste (usando uma amostra pequena)
    from train_eeg_motor_imagery import load_all_data
    
    print(f"📊 Carregando dados de teste...")
    all_segments = load_all_data(data_dir)
    
    # Usa apenas uma pequena amostra para comparação rápida
    sample_size = min(100, len(all_segments["left"]), len(all_segments["right"]))
    
    test_segments = {
        "left": all_segments["left"][:sample_size],
        "right": all_segments["right"][:sample_size]
    }
    
    X_test, y_test = preprocess_data(test_segments)
    
    print(f"🧪 Dados de teste: {X_test.shape[0]} amostras")
    
    results = []
    
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        
        print(f"\n🔍 Testando modelo {i+1}/{len(model_files)}: {model_file}")
        
        try:
            # Carrega o modelo
            model = tf.keras.models.load_model(model_path)
            
            # Avalia o modelo
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Extrai informações do nome do arquivo
            model_info = {
                'file': model_file,
                'accuracy': test_accuracy,
                'loss': test_loss,
                'params': model.count_params(),
                'size_mb': os.path.getsize(model_path) / (1024 * 1024)
            }
            
            results.append(model_info)
            
            print(f"  ✓ Acurácia: {test_accuracy:.4f}")
            print(f"  ✓ Perda: {test_loss:.4f}")
            print(f"  ✓ Parâmetros: {model_info['params']:,}")
            print(f"  ✓ Tamanho: {model_info['size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"  ❌ Erro ao carregar modelo: {str(e)}")
    
    # Relatório de comparação
    if results:
        print(f"\n{'='*80}")
        print(f"📊 RELATÓRIO DE COMPARAÇÃO DE MODELOS")
        print(f"{'='*80}")
        
        # Ordena por acurácia (melhor primeiro)
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n🏆 Ranking por Acurácia:")
        print(f"{'Pos':<4} {'Arquivo':<40} {'Acurácia':<10} {'Perda':<10} {'Tamanho':<10}")
        print(f"{'-'*80}")
        
        for i, result in enumerate(results, 1):
            print(f"{i:<4} {result['file'][:40]:<40} {result['accuracy']:<10.4f} "
                  f"{result['loss']:<10.4f} {result['size_mb']:<10.2f}")
        
        # Estatísticas
        accuracies = [r['accuracy'] for r in results]
        
        print(f"\n📈 Estatísticas Gerais:")
        print(f"  Melhor acurácia: {max(accuracies):.4f}")
        print(f"  Pior acurácia: {min(accuracies):.4f}")
        print(f"  Acurácia média: {np.mean(accuracies):.4f}")
        print(f"  Desvio padrão: {np.std(accuracies):.4f}")
    
    return results
