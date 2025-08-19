import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


MODEL_NAME = "models/S007-86.67-acc-loss-0.37.keras"  # Melhor modelo com 86.7% de acurácia

CLIP = False  # Desabilitado inicialmente
CLIP_VAL = 10  # valor do clip, se usado

model = tf.keras.models.load_model(MODEL_NAME)

VALDIR = 'processed_eeg_data'  # Diretório com os dados processados
ACTIONS = ['T0', 'T1', 'T2']  # T0 = repouso, T1 = mão esquerda, T2 = mão direita
PRED_BATCH = 32


def get_val_data(valdir, action, batch_size):
    argmax_dict = {0: 0, 1: 0, 2: 0}
    raw_pred_dict = {0: 0, 1: 0, 2: 0}

    action_dir = os.path.join(valdir, action)
    for subject_dir in os.listdir(action_dir):
        subject_path = os.path.join(action_dir, subject_dir)
        if os.path.isdir(subject_path):
            for session_file in os.listdir(subject_path):
                if session_file.endswith('.npy'):
                    filepath = os.path.join(subject_path, session_file)
                    data = np.load(filepath)
                    
                    if CLIP:
                        data = np.clip(data, -CLIP_VAL, CLIP_VAL) / CLIP_VAL

                    preds = model.predict([data.reshape(-1, 16, 125)], batch_size=batch_size)

                    for pred in preds:
                        argmax = np.argmax(pred)
                        argmax_dict[argmax] += 1
                        for idx, value in enumerate(pred):
                            raw_pred_dict[idx] += value

    argmax_pct_dict = {}

    for i in argmax_dict:
        total = 0
        correct = argmax_dict[i]
        for ii in argmax_dict:
            total += argmax_dict[ii]

        argmax_pct_dict[i] = round(correct/total, 3)

    return argmax_dict, raw_pred_dict, argmax_pct_dict


def make_conf_mat(rest, left, right):
    action_dict = {"Repouso (T0)": rest, "Mão Esquerda (T1)": left, "Mão Direita (T2)": right}
    action_conf_mat = pd.DataFrame(action_dict)
    actions = [i for i in action_dict]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.matshow(action_conf_mat, cmap=plt.cm.RdYlGn)
    plt.colorbar(im)
    
    ax.set_xticklabels([""]+actions, rotation=45, ha='left')
    ax.set_yticklabels([""]+actions)

    print("__________")
    print(action_dict)
    for idx, i in enumerate(action_dict):
        for idx2, ii in enumerate(action_dict[i]):
            ax.text(idx, idx2, f"{round(float(action_dict[i][ii]),2)}", 
                   va='center', ha='center',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title("Matriz de Confusão - Ações Imaginadas")
    plt.ylabel("Ação Prevista")
    plt.xlabel("Ação Real")
    plt.tight_layout()
    plt.show()


# Obter dados de validação para cada classe
t0_argmax_dict, t0_raw_pred_dict, t0_argmax_pct_dict = get_val_data(VALDIR, "T0", PRED_BATCH)
t1_argmax_dict, t1_raw_pred_dict, t1_argmax_pct_dict = get_val_data(VALDIR, "T1", PRED_BATCH)
t2_argmax_dict, t2_raw_pred_dict, t2_argmax_pct_dict = get_val_data(VALDIR, "T2", PRED_BATCH)

# Criar matriz de confusão
make_conf_mat(t0_argmax_pct_dict, t1_argmax_pct_dict, t2_argmax_pct_dict)

# Imprimir acurácias individuais
print("\nAcurácias por classe:")
print(f"Repouso (T0): {t0_argmax_pct_dict[0]:.2%}")
print(f"Mão Esquerda (T1): {t1_argmax_pct_dict[1]:.2%}")
print(f"Mão Direita (T2): {t2_argmax_pct_dict[2]:.2%}")

# Calcular acurácia média
mean_acc = (t0_argmax_pct_dict[0] + t1_argmax_pct_dict[1] + t2_argmax_pct_dict[2]) / 3
print(f"\nAcurácia média: {mean_acc:.2%}")
