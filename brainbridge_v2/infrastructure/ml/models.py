"""
Definições de modelos de ML para BrainBridge v2

Inclui construção de um modelo CNN 1D simples para classificação de MI
e utilitários para carregar/salvar modelos Keras.

Perfil canonico: 16 canais @ 125 Hz (OpenBCI Daisy), janela 250 (2 s).
A rede e adaptativa no tempo (GlobalAveragePooling1D): aceita qualquer
numero de timesteps e qualquer fs, pois o preprocessing reamostra para o
canonico antes da inferencia. Canais: qualquer combinacao e mapeada para
16 via truncamento/padding ou correspondencia por nome (ver eeg_pipeline).
"""

from typing import Tuple, Optional
import importlib


def build_cnn_1d(input_shape: Tuple[int, int] = (250, 16), num_classes: int = 2):
	"""Constroi um modelo CNN 1D simples em Keras.

	Args:
		input_shape: (timesteps, channels)
		num_classes: número de classes (2 para T1/T2)

	Returns:
		tf.keras.Model
	"""
	
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e

	inputs = layers.Input(shape=input_shape)
	x = inputs

	# Conv1D espera (timesteps, features). Aqui features=canais.
	x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.GlobalAveragePooling1D()(x)

	x = layers.Dropout(0.3)(x)
	x = layers.Dense(64, activation='relu')(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)

	model = models.Model(inputs=inputs, outputs=outputs, name='cnn1d_mi')
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	return model


def load_keras_model(model_path: str):
	"""Carrega um modelo Keras salvo (.keras ou .h5)."""
	try:
		keras_models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e
	return keras_models.load_model(model_path)


def build_simple_eegnet(input_shape: Tuple[int, int] = (250, 16), num_classes: int = 2):
	"""Variante simplificada tipo EEGNet, compativel com dados (T,C)."""
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError("TensorFlow não está disponível. Instale com 'pip install tensorflow'") from e

	inputs = layers.Input(shape=input_shape)
	x = layers.Conv1D(16, 64, padding='same')(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Conv1D(32, 1)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.MaxPooling1D(pool_size=4)(x)
	x = layers.Dropout(0.25)(x)

	x = layers.Conv1D(32, 16, padding='same', groups=1)(x)
	x = layers.Conv1D(32, 1)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.MaxPooling1D(pool_size=8)(x)
	x = layers.Dropout(0.25)(x)

	x = layers.Flatten()(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	model = models.Model(inputs, outputs, name='eegnet_simple')
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


def save_model(model, path: str) -> None:
	"""Salva modelo Keras em path (.keras recomendado)."""
	import os
	os.makedirs(os.path.dirname(path), exist_ok=True)
	model.save(path)


def build_shallow_convnet_adaptive(num_classes: int = 2, channels: int = 16,
                                  dropout: float = 0.5, l2: float = 0.0):
	"""ShallowConvNet adaptativa no tempo (None, C) + GAP.

	Top DL no benchmark MOABB: conv temporal (25) -> conv espacial (1x1
	sobre canais, analogo ao CSP) -> square -> pool -> log -> GAP -> dense.
	"""
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
		regularizers = importlib.import_module('tensorflow.keras.regularizers')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e

	reg = regularizers.l2(float(l2)) if l2 and float(l2) > 0 else None
	inputs = layers.Input(shape=(None, int(channels)))
	x = layers.Conv1D(40, 25, padding='same', use_bias=False,
	                  kernel_regularizer=reg)(inputs)
	x = layers.Conv1D(40, 1, padding='same', use_bias=False,
	                  kernel_regularizer=reg)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Lambda(lambda t: t ** 2)(x)
	x = layers.AveragePooling1D(pool_size=75, strides=15, padding='same')(x)
	x = layers.Lambda(lambda t: tf_log(t))(x)
	x = layers.Dropout(float(dropout))(x)
	x = layers.GlobalAveragePooling1D()(x)
	outputs = layers.Dense(int(num_classes), activation='softmax',
	                       kernel_regularizer=reg)(x)
	model = models.Model(inputs=inputs, outputs=outputs, name='shallow_convnet_adaptive')
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model


def tf_log(t):
	"""log() numericamente seguro para a etapa log-var da ShallowConvNet."""
	import importlib as _il
	backend = _il.import_module('tensorflow.keras.backend')
	return backend.log(backend.maximum(t, 1e-6))


def build_eegnet_adaptive(num_classes: int = 2, channels: int = 16,
                          F1: int = 8, D: int = 2, dropout: float = 0.25,
                          l2: float = 1e-4):
	"""EEGNet compacta e adaptativa no tempo (None, C) + GAP.

	Padrao-ouro para MI cross-subject: conv temporal -> separable
	espacial/temporal -> pooling -> GAP. Aceita qualquer n. de timesteps
	(125/128/250 Hz); canais fixos em 16 (mapeados no preprocessing).
	"""
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
		regularizers = importlib.import_module('tensorflow.keras.regularizers')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e

	reg = regularizers.l2(float(l2)) if l2 and float(l2) > 0 else None
	inputs = layers.Input(shape=(None, int(channels)))
	x = layers.Conv1D(int(F1), 64, padding='same', use_bias=False,
	                  kernel_regularizer=reg)(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.SeparableConv1D(int(F1 * D), 16, padding='same', use_bias=False,
	                           depthwise_regularizer=reg,
	                           pointwise_regularizer=reg)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.AveragePooling1D(pool_size=4, padding='same')(x)
	x = layers.Dropout(float(dropout))(x)
	x = layers.SeparableConv1D(int(F1 * D * 2), 16, padding='same', use_bias=False,
	                           depthwise_regularizer=reg,
	                           pointwise_regularizer=reg)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.AveragePooling1D(pool_size=8, padding='same')(x)
	x = layers.Dropout(float(dropout))(x)
	x = layers.GlobalAveragePooling1D()(x)
	outputs = layers.Dense(int(num_classes), activation='softmax',
	                       kernel_regularizer=reg)(x)
	model = models.Model(inputs=inputs, outputs=outputs, name='eegnet_adaptive')
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model


def build_adaptive_cnn_1d(num_classes: int = 2, channels: int = 16,
                           filters=(64, 64, 128), dropout: float = 0.3):
	"""CNN 1D adaptativa: aceita qualquer numero de timesteps (None, C).

	Usa GlobalAveragePooling1D, entao a mesma rede consome janelas de
	125/128/250 Hz (reamostradas ou nao) sem trocar de checkpoint.
	C Canais fixos em 16 (qualquer combinacao de entrada e mapeada para 16).
	"""
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e

	inputs = layers.Input(shape=(None, int(channels)))
	x = inputs
	for f, k in zip(filters, (5, 5, 3)):
		x = layers.Conv1D(int(f), kernel_size=int(k), padding='same',
		                  activation='relu')(x)
		x = layers.BatchNormalization()(x)
		# Pooling adaptativo: nunca colapsa janelas curtas.
		x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dropout(float(dropout))(x)
	x = layers.Dense(64, activation='relu')(x)
	outputs = layers.Dense(int(num_classes), activation='softmax')(x)
	model = models.Model(inputs=inputs, outputs=outputs, name='cnn1d_mi_adaptive')
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model
