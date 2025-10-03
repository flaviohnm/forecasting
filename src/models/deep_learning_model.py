# File: src/models/deep_learning_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer, NHITS

# --- Modelos baseados em Keras (N-BEATS, LSTM) ---

def build_nbeats_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """Constrói um modelo N-BEATS com uma única saída (para abordagem Direta)."""
    model = Sequential()
    model.add(Dense(n_neurons[0], activation='relu', input_shape=input_shape))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_lstm_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """Constrói um modelo LSTM simples com Keras."""
    model = Sequential()
    model.add(LSTM(n_neurons[0], activation='relu', input_shape=(input_shape[0], 1)))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_nbeats_mimo_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """Constrói um modelo N-BEATS com múltiplas saídas para a abordagem MIMO."""
    model = Sequential()
    model.add(Dense(n_neurons[0], activation='relu', input_shape=input_shape))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape)) # Camada de saída com 'output_shape' neurônios
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_save_keras_model(X_train, y_train, model_path: str, model_params: dict, model_builder, output_shape=1):
    """Função genérica para treinar e salvar modelos Keras."""
    print(f"Treinando modelo Keras para {os.path.basename(model_path)}...")
    
    if model_builder == build_lstm_model:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = (X_train.shape[1],)

    model = model_builder(
        input_shape, output_shape,
        n_neurons=model_params['n_neurons'],
        learning_rate=model_params['learning_rate']
    )
    model.fit(X_train, y_train, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
    print("Treinamento Keras concluído.")

    model.save(model_path)
    print(f"Modelo Keras salvo em: {model_path}")

def load_and_predict_keras_direct(model_path: str, input_data, is_lstm: bool):
    """Carrega um modelo Keras de saída única e faz uma previsão."""
    if not os.path.exists(model_path): raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = load_model(model_path)
    
    if input_data.ndim == 1: input_data = input_data.reshape(1, -1)
    if is_lstm: input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        
    prediction = model.predict(input_data, verbose=0)
    return prediction[0, 0]

def load_and_predict_keras_mimo(model_path: str, input_data):
    """Carrega um modelo Keras MIMO e retorna todas as previsões de uma vez."""
    if not os.path.exists(model_path): raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = load_model(model_path)

    if input_data.ndim == 1: input_data = input_data.reshape(1, -1)
        
    prediction = model.predict(input_data, verbose=0)
    return prediction.flatten()

# --- Modelos baseados em NeuralForecast ---

def train_and_predict_neuralforecast(train_series: pd.Series, horizon: int, model_class, model_params: dict):
    """
    Função genérica para treinar e prever com modelos da NeuralForecast.
    """
    print(f"Treinando e prevendo com {model_class.__name__} via NeuralForecast...")
    
    df = train_series.to_frame(name='y').reset_index()
    df = df.rename(columns={df.columns[0]: 'ds'})
    df['unique_id'] = 'series_1'
    
    model = model_class(h=horizon, **model_params)
    
    nf = NeuralForecast(models=[model], freq=pd.infer_freq(train_series.index))
    nf.fit(df=df)
    
    forecast_df = nf.predict()
    
    print("Previsão com NeuralForecast concluída.")
    return forecast_df[model_class.__name__.upper()].values