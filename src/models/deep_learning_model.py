# File: src/models/deep_learning_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import time

from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer, NHITS, NBEATS


def build_mlp_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """Constrói um modelo MLP simples com Keras."""
    model = Sequential()
    # --- CORREÇÃO APLICADA ---
    # Adicionamos uma camada de Input explícita
    model.add(Input(shape=input_shape))
    # A primeira camada Dense não precisa mais do 'input_shape'
    model.add(Dense(n_neurons[0], activation='relu'))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_lstm_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """Constrói um modelo LSTM simples com Keras."""
    model = Sequential()
    # --- CORREÇÃO APLICADA ---
    # Adicionamos uma camada de Input explícita
    model.add(Input(shape=(input_shape[0], 1)))
    # A camada LSTM não precisa mais do 'input_shape'
    model.add(LSTM(n_neurons[0], activation='relu'))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_and_save_keras_model(X_train,
                               y_train,
                               model_path: str,
                               model_params: dict,
                               model_builder,
                               output_shape=1):
    """Função genérica para treinar e salvar modelos Keras."""
    # ... (código sem alterações)
    print(f"Treinando modelo Keras para {os.path.basename(model_path)}...")

    if model_builder == build_lstm_model:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = (X_train.shape[1], )

    model = model_builder(input_shape,
                          output_shape,
                          n_neurons=model_params['n_neurons'],
                          learning_rate=model_params['learning_rate'])
    model.fit(X_train,
              y_train,
              epochs=model_params['epochs'],
              batch_size=model_params['batch_size'],
              verbose=0)
    print("Treinamento Keras concluído.")

    model.save(model_path)
    print(f"Modelo Keras salvo em: {model_path}")


def load_and_predict_keras_direct(model_path: str, input_data, is_lstm: bool):
    """Carrega um modelo Keras de saída única e faz uma previsão."""
    # ... (código sem alterações)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = load_model(model_path)

    if input_data.ndim == 1: input_data = input_data.reshape(1, -1)
    if is_lstm:
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], 1))

    prediction = model.predict(input_data, verbose=0)
    return prediction[0, 0]


# --- Modelos baseados em NeuralForecast ---


def train_and_predict_neuralforecast(train_series: pd.Series, horizon: int,
                                     model_class, model_configs: dict):
    """
    Função genérica para treinar e prever com modelos da NeuralForecast (abordagem MIMO).
    """
    print(
        f"Treinando e prevendo com {model_class.__name__} via NeuralForecast (MIMO)..."
    )

    df = train_series.to_frame(name='y').reset_index()
    df = df.rename(columns={df.columns[0]: 'ds'})
    df['unique_id'] = 'series_1'

    model_kwargs = model_configs.get('model_kwargs', {})
    trainer_kwargs = model_configs.get('trainer_kwargs', {})

    combined_params = {**model_kwargs, **trainer_kwargs}
    model = model_class(h=horizon, **combined_params)

    nf = NeuralForecast(models=[model], freq=pd.infer_freq(train_series.index))
    nf.fit(df=df)

    forecast_df = nf.predict()

    print("Previsão com NeuralForecast concluída.")

    # --- LÓGICA DE RETORNO CORRIGIDA E ROBUSTA ---
    # Encontra o nome da coluna da previsão dinamicamente
    cols = forecast_df.columns
    model_col = [c for c in cols if c not in ['unique_id', 'ds']][0]

    return forecast_df[model_col].values


def predict_residuals_direct_nf(residuals_train: pd.Series, horizon: int,
                                input_lags: int, model_class,
                                model_configs: dict):
    """
    Implementa a abordagem de previsão DIRETA para resíduos usando modelos da NeuralForecast.
    """
    print(
        f"Iniciando previsão direta com {model_class.__name__} para {horizon} passos..."
    )

    model_kwargs = model_configs.get('model_kwargs', {}).copy()
    trainer_kwargs = model_configs.get('trainer_kwargs', {})

    model_kwargs['input_size'] = input_lags

    combined_params = {**model_kwargs, **trainer_kwargs}

    all_predictions = []

    for h in range(1, horizon + 1):
        start_time = time.time()
        print(f"  - Treinando modelo especialista para o horizonte h={h}...")

        end_idx = len(residuals_train) - h

        df_h = pd.DataFrame({
            'ds': residuals_train.index[:end_idx],
            'y': residuals_train.values[h:]
        })
        df_h['unique_id'] = 'series_1'

        model = model_class(h=1, **combined_params)

        nf = NeuralForecast(models=[model],
                            freq=pd.infer_freq(residuals_train.index))
        nf.fit(df=df_h)

        forecast = nf.predict()

        # --- LÓGICA DE RETORNO CORRIGIDA E ROBUSTA ---
        # Encontra o nome da coluna da previsão dinamicamente
        cols = forecast.columns
        model_col = [c for c in cols if c not in ['unique_id', 'ds']][0]

        prediction_value = forecast[model_col].values[0]
        all_predictions.append(prediction_value)

        end_time = time.time()
        print(
            f"  - Concluído em {end_time - start_time:.2f}s. Previsão para h={h}: {prediction_value:.4f}"
        )

    return np.array(all_predictions)
