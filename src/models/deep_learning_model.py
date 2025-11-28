# File: src/models/deep_learning_model.py

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import numpy as np
import logging
from neuralforecast import NeuralForecast

# --- CONSTRUTORES KERAS ---


def build_mlp_model(input_shape: tuple, output_shape: int, n_neurons: list,
                    learning_rate: float):
    """Constrói um modelo MLP com uma única saída."""
    model = Sequential([
        Input(shape=input_shape),
        Dense(n_neurons[0], activation='relu'),
    ])
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def build_lstm_model(input_shape: tuple, output_shape: int, n_neurons: list,
                     learning_rate: float):
    """Constrói um modelo LSTM simples com Keras."""
    model = Sequential([
        Input(shape=(input_shape[0], 1)),
        LSTM(n_neurons[0], activation='relu'),
    ])
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def build_nbeats_mimo_model(input_shape: tuple, output_shape: int,
                            n_neurons: list, learning_rate: float):
    """Constrói um modelo MLP com múltiplas saídas (MIMO)."""
    model = Sequential([
        Input(shape=input_shape),
        Dense(n_neurons[0], activation='relu'),
    ])
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# --- FUNÇÕES DE TREINO E PREVISÃO KERAS ---


def train_and_save_keras_model(X_train,
                               y_train,
                               model_path: str,
                               model_params: dict,
                               model_builder,
                               output_shape=1):
    """Função genérica para treinar e salvar modelos Keras com Early Stopping."""
    print(f"Treinando modelo Keras para {os.path.basename(model_path)}...")

    if model_builder == build_lstm_model:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    input_shape = (X_train.shape[1], )

    builder_params = {
        'n_neurons': model_params.get('n_neurons', [100, 50]),
        'learning_rate': model_params.get('learning_rate', 0.001)
    }
    fit_params = {
        'epochs': model_params.get('epochs', 100),
        'batch_size': model_params.get('batch_size', 16),
        'verbose': 0
    }

    # --- CONFIGURAÇÃO DO EARLY STOPPING ---
    patience = model_params.get('patience',
                                10)  # Default 10 se não especificado
    callbacks = []
    if patience > 0:
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   restore_best_weights=True,
                                   verbose=1)
        callbacks.append(early_stop)
        # Adiciona split de validação automático para o Early Stopping funcionar
        fit_params['validation_split'] = 0.2

    model = model_builder(input_shape, output_shape, **builder_params)

    # Treina com callbacks
    model.fit(X_train, y_train, callbacks=callbacks, **fit_params)

    print("Treinamento Keras concluído.")
    model.save(model_path)
    print(f"Modelo Keras salvo em: {model_path}")


def load_and_predict_keras_direct(model_path: str, input_data, is_lstm: bool):
    """Carrega um modelo Keras de saída única e faz uma previsão."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = load_model(model_path)

    if input_data.ndim == 1: input_data = input_data.reshape(1, -1)
    if is_lstm:
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], 1))

    prediction = model.predict(input_data, verbose=0)
    return prediction[0, 0] if prediction.ndim == 2 else prediction[0]


def load_and_predict_keras_mimo(model_path: str,
                                input_data,
                                is_lstm: bool = False):
    """Carrega um modelo Keras MIMO e retorna todas as previsões de uma vez."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = load_model(model_path)

    if input_data.ndim == 1: input_data = input_data.reshape(1, -1)
    if is_lstm:
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], 1))

    prediction = model.predict(input_data, verbose=0)
    return prediction.flatten()


# --- FUNÇÕES NEURALFORECAST ---


def get_freq(index):
    """Helper para inferir frequência de forma segura."""
    freq = pd.infer_freq(index)
    if freq is None:
        logging.warning(
            "NeuralForecast: Não foi possível inferir frequência. Usando 'M' (mensal) como fallback."
        )
        return 'M'
    return freq


def predict_residuals_direct_nf(data_series: pd.Series, horizon: int,
                                input_lags: int, model_class,
                                model_configs: dict):
    """
    Implementa a abordagem de previsão DIRETA usando modelos da NeuralForecast.
    """
    print(
        f"Iniciando previsão direta com {model_class.__name__} para {horizon} passos..."
    )

    model_kwargs = model_configs.get('model_kwargs', {}).copy()
    trainer_kwargs = model_configs.get('trainer_kwargs', {})
    model_kwargs['input_size'] = input_lags

    all_predictions = []
    data_freq = get_freq(data_series.index)

    for h in range(1, horizon + 1):
        end_idx_train = len(data_series) - h

        df_h = pd.DataFrame({
            'ds': data_series.index[:end_idx_train],
            'y': data_series.values[h:]
        })
        df_h['unique_id'] = 'series_1'

        if df_h.empty:
            logging.warning(f"  AVISO: DataFrame vazio para h={h}. Pulando.")
            all_predictions.append(np.nan)
            continue

        combined_params = {**model_kwargs, **trainer_kwargs}
        model = model_class(h=1, **combined_params)

        nf = NeuralForecast(models=[model], freq=data_freq)

        # Adiciona val_size se early stopping estiver ativado
        fit_kwargs = {}
        if 'early_stop_patience_steps' in trainer_kwargs:
            # Usa os últimos 10% ou pelo menos input_lags para validação
            val_size = max(input_lags, int(len(df_h) * 0.1))
            fit_kwargs['val_size'] = val_size

        nf.fit(df=df_h,
               **fit_kwargs)  # Passa val_size para ativar early stopping

        forecast = nf.predict()
        cols = forecast.columns
        model_col = [c for c in cols if c not in ['unique_id', 'ds']][0]
        prediction_value = forecast[model_col].values[0]
        all_predictions.append(prediction_value)

    return np.array(all_predictions)


def train_and_predict_neuralforecast(train_series: pd.Series, horizon: int,
                                     model_class, model_configs: dict):
    """
    Função genérica para treinar e prever com modelos da NeuralForecast.
    """
    print(
        f"Treinando e prevendo com {model_class.__name__} via NeuralForecast.."
    )

    df = train_series.to_frame(name='y').reset_index()
    df = df.rename(columns={df.columns[0]: 'ds'})
    df['unique_id'] = 'series_1'

    model_kwargs = model_configs.get('model_kwargs', {})
    trainer_kwargs = model_configs.get('trainer_kwargs', {})

    combined_params = {**model_kwargs, **trainer_kwargs}
    model = model_class(h=horizon, **combined_params)

    data_freq = get_freq(train_series.index)

    nf = NeuralForecast(models=[model], freq=data_freq)

    # Adiciona val_size se early stopping estiver ativado
    fit_kwargs = {}
    if 'early_stop_patience_steps' in trainer_kwargs:
        # Usa os últimos 20% ou 2x horizonte para validação
        val_size = max(horizon * 2, int(len(df) * 0.2))
        fit_kwargs['val_size'] = val_size

    nf.fit(df=df, **fit_kwargs)  # Passa val_size para ativar early stopping

    forecast_df = nf.predict()
    print("Previsão com NeuralForecast concluída.")

    cols = forecast_df.columns
    model_col = [c for c in cols if c not in ['unique_id', 'ds']][0]
    return forecast_df[model_col].values
