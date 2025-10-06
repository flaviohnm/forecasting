# File: src/pipelines/train_pipeline.py

import os
import pandas as pd
import joblib
import numpy as np

from src.data_management import preprocessing
from src.models import arima_model, deep_learning_model, ets_model

def run(main_config: dict, model_conf: dict, dataset_conf: dict, execution_name: str):
    """
    Executa o pipeline de treinamento para uma combinação de modelo e dataset.
    """
    model_type = model_conf['model_type']
    models_path = main_config['models_path']
    
    train_series, _ = preprocessing.load_and_prepare_data(main_config, dataset_conf)

    if model_type == 'ARIMA':
        model_path = os.path.join(models_path, f"{execution_name}.joblib")
        arima_params = model_conf['arima_params'].copy()
        arima_params['seasonal'] = dataset_conf['seasonal_period'] > 1
        arima_params['m'] = dataset_conf['seasonal_period']
        arima_model.train_and_save_arima(train_series, model_path, arima_params)

    elif model_type == 'ETS':
        model_path = os.path.join(models_path, f"{execution_name}.pkl")
        ets_model.train_and_save_ets(
            train_series, model_path, model_conf['ets_params'], dataset_conf['seasonal_period']
        )
        
    elif model_type == 'LSTM':
        horizon = dataset_conf['forecast_horizon']
        input_lags = model_conf['lstm_params']['input_lags']
        lstm_datasets = preprocessing.create_direct_forecast_datasets(train_series, input_lags, horizon)
        
        for h in range(1, horizon + 1):
            X_train, y_train = lstm_datasets[h]
            model_path = os.path.join(models_path, f"{execution_name}_h{h}.keras")
            deep_learning_model.train_and_save_keras_model(
                X_train, y_train, model_path, model_conf['lstm_params'], 
                model_builder=deep_learning_model.build_lstm_model, output_shape=1
            )
            
    elif model_type in [
        'iTransformer', 'NHiTS', 'Hybrid_MIMO_NHITS', 'Hybrid_Direct_NHITS',
        'Hybrid_MIMO_NBEATS_NF', 'Hybrid_Direct_NBEATS_NF'
    ]:
        print(f"INFO: Modelo {model_type} (baseado em NeuralForecast) não requer um passo de treino separado. O treino ocorrerá durante a avaliação.")
        pass

    else:
        print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de treino definida.")