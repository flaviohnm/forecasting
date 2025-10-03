# File: src/pipelines/evaluate_pipeline.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error

from src.data_management import preprocessing
from src.models import deep_learning_model, ets_model
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer, NHITS


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.sum(np.abs(y_train[1:] - y_train[:-1])) / (n - 1)
    if d == 0: return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred)) / d

def run(main_config: dict, model_conf: dict, dataset_conf: dict, execution_name: str):
    model_type = model_conf['model_type']
    models_path = main_config['models_path']
    horizon = dataset_conf['forecast_horizon']

    train_series, test_series = preprocessing.load_and_prepare_data(main_config, dataset_conf)
    if len(test_series) > horizon: test_series = test_series.iloc[:horizon]

    final_forecast, forecasts_df = None, pd.DataFrame({'real': test_series})

    if model_type == 'ARIMA':
        print("Avaliando modelo ARIMA puro...")
        model_path = os.path.join(models_path, f"{execution_name}.joblib")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Modelo não encontrado: '{model_path}'.")
        arima_instance = joblib.load(model_path)
        predictions = arima_instance.predict(n_periods=len(test_series))
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast

    elif model_type == 'ETS':
        print("Avaliando modelo ETS puro...")
        model_path = os.path.join(models_path, f"{execution_name}.pkl")
        predictions = ets_model.load_and_forecast_ets(model_path, len(test_series))
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast
        
    elif model_type == 'LSTM':
        print("Avaliando modelo LSTM puro...")
        input_lags = model_conf['lstm_params']['input_lags']
        last_data_points = train_series.values[-input_lags:]
        predictions = [deep_learning_model.load_and_predict_keras_direct(os.path.join(models_path, f"{execution_name}_h{h}.keras"), last_data_points, is_lstm=True) for h in range(1, len(test_series) + 1)]
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast

    elif model_type in ['Transformer', 'NHiTS']:
        model_map = {'Transformer': (Transformer, model_conf['transformer_params']), 'NHiTS': (NHiTS, model_conf['nhits_params'])}
        model_class, model_params = model_map[model_type]
        predictions = deep_learning_model.train_and_predict_neuralforecast(train_series, len(test_series), model_class, model_params)
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast

    elif model_type in ['Hybrid_Direct', 'Hybrid_MIMO', 'Hybrid_Direct_NHITS', 'Hybrid_MIMO_NHITS']:
        print(f"Avaliando modelo {model_type}...")
        dependency_name = model_conf.get('depends_on')
        arima_model_path = os.path.join(models_path, f"{dataset_conf['name']}_{dependency_name}.joblib")
        if not os.path.exists(arima_model_path): raise FileNotFoundError(f"Modelo ARIMA dependente não encontrado: '{arima_model_path}'.")
            
        arima_instance = joblib.load(arima_model_path)
        linear_forecasts = arima_instance.predict(n_periods=len(test_series))
        
        residuals_train = train_series - pd.Series(arima_instance.predict_in_sample(), index=train_series.index)
        
        if model_type == 'Hybrid_Direct':
            input_lags = model_conf['nbeats_params']['input_lags']
            last_residuals = residuals_train.values[-input_lags:]
            nonlinear_forecasts = [deep_learning_model.load_and_predict_keras_direct(os.path.join(models_path, f"{execution_name}_h{h}.keras"), last_residuals, is_lstm=False) for h in range(1, len(test_series) + 1)]
        
        elif model_type == 'Hybrid_MIMO':
            input_lags = model_conf['nbeats_params']['input_lags']
            last_residuals = residuals_train.values[-input_lags:]
            mimo_model_path = os.path.join(models_path, f"{execution_name}.keras")
            nonlinear_forecasts = deep_learning_model.load_and_predict_keras_mimo(mimo_model_path, last_residuals)

        elif model_type == 'Hybrid_MIMO_NHITS':
            nonlinear_forecasts = deep_learning_model.train_and_predict_neuralforecast(
                residuals_train, len(test_series), NHiTS, model_conf['nhits_params']
            )