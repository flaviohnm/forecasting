# File: src/pipelines/evaluate_pipeline.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error

from src.data_management import preprocessing
from src.models import deep_learning_model, ets_model

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    Calcula o MASE (Mean Absolute Scaled Error).
    """
    n = len(y_train)
    d = np.sum(np.abs(y_train[1:] - y_train[:-1])) / (n - 1)
    if d == 0:
        return np.mean(np.abs(y_true - y_pred))
    
    errors = np.mean(np.abs(y_true - y_pred))
    return errors / d

def run(main_config: dict, model_conf: dict, dataset_conf: dict, execution_name: str):
    """
    Executa o pipeline de avaliação para uma combinação de modelo e dataset.
    """
    model_type = model_conf['model_type']
    models_path = main_config['models_path']
    horizon = dataset_conf['forecast_horizon']

    train_series, test_series = preprocessing.load_and_prepare_data(main_config, dataset_conf)
    
    if len(test_series) > horizon:
        test_series = test_series.iloc[:horizon]

    final_forecast = None
    forecasts_df = pd.DataFrame(index=test_series.index)
    forecasts_df['real'] = test_series

    if model_type == 'ARIMA':
        print("Avaliando modelo ARIMA puro...")
        model_path = os.path.join(models_path, f"{execution_name}_arima.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo ARIMA não encontrado em '{model_path}'.")
            
        arima_instance = joblib.load(model_path)
        predictions = arima_instance.predict(n_periods=len(test_series))
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast

    elif model_type == 'ETS':
        print("Avaliando modelo ETS puro...")
        model_path = os.path.join(models_path, f"{execution_name}_ets.pkl")
        predictions = ets_model.load_and_forecast_ets(model_path, len(test_series))
        final_forecast = pd.Series(predictions, index=test_series.index)
        forecasts_df['previsao'] = final_forecast
        
    elif model_type == 'Hybrid':
        print("Avaliando modelo Híbrido...")
        dependency_model_name = model_conf.get('depends_on')
        if not dependency_model_name:
            raise ValueError(f"Modelo Híbrido '{execution_name}' não define 'depends_on'.")

        dependency_execution_name = f"{dataset_conf['name']}_{dependency_model_name}"
        arima_model_path = os.path.join(models_path, f"{dependency_execution_name}_arima.joblib")
        if not os.path.exists(arima_model_path):
            raise FileNotFoundError(f"Modelo ARIMA dependente não encontrado em '{arima_model_path}'.")
            
        arima_instance = joblib.load(arima_model_path)
        linear_forecasts = arima_instance.predict(n_periods=len(test_series))
        
        linear_preds_in_sample = arima_instance.predict_in_sample()
        residuals_train = train_series - pd.Series(linear_preds_in_sample, index=train_series.index)
        
        input_lags = model_conf['nbeats_params']['input_lags']
        last_residuals = residuals_train.values[-input_lags:]

        nonlinear_forecasts = []
        for h in range(1, len(test_series) + 1):
            nbeats_model_path = os.path.join(models_path, f"{execution_name}_h{h}.keras")
            if not os.path.exists(nbeats_model_path):
                raise FileNotFoundError(f"Modelo N-BEATS para horizonte {h} não encontrado em '{nbeats_model_path}'.")

            residual_pred = deep_learning_model.load_and_predict_direct(nbeats_model_path, last_residuals)
            nonlinear_forecasts.append(residual_pred)
        
        final_forecast = linear_forecasts + nonlinear_forecasts
        final_forecast = pd.Series(final_forecast, index=test_series.index)

        forecasts_df['previsao'] = final_forecast
        forecasts_df['previsao_arima'] = linear_forecasts
        forecasts_df['previsao_residuos'] = nonlinear_forecasts
        
    else:
        print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de avaliação definida. Pulando.")
        return

    y_true = test_series.values
    y_pred = final_forecast.values
    y_train = train_series.values
    
    y_true_for_mape, y_pred_for_mape = y_true, y_pred
    if dataset_conf.get('log_transform', False):
        print("Revertendo transformação de log para cálculo do MAPE.")
        y_true_for_mape = np.exp(y_true)
        y_pred_for_mape = np.exp(y_pred)

    mape = mean_absolute_percentage_error(y_true_for_mape, y_pred_for_mape)
    mase = mean_absolute_scaled_error(y_true, y_pred, y_train)

    print(f"Resultados para '{execution_name}':")
    print(f"  - MAPE: {mape:.4f}")
    print(f"  - MASE: {mase:.4f}")

    metrics_df = pd.DataFrame({
        'execution_name': [execution_name], 'model_type': [model_type],
        'dataset': [dataset_conf['name']], 'MAPE': [mape], 'MASE': [mase]
    })
    
    metrics_filepath = os.path.join(main_config['results_paths']['metrics'], f"metrics_{execution_name}.csv")
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Métricas salvas em: {metrics_filepath}")

    forecasts_filepath = os.path.join(main_config['results_paths']['plots'], f"forecasts_{execution_name}.csv")
    forecasts_df.to_csv(forecasts_filepath)
    print(f"Previsões salvas em: {forecasts_filepath}")