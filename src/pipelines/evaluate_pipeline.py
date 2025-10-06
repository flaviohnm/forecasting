# File: src/pipelines/evaluate_pipeline.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error

from src.data_management import preprocessing
from src.models import deep_learning_model, ets_model
from neuralforecast.models import iTransformer, NHITS, NBEATS


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    # Garante que y_train seja um array numpy para operações de slicing e subtração
    y_train = np.asarray(y_train)
    d = np.sum(np.abs(y_train[1:] - y_train[:-1])) / (n - 1)
    if d == 0: return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred)) / d


def run(main_config: dict, model_conf: dict, dataset_conf: dict,
        execution_name: str):
    
    # --- NOVO BLOCO TRY...EXCEPT PARA ROBUSTEZ ---
    try:
        model_type = model_conf['model_type']
        models_path = main_config['models_path']
        horizon = dataset_conf['forecast_horizon']

        train_series, test_series = preprocessing.load_and_prepare_data(
            main_config, dataset_conf)
        if len(test_series) > horizon: test_series = test_series.iloc[:horizon]

        final_forecast, forecasts_df = None, pd.DataFrame({'real': test_series})

        if model_type == 'ARIMA':
            print("Avaliando modelo ARIMA puro...")
            model_path = os.path.join(models_path, f"{execution_name}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo não encontrado: '{model_path}'.")
            arima_instance = joblib.load(model_path)
            predictions = arima_instance.predict(n_periods=len(test_series))
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type == 'ETS':
            print("Avaliando modelo ETS puro...")
            model_path = os.path.join(models_path, f"{execution_name}.pkl")
            predictions = ets_model.load_and_forecast_ets(model_path,
                                                          len(test_series))
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type == 'LSTM':
            print("Avaliando modelo LSTM puro...")
            input_lags = model_conf['lstm_params']['input_lags']
            last_data_points = train_series.values[-input_lags:]
            predictions = [
                deep_learning_model.load_and_predict_keras_direct(os.path.join(
                    models_path, f"{execution_name}_h{h}.keras"),
                                                                  last_data_points,
                                                                  is_lstm=True)
                for h in range(1,
                               len(test_series) + 1)
            ]
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type in ['iTransformer', 'NHiTS']:
            print(f"Avaliando modelo {model_type} puro...")
            model_class = iTransformer if model_type == 'iTransformer' else NHITS
            model_params = model_conf['itransformer_params'] if model_type == 'iTransformer' else model_conf['nhits_params']

            predictions = deep_learning_model.train_and_predict_neuralforecast(
                train_series, len(test_series), model_class, model_params)
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type.startswith('Hybrid_'):
            print(f"Avaliando modelo Híbrido: {model_type}...")
            dependency_name = model_conf.get('depends_on')
            arima_model_path = os.path.join(
                models_path, f"{dataset_conf['name']}_{dependency_name}.joblib")
            if not os.path.exists(arima_model_path):
                raise FileNotFoundError(f"Modelo ARIMA dependente não encontrado: '{arima_model_path}'.")

            arima_instance = joblib.load(arima_model_path)
            linear_forecasts = arima_instance.predict(n_periods=len(test_series))
            residuals_train = train_series - pd.Series(
                arima_instance.predict_in_sample(), index=train_series.index)
            
            nonlinear_forecasts = None

            if model_type == 'Hybrid_MIMO_NHITS':
                nonlinear_forecasts = deep_learning_model.train_and_predict_neuralforecast(
                    residuals_train, len(test_series), NHITS, model_conf['nhits_params']
                )
            elif model_type == 'Hybrid_MIMO_NBEATS_NF':
                nonlinear_forecasts = deep_learning_model.train_and_predict_neuralforecast(
                    residuals_train, len(test_series), NBEATS, model_conf['nbeats_params']
                )
            elif model_type == 'Hybrid_Direct_NHITS':
                input_lags = model_conf['nhits_params']['model_kwargs']['input_size']
                nonlinear_forecasts = deep_learning_model.predict_residuals_direct_nf(
                    residuals_train, len(test_series), input_lags, NHITS, model_conf['nhits_params']
                )
            elif model_type == 'Hybrid_Direct_NBEATS_NF':
                input_lags = model_conf['nbeats_params']['model_kwargs']['input_size']
                nonlinear_forecasts = deep_learning_model.predict_residuals_direct_nf(
                    residuals_train, len(test_series), input_lags, NBEATS, model_conf['nbeats_params']
                )
            
            if nonlinear_forecasts is not None:
                final_forecast = pd.Series(linear_forecasts + nonlinear_forecasts, index=test_series.index)
                forecasts_df['previsao'] = final_forecast
                forecasts_df['previsao_arima'] = linear_forecasts
                forecasts_df['previsao_residuos'] = nonlinear_forecasts
            else:
                raise ValueError(f"Lógica de previsão não implementada para o modelo_type Híbrido: {model_type}")

        else:
            print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de avaliação definida.")
            return

        # --- Verificação de Previsão Válida ---
        # Se final_forecast não foi gerada ou contém valores inválidos, não prossiga.
        if final_forecast is None or final_forecast.isnull().any() or np.isinf(final_forecast).any():
             raise ValueError("A previsão gerou valores inválidos (NaN ou Inf).")


        # --- Cálculo de Métricas e Salvamento ---
        y_true, y_pred, y_train = test_series.values, final_forecast.values, train_series.values
        
        if dataset_conf.get('log_transform', False):
            y_true_orig = np.exp(y_true)
            y_pred_orig = np.exp(y_pred)
        else:
            y_true_orig, y_pred_orig = y_true, y_pred

        mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
        mase = mean_absolute_scaled_error(y_true, y_pred, y_train)

        print(f"Resultados para '{execution_name}':\n  - MAPE: {mape:.4f}\n  - MASE: {mase:.4f}")

        metrics_df = pd.DataFrame({
            'execution_name': [execution_name], 'model_type': [model_type],
            'dataset': [dataset_conf['name']], 'MAPE': [mape], 'MASE': [mase]
        })
        metrics_df.to_csv(os.path.join(main_config['results_paths']['metrics'],
                                       f"metrics_{execution_name}.csv"), index=False)
        forecasts_df.to_csv(
            os.path.join(main_config['results_paths']['plots'],
                         f"forecasts_{execution_name}.csv"))
    
    except Exception as e:
        print(f"\n!!!!!! AVALIAÇÃO FALHOU para '{execution_name}' !!!!!!!")
        print(f"Causa: {e}\n")