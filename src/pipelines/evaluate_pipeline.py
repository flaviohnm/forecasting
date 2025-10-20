# File: src/pipelines/evaluate_pipeline.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import logging # Para suprimir avisos

from src.data_management import preprocessing
from src.models import deep_learning_model, ets_model
from neuralforecast.models import iTransformer, NHITS, NBEATS


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    y_train = np.asarray(y_train)
    if n <= 1: return np.mean(np.abs(y_true - y_pred))
    # Calcula a diferença apenas onde possível (evita erro se tiver NaNs)
    diff = np.diff(y_train)
    if len(diff) == 0: return np.mean(np.abs(y_true - y_pred)) # Fallback se diff estiver vazio
    d = np.nanmean(np.abs(diff)) # Usa nanmean para robustez
    if d == 0 or np.isnan(d): return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred)) / d


def run(main_config: dict, model_conf: dict, dataset_conf: dict,
        execution_name: str):

    try:
        model_type = model_conf['model_type']
        models_path = main_config['models_path']
        horizon = dataset_conf['forecast_horizon']

        train_series, test_series = preprocessing.load_and_prepare_data(
            main_config, dataset_conf)
        if len(test_series) > horizon: test_series = test_series.iloc[:horizon]

        final_forecast, forecasts_df = None, pd.DataFrame({'real': test_series})

        # --- CÁLCULO DO DENOMINADOR RMSSE (Erro Naive In-Sample) ---
        mean_squared_naive_error = np.nan
        train_vals_clean = train_series.dropna().values # Remove NaNs antes de calcular
        if len(train_vals_clean) > 1:
            naive_errors_in_sample = train_vals_clean[1:] - train_vals_clean[:-1]
            mean_squared_naive_error = np.mean(np.square(naive_errors_in_sample)) # mean já ignora NaN implicitamente aqui
            if mean_squared_naive_error < 1e-9: mean_squared_naive_error = 1e-9

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

        elif model_type == 'NAIVE':
            print("Avaliando modelo NAIVE...")
            last_value = train_series.iloc[-1]
            predictions = np.full(shape=len(test_series), fill_value=last_value)
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type == 'SEASONAL_NAIVE':
             print("Avaliando modelo SEASONAL NAIVE...")
             m = dataset_conf.get('seasonal_period', 1)
             n_train = len(train_series)
             if m <= 1 or n_train < m:
                 print("INFO: Usando Naive simples para SEASONAL_NAIVE (m<=1 ou dados curtos).")
                 last_value = train_series.iloc[-1]
                 predictions = np.full(shape=len(test_series), fill_value=last_value)
             else:
                 predictions = []
                 for h in range(len(test_series)):
                     # Usa o índice relativo ao fim: -m + (h % m)
                     idx = -m + (h % m)
                     try:
                         predictions.append(train_series.iloc[idx])
                     except IndexError:
                         logging.warning(f"SNAIVE: Índice sazonal inválido ({idx}) para h={h}. Usando último valor.")
                         predictions.append(train_series.iloc[-1])
             final_forecast = pd.Series(predictions, index=test_series.index)
             forecasts_df['previsao'] = final_forecast

        elif model_type in ['MLP_Direct', 'LSTM']:
            print(f"Avaliando modelo {model_type} puro...")
            params_key = 'mlp_params' if model_type == 'MLP_Direct' else 'lstm_params'
            is_lstm = model_type == 'LSTM'
            input_lags = model_conf[params_key]['input_lags']
            scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            last_data_points = train_series.values[-input_lags:].reshape(-1, 1)
            last_data_points_scaled = scaler.transform(last_data_points).flatten() if scaler else last_data_points.flatten()
            predictions_scaled = []
            for h in range(1, len(test_series) + 1):
                model_path = os.path.join(models_path, f"{execution_name}_h{h}.keras")
                pred_scaled = deep_learning_model.load_and_predict_keras_direct(
                    model_path, last_data_points_scaled, is_lstm=is_lstm)
                predictions_scaled.append(pred_scaled)
            if scaler:
                predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
            else:
                predictions = np.array(predictions_scaled)
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type == 'MLP_MIMO':
            print(f"Avaliando modelo {model_type} puro...")
            input_lags = model_conf['mlp_params']['input_lags']
            scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            last_data_points = train_series.values[-input_lags:].reshape(-1, 1)
            last_data_points_scaled = scaler.transform(last_data_points).flatten() if scaler else last_data_points.flatten()
            model_path = os.path.join(models_path, f"{execution_name}.keras")
            model = load_model(model_path)
            prediction_scaled = model.predict(last_data_points_scaled.reshape(1, -1), verbose=0)
            if scaler:
                predictions = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
            else:
                predictions = prediction_scaled.flatten()
            final_forecast = pd.Series(predictions, index=test_series.index)
            forecasts_df['previsao'] = final_forecast

        elif model_type in ['iTransformer', 'NHiTS', 'NBEATS_MIMO', 'NBEATS_Direct']:
            print(f"Avaliando modelo {model_type} puro...")
            model_class = None
            if model_type == 'iTransformer': model_class = iTransformer
            elif model_type == 'NHiTS': model_class = NHITS
            else: model_class = NBEATS
            params_key = None
            if model_type == 'iTransformer': params_key = 'itransformer_params'
            elif model_type == 'NHiTS': params_key = 'nhits_params'
            else: params_key = 'nbeats_params'
            model_params = model_conf[params_key]
            if model_type == 'NBEATS_Direct':
                input_lags = model_params['model_kwargs']['input_size']
                predictions = deep_learning_model.predict_residuals_direct_nf(
                    train_series, len(test_series), input_lags, NBEATS, model_params)
            else:
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
            residuals_train = train_series - pd.Series(arima_instance.predict_in_sample(), index=train_series.index)
            nonlinear_forecasts = None

            if model_type == 'Hybrid_MLP_Direct':
                print(f"Avaliando modelo Híbrido: {model_type}...")
                input_lags = model_conf['mlp_params']['input_lags']
                scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                last_residuals = residuals_train.values[-input_lags:].reshape(-1, 1)
                last_residuals_scaled = scaler.transform(last_residuals).flatten() if scaler else last_residuals.flatten()
                predictions_scaled = []
                for h in range(1, len(test_series) + 1):
                    model_path = os.path.join(models_path, f"{execution_name}_h{h}.keras")
                    pred_scaled = deep_learning_model.load_and_predict_keras_direct(
                        model_path, last_residuals_scaled, is_lstm=False)
                    predictions_scaled.append(pred_scaled)
                if scaler:
                    nonlinear_forecasts = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
                else:
                    nonlinear_forecasts = np.array(predictions_scaled)

            elif model_type in ['Hybrid_MLP_Recursive', 'Hybrid_LSTM_Recursive']:
                params_key = 'mlp_params' if 'MLP' in model_type else 'lstm_params'
                is_lstm = 'LSTM' in model_type
                input_lags = model_conf[params_key]['input_lags']
                model_path = os.path.join(models_path, f"{execution_name}.keras")
                scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                last_residuals = residuals_train.values[-input_lags:].reshape(-1, 1)
                last_residuals_scaled = scaler.transform(last_residuals).flatten().tolist() if scaler else last_residuals.flatten().tolist()
                nonlinear_forecasts_scaled = []
                for _ in range(len(test_series)):
                    input_window = np.array(last_residuals_scaled)
                    pred_scaled = deep_learning_model.load_and_predict_keras_direct(
                        model_path, input_window, is_lstm=is_lstm)
                    nonlinear_forecasts_scaled.append(pred_scaled)
                    last_residuals_scaled.pop(0)
                    last_residuals_scaled.append(pred_scaled)
                if scaler:
                    nonlinear_forecasts = scaler.inverse_transform(np.array(nonlinear_forecasts_scaled).reshape(-1, 1)).flatten()
                else:
                    nonlinear_forecasts = np.array(nonlinear_forecasts_scaled)

            elif model_type == 'Hybrid_MLP_MIMO':
                input_lags = model_conf['mlp_params']['input_lags']
                model_path = os.path.join(models_path, f"{execution_name}.keras")
                scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                last_residuals = residuals_train.values[-input_lags:].reshape(-1, 1)
                last_residuals_scaled = scaler.transform(last_residuals).flatten() if scaler else last_residuals.flatten()
                model = load_model(model_path)
                prediction_scaled = model.predict(last_residuals_scaled.reshape(1, -1), verbose=0)
                if scaler:
                    nonlinear_forecasts = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
                else:
                    nonlinear_forecasts = prediction_scaled.flatten()

            elif model_type in ['Hybrid_MIMO_NHITS', 'Hybrid_MIMO_NBEATS_NF', 'Hybrid_Direct_NHITS', 'Hybrid_Direct_NBEATS_NF']:
                if model_type == 'Hybrid_MIMO_NHITS':
                    nonlinear_forecasts = deep_learning_model.train_and_predict_neuralforecast(residuals_train, len(test_series), NHITS, model_conf['nhits_params'])
                elif model_type == 'Hybrid_MIMO_NBEATS_NF':
                    nonlinear_forecasts = deep_learning_model.train_and_predict_neuralforecast(residuals_train, len(test_series), NBEATS, model_conf['nbeats_params'])
                elif model_type == 'Hybrid_Direct_NHITS':
                    input_lags = model_conf['nhits_params']['model_kwargs']['input_size']
                    nonlinear_forecasts = deep_learning_model.predict_residuals_direct_nf(residuals_train, len(test_series), input_lags, NHITS, model_conf['nhits_params'])
                elif model_type == 'Hybrid_Direct_NBEATS_NF':
                    input_lags = model_conf['nbeats_params']['model_kwargs']['input_size']
                    nonlinear_forecasts = deep_learning_model.predict_residuals_direct_nf(residuals_train, len(test_series), input_lags, NBEATS, model_conf['nbeats_params'])
            
            if nonlinear_forecasts is not None:
                final_forecast = pd.Series(linear_forecasts + nonlinear_forecasts, index=test_series.index)
                forecasts_df['previsao'] = final_forecast
                forecasts_df['previsao_arima'] = linear_forecasts
                forecasts_df['previsao_residuos'] = nonlinear_forecasts
            else:
                # Se chegou aqui e nonlin_forecasts é None, significa que o tipo híbrido não foi tratado acima
                raise ValueError(f"Lógica de previsão Híbrida não implementada para: {model_type}")

        else:
            print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de avaliação definida.")
            return

        if final_forecast is None or final_forecast.isnull().any() or np.isinf(final_forecast).any():
             num_invalid = final_forecast.isnull().sum() + np.isinf(final_forecast).sum()
             raise ValueError(f"A previsão gerou {num_invalid} valor(es) inválido(s) (NaN ou Inf).")

        y_true, y_pred, y_train_full = test_series.values, final_forecast.values, train_series.values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not np.all(valid_mask):
             logging.warning(f"Removendo {np.sum(~valid_mask)} par(es) (real, previsto) com NaN antes das métricas.")
             y_true = y_true[valid_mask]
             y_pred = y_pred[valid_mask]
        
        # Usa y_train limpo para denominadores MASE/RMSSE
        y_train_clean = train_series.dropna().values 
        if len(y_train_clean) != len(y_train_full) and len(y_train_clean) > 1:
             naive_errors_in_sample_clean = y_train_clean[1:] - y_train_clean[:-1]
             mean_squared_naive_error = np.mean(np.square(naive_errors_in_sample_clean))
             if mean_squared_naive_error < 1e-9: mean_squared_naive_error = 1e-9

        if len(y_true) == 0: raise ValueError("Nenhum valor válido restante para métricas.")

        if dataset_conf.get('log_transform', False):
            y_true_orig = np.exp(y_true)
            y_pred_orig = np.exp(y_pred)
        else:
            y_true_orig, y_pred_orig = y_true, y_pred

        # Trata possíveis valores zerados ou muito pequenos em y_true_orig antes do MAPE
        zero_mask = np.abs(y_true_orig) < 1e-9
        if np.any(zero_mask):
             logging.warning(f"MAPE: {np.sum(zero_mask)} valor(es) real(is) próximo(s) de zero. MAPE pode ser inflado ou NaN.")
             # Opção: Calcular MAPE apenas onde y_true_orig é não-zero
             mape = mean_absolute_percentage_error(y_true_orig[~zero_mask], y_pred_orig[~zero_mask])
        else:
             mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
             
        mase = mean_absolute_scaled_error(y_true, y_pred, y_train_clean)
        mse_model = mean_squared_error(y_true, y_pred)
        rmsse = np.sqrt(mse_model / mean_squared_naive_error) if mean_squared_naive_error is not np.nan and mean_squared_naive_error > 0 else np.nan

        print(f"Resultados para '{execution_name}':\n  - MAPE: {mape:.4f}\n  - MASE: {mase:.4f}\n  - RMSSE: {rmsse:.4f}")

        metrics_df = pd.DataFrame({
            'execution_name': [execution_name], 'model_type': [model_type],
            'dataset': [dataset_conf['name']], 'MAPE': [mape], 'MASE': [mase],
            'RMSSE': [rmsse]
        })
        metrics_df.to_csv(os.path.join(main_config['results_paths']['metrics'], f"metrics_{execution_name}.csv"), index=False)
        forecasts_df.to_csv(os.path.join(main_config['results_paths']['plots'], f"forecasts_{execution_name}.csv"))

    except Exception as e:
        import traceback
        print(f"\n!!!!!! AVALIAÇÃO FALHOU para '{execution_name}' !!!!!!!")
        print(f"Causa: {e}\n{traceback.format_exc()}\n")