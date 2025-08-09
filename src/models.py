# /forecasting/src/models.py (VERSÃO COM TRATAMENTO DE ERRO ROBUSTO)
import pandas as pd
import pmdarima as pm
import joblib
from pathlib import Path
import numpy as np
from statsmodels.tsa.api import VAR
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE

def save_forecasts(df, forecast_dir, model_name, dataset_name):
    # (Esta função não muda)
    output_path = Path(forecast_dir)
    csv_path_dir = output_path / "csv"; csv_path_dir.mkdir(parents=True, exist_ok=True)
    pkl_path_dir = output_path / "pkl"; pkl_path_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"{dataset_name}_{model_name}_forecasts"
    csv_path = csv_path_dir / f"{base_filename}.csv"; df.to_csv(csv_path, index=False)
    pkl_path = pkl_path_dir / f"{base_filename}.pkl"; df.to_pickle(pkl_path)
    return str(csv_path)

def create_standard_forecast_df(test_df, forecast_values, dataset_name, target_column, ds_column='ds'):
    # (Esta função não muda)
    return pd.DataFrame({'unique_id': dataset_name, 'ds': test_df[ds_column].values, 'actual': test_df[target_column].values, 'forecast': forecast_values})

def _prepare_nbeats_df(df_time, y_series, unique_id, ds_column='ds'):
    # (Esta função não muda)
    return pd.DataFrame({'ds': df_time[ds_column], 'y': y_series.astype('float32'), 'unique_id': unique_id})

def train_and_forecast_arima(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality):
    try:
        train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(test_path, parse_dates=['ds'])
        target_column = 'passengers' if dataset_name == 'airline' else 'sunspot_count'
        model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
        forecasts = model.predict(n_periods=len(test_df))
        df_out = create_standard_forecast_df(test_df, forecasts, dataset_name, target_column)
        return save_forecasts(df_out, forecast_dir, "ARIMA", dataset_name)
    except Exception as e:
        print(f"ERRO ao treinar o modelo ARIMA: {e}")
        return None

def train_and_forecast_nbeats_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    try:
        train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(test_path, parse_dates=['ds'])
        horizon = len(test_df)
        target_column = 'passengers' if dataset_name == 'airline' else 'sunspot_count'
        freq = 'M' if dataset_name == 'airline' else 'AS-JAN'
        train_df_fmt = _prepare_nbeats_df(train_df, train_df[target_column], dataset_name)
        models = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt)
        forecasts_df_raw = nf.predict()
        df_out = create_standard_forecast_df(test_df, forecasts_df_raw['NBEATS'].values, dataset_name, target_column)
        return save_forecasts(df_out, forecast_dir, "NBEATS_direct", dataset_name)
    except Exception as e:
        print(f"ERRO ao treinar o modelo NBEATS_direct: {e}")
        return None

def train_and_forecast_hybrid_recursive_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    try:
        train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(test_path, parse_dates=['ds'])
        horizon = len(test_df)
        target_column = 'passengers'
        freq = 'M'
        arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
        arima_forecasts = arima_model.predict(n_periods=horizon)
        residuals_train = pd.Series(arima_model.resid()).astype('float32')
        residuals_train_df_fmt = _prepare_nbeats_df(train_df.iloc[:len(residuals_train)], residuals_train, f'{dataset_name}_residuals')
        nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
        nf_residuals.fit(df=residuals_train_df_fmt)
        residuals_forecasts = nf_residuals.predict()['NBEATS'].values
        final_forecast = arima_forecasts + residuals_forecasts
        df_out = create_standard_forecast_df(test_df, final_forecast, dataset_name, target_column)
        return save_forecasts(df_out, forecast_dir, "Hybrid_RecursiveDirect", dataset_name)
    except Exception as e:
        print(f"ERRO ao treinar o modelo Hybrid_RecursiveDirect: {e}")
        return None

def train_and_forecast_hybrid_mimo(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    try:
        train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(test_path, parse_dates=['ds'])
        horizon = len(test_df)
        target_column = 'passengers'
        freq = 'M'
        var_model = VAR(train_df[[target_column]])
        try:
            var_results = var_model.fit(maxlags=seasonality*2, ic='aic')
        except Exception as e:
            var_results = var_model.fit(seasonality)
        lag_order = var_results.k_ar
        var_forecasts = var_results.forecast(y=train_df[[target_column]].values[-lag_order:], steps=horizon)
        var_forecasts = var_forecasts[:, 0]
        var_train_preds = var_results.fittedvalues[target_column]
        residuals_train = (train_df[target_column].iloc[lag_order:] - var_train_preds).astype('float32')
        residuals_train_df_fmt = _prepare_nbeats_df(train_df.iloc[lag_order:], residuals_train, f'{dataset_name}_residuals')
        nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
        nf_residuals.fit(df=residuals_train_df_fmt)
        residuals_forecasts = nf_residuals.predict()['NBEATS'].values
        final_forecast = var_forecasts + residuals_forecasts
        df_out = create_standard_forecast_df(test_df, final_forecast, dataset_name, target_column)
        return save_forecasts(df_out, forecast_dir, "Hybrid_MIMO", dataset_name)
    except Exception as e:
        print(f"ERRO ao treinar o modelo Hybrid_MIMO: {e}")
        return None