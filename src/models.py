# /forecasting/src/models.py (VERSÃO REATORADA E COMPLETA)
import pandas as pd
import pmdarima as pm
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, LSTM, MLP, iTransformer, NHITS
from neuralforecast.losses.pytorch import MAE

# --- Função Auxiliar (Apenas a necessária para a modelagem) ---
def prepare_df_dl(df_time, y_series, unique_id, ds_column='ds'):
    """Prepara um DataFrame para o formato exigido pela NeuralForecast."""
    return pd.DataFrame({'ds': df_time[ds_column], 'y': y_series.astype('float32'), 'unique_id': unique_id})

# --- Modelos Puros (Baselines) ---
def train_and_forecast_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, seasonality: int, target_column: str, **kwargs):
    model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    forecasts = model.predict(n_periods=len(test_df))
    return forecasts.values

def train_and_forecast_ets(train_df: pd.DataFrame, test_df: pd.DataFrame, seasonality: int, target_column: str, **kwargs):
    train_series = train_df.set_index('ds')[target_column]
    model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=seasonality).fit()
    forecasts = model.forecast(steps=len(test_df))
    return forecasts.values

def train_and_forecast_nbeats_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [NBEATS(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42, stack_types=['identity'])]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['NBEATS'].values[0]
        all_forecasts.append(forecast_h)
    return np.array(all_forecasts)

def train_and_forecast_nbeats_mimo(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
    models = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    return forecasts_df_raw['NBEATS'].values

def train_and_forecast_nhits_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    all_forecasts = []
    print("[NHiTS-Direct] Treinando 'h' modelos, um para cada passo do horizonte...")
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [NHITS(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['NHITS'].values[0]
        all_forecasts.append(forecast_h)
    return np.array(all_forecasts)

def train_and_forecast_nhits_mimo(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
    models = [NHITS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    return forecasts_df_raw['NHITS'].values

def train_and_forecast_lstm_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [LSTM(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['LSTM'].values[0]
        all_forecasts.append(forecast_h)
    return np.array(all_forecasts)

def train_and_forecast_lstm_mimo(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
    models = [LSTM(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    return forecasts_df_raw['LSTM'].values

def train_and_forecast_mlp_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [MLP(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['MLP'].values[0]
        all_forecasts.append(forecast_h)
    return np.array(all_forecasts)

def train_and_forecast_mlp_mimo(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
    models = [MLP(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    return forecasts_df_raw['MLP'].values

def train_and_forecast_transformer_mimo(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
    models = [iTransformer(input_size=2 * horizon, h=horizon, n_series=1, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    return forecasts_df_raw['iTransformer'].values
    
# --- Modelos Híbridos ---
def train_and_forecast_hybrid_direct_nbeats(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    all_residuals_forecasts = []
    for h in range(1, horizon + 1):
        target_residuals = residuals_train.shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_residuals)]
        residuals_train_df_fmt_h = prepare_df_dl(train_df_h, target_residuals, f'{dataset_name}_residuals_h{h}')
        nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42, stack_types=['identity'])]
        nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
        nf_residuals.fit(df=residuals_train_df_fmt_h)
        forecast_h = nf_residuals.predict()['NBEATS'].values[0]
        all_residuals_forecasts.append(forecast_h)
    residuals_forecasts = np.array(all_residuals_forecasts)
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast

def train_and_forecast_hybrid_mimo_nbeats(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(residuals_train)], residuals_train, f'{dataset_name}_residuals')
    nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['NBEATS'].values
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast
    
def train_and_forecast_hybrid_arima_mlp(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(residuals_train)], residuals_train, f'{dataset_name}_residuals')
    mlp_residuals_model = [MLP(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=mlp_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['MLP'].values
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast

def train_and_forecast_hybrid_recursive_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(residuals_train)], residuals_train, f'{dataset_name}_residuals')
    lstm_residuals_model = [LSTM(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=lstm_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['LSTM'].values
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast

def train_and_forecast_hybrid_direct_nhits(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    all_residuals_forecasts = []
    for h in range(1, horizon + 1):
        target_residuals = residuals_train.shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_residuals)]
        residuals_train_df_fmt_h = prepare_df_dl(train_df_h, target_residuals, f'{dataset_name}_residuals_h{h}')
        nhits_residuals_model = [NHITS(input_size=2 * horizon, h=1, loss=MAE(), max_steps=max_steps, random_seed=42)]
        nf_residuals = NeuralForecast(models=nhits_residuals_model, freq=freq)
        nf_residuals.fit(df=residuals_train_df_fmt_h)
        forecast_h = nf_residuals.predict()['NHITS'].values[0]
        all_residuals_forecasts.append(forecast_h)
    residuals_forecasts = np.array(all_residuals_forecasts)
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast

def train_and_forecast_hybrid_mimo_nhits(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str, seasonality: int, max_steps: int, target_column: str, freq: str, **kwargs):
    horizon = len(test_df)
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(residuals_train)], residuals_train, f'{dataset_name}_residuals')
    nhits_residuals_model = [NHITS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=nhits_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['NHITS'].values
    final_forecast = arima_forecasts.values + residuals_forecasts
    return final_forecast