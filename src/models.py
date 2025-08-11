# /forecasting/src/models.py (VERSÃO COMPLETA E CORRIGIDA COM TODOS OS MODELOS)
import pandas as pd
import pmdarima as pm
import joblib
from pathlib import Path
import numpy as np
from neuralforecast import NeuralForecast
# --- MUDANÇA AQUI: Importando todos os modelos necessários ---
from neuralforecast.models import NBEATS, LSTM, MLP
from neuralforecast.losses.pytorch import MAE


def save_forecasts(df, forecast_dir, model_name, dataset_name):
    """Função auxiliar para salvar previsões nos formatos CSV e PKL."""
    output_path = Path(forecast_dir)
    csv_path_dir = output_path / "csv"
    csv_path_dir.mkdir(parents=True, exist_ok=True)
    pkl_path_dir = output_path / "pkl"
    pkl_path_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"{dataset_name}_{model_name}_forecasts"
    csv_path = csv_path_dir / f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    pkl_path = pkl_path_dir / f"{base_filename}.pkl"
    df.to_pickle(pkl_path)
    return str(csv_path)


def prepare_df_dl(df_time, y_series, unique_id, ds_column='ds'):
    """Prepara o DataFrame para o formato exigido pela NeuralForecast."""
    return pd.DataFrame({'ds': df_time[ds_column], 'y': y_series.astype('float32'), 'unique_id': unique_id})


def create_standard_forecast_df(test_df, forecast_values, dataset_name, target_column, ds_column='ds'):
    """Cria um DataFrame de previsão padronizado."""
    return pd.DataFrame({'unique_id': dataset_name, 'ds': test_df[ds_column].values, 'actual': test_df[target_column].values, 'forecast': forecast_values})

# --- Modelos Puros (Baselines) ---


def train_and_forecast_arima(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    target_column = 'passengers'
    model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True,
                          trace=False, suppress_warnings=True, error_action='ignore')
    forecasts = model.predict(n_periods=len(test_df))
    df_out = create_standard_forecast_df(
        test_df, forecasts, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "ARIMA", dataset_name)


def train_and_forecast_nbeats_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(
            train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [NBEATS(input_size=2 * horizon, h=1, loss=MAE(),
                         max_steps=max_steps, random_seed=42, stack_types=['identity'])]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['NBEATS'].values[0]
        all_forecasts.append(forecast_h)
    forecasts = np.array(all_forecasts)
    df_out = create_standard_forecast_df(
        test_df, forecasts, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "NBEATS_Direct", dataset_name)


def train_and_forecast_nbeats_mimo(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    train_df_fmt = prepare_df_dl(
        train_df, train_df[target_column], dataset_name)
    models = [NBEATS(input_size=2 * horizon, h=horizon,
                     loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    df_out = create_standard_forecast_df(
        test_df, forecasts_df_raw['NBEATS'].values, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "NBEATS_MIMO", dataset_name)


def train_and_forecast_lstm_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(
            train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [LSTM(input_size=2 * horizon, h=1, loss=MAE(),
                       max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['LSTM'].values[0]
        all_forecasts.append(forecast_h)
    forecasts = np.array(all_forecasts)
    df_out = create_standard_forecast_df(
        test_df, forecasts, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "LSTM_Direct", dataset_name)


def train_and_forecast_lstm_mimo(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    train_df_fmt = prepare_df_dl(
        train_df, train_df[target_column], dataset_name)
    models = [LSTM(input_size=2 * horizon, h=horizon, loss=MAE(),
                   max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    df_out = create_standard_forecast_df(
        test_df, forecasts_df_raw['LSTM'].values, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "LSTM_MIMO", dataset_name)


def train_and_forecast_mlp_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    """Orquestra o treino e previsão usando MLP com estratégia Direta Fiel."""
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    all_forecasts = []
    for h in range(1, horizon + 1):
        target_series = train_df[target_column].shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_series)]
        train_df_fmt_h = prepare_df_dl(
            train_df_h, target_series, f'{dataset_name}_h{h}')
        models = [MLP(input_size=2 * horizon, h=1, loss=MAE(),
                      max_steps=max_steps, random_seed=42)]
        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train_df_fmt_h)
        forecast_h = nf.predict()['MLP'].values[0]
        all_forecasts.append(forecast_h)
    forecasts = np.array(all_forecasts)
    df_out = create_standard_forecast_df(
        test_df, forecasts, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "MLP_Direct", dataset_name)


def train_and_forecast_mlp_mimo(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    """Orquestra o treino e previsão usando MLP com estratégia MIMO."""
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    train_df_fmt = prepare_df_dl(
        train_df, train_df[target_column], dataset_name)
    models = [MLP(input_size=2 * horizon, h=horizon, loss=MAE(),
                  max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    forecasts_df_raw = nf.predict()
    df_out = create_standard_forecast_df(
        test_df, forecasts_df_raw['MLP'].values, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "MLP_MIMO", dataset_name)

# --- Modelos Híbridos ---


def train_and_forecast_hybrid_direct_nbeats(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality,
                                seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    all_residuals_forecasts = []
    for h in range(1, horizon + 1):
        target_residuals = residuals_train.shift(-h).dropna()
        train_df_h = train_df.iloc[:len(target_residuals)]
        residuals_train_df_fmt_h = prepare_df_dl(
            train_df_h, target_residuals, f'{dataset_name}_residuals_h{h}')
        nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=1, loss=MAE(
        ), max_steps=max_steps, random_seed=42, stack_types=['identity'])]
        nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
        nf_residuals.fit(df=residuals_train_df_fmt_h)
        forecast_h = nf_residuals.predict()['NBEATS'].values[0]
        all_residuals_forecasts.append(forecast_h)
    residuals_forecasts = np.array(all_residuals_forecasts)
    final_forecast = arima_forecasts + residuals_forecasts
    df_out = create_standard_forecast_df(
        test_df, final_forecast, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "Hybrid_Direct_N-BEATS", dataset_name)


def train_and_forecast_hybrid_mimo_nbeats(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality,
                                seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(
        residuals_train)], residuals_train, f'{dataset_name}_residuals')
    nbeats_residuals_model = [NBEATS(
        input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['NBEATS'].values
    final_forecast = arima_forecasts + residuals_forecasts
    df_out = create_standard_forecast_df(
        test_df, final_forecast, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "Hybrid_MIMO_N-BEATS", dataset_name)


def train_and_forecast_hybrid_arima_mlp(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    """Implementa o híbrido ARIMA (Recursivo) + MLP (MIMO nos resíduos)."""
    train_df, test_df = pd.read_csv(train_path, parse_dates=['ds']), pd.read_csv(
        test_path, parse_dates=['ds'])
    horizon, target_column, freq = len(test_df), 'passengers', 'ME'

    arima_model = pm.auto_arima(train_df[target_column], m=seasonality,
                                seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')

    residuals_train_df_fmt = prepare_df_dl(train_df.iloc[:len(
        residuals_train)], residuals_train, f'{dataset_name}_residuals')
    mlp_residuals_model = [MLP(
        input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=mlp_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    residuals_forecasts = nf_residuals.predict()['MLP'].values

    final_forecast = arima_forecasts + residuals_forecasts
    df_out = create_standard_forecast_df(
        test_df, final_forecast, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "Hybrid_ARIMA_MLP", dataset_name)
