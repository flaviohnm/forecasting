# /forecasting/src/models.py (VERSÃO DE DEPURACAO)
import pandas as pd
import pmdarima as pm
import joblib
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE

def save_forecasts(df, forecast_dir, model_name, dataset_name):
    # (Esta função não muda)
    output_path = Path(forecast_dir)
    csv_path_dir = output_path / "csv"
    pkl_path_dir = output_path / "pkl"
    csv_path_dir.mkdir(parents=True, exist_ok=True)
    pkl_path_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"{dataset_name}_{model_name}_forecasts"
    csv_path = csv_path_dir / f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    pkl_path = pkl_path_dir / f"{base_filename}.pkl"
    df.to_pickle(pkl_path)
    return str(csv_path)

def create_standard_forecast_df(test_df, forecast_values, dataset_name, target_column, ds_column='ds'):
    """Cria um DataFrame de previsão padronizado."""
    return pd.DataFrame({
        'unique_id': dataset_name,
        'ds': test_df[ds_column].values,
        'actual': test_df[target_column].values,
        'forecast': forecast_values
    })

def train_and_forecast_arima(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality):
    """Orquestra o treino e previsão do modelo ARIMA."""
    train_df = pd.read_csv(train_path, parse_dates=['ds'])
    test_df = pd.read_csv(test_path, parse_dates=['ds'])
    target_column = 'passengers' if dataset_name == 'airline' else 'sunspot_count'
    
    print(f"\n[ARIMA DEBUG] Treinando com {len(train_df)} observações.")
    model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    
    print("\n[ARIMA DEBUG] Sumário do modelo encontrado:")
    print(model.summary())
    
    n_periods = len(test_df)
    forecasts = model.predict(n_periods=n_periods)
    
    # --- LOG DE DEPURACAO ---
    print(f"\n[ARIMA DEBUG] Tipo da previsão: {type(forecasts)}")
    print(f"[ARIMA DEBUG] Comprimento da previsão: {len(forecasts)}")
    print(f"[ARIMA DEBUG] 5 primeiros valores da previsão: \n{forecasts[:5]}")
    # -------------------------

    df_out = create_standard_forecast_df(test_df, forecasts, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "ARIMA", dataset_name)

def train_and_forecast_nbeats_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    """Orquestra o treino e previsão usando o modelo N-BEATS."""
    train_df = pd.read_csv(train_path, parse_dates=['ds'])
    test_df = pd.read_csv(test_path, parse_dates=['ds'])
    horizon = len(test_df)
    target_column = 'passengers' if dataset_name == 'airline' else 'sunspot_count'
    freq = 'M' if dataset_name == 'airline' else 'AS-JAN'

    train_df_fmt = pd.DataFrame({'ds': train_df['ds'], 'y': train_df[target_column], 'unique_id': dataset_name})
    
    models = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=train_df_fmt)
    
    forecasts_df_raw = nf.predict()
    
    # --- LOG DE DEPURACAO ---
    print(f"\n[NBEATS DEBUG] Tipo da previsão: {type(forecasts_df_raw)}")
    print(f"[NBEATS DEBUG] Comprimento da previsão: {len(forecasts_df_raw)}")
    print(f"[NBEATS DEBUG] 5 primeiros valores da previsão: \n{forecasts_df_raw.head()}")
    # -------------------------

    df_out = pd.DataFrame({
        'unique_id': dataset_name,
        'ds': forecasts_df_raw['ds'],
        'actual': test_df[target_column].values,
        'forecast': forecasts_df_raw['NBEATS'].values
    })
    
    return save_forecasts(df_out, forecast_dir, "NBEATS_direct", dataset_name)

def train_and_forecast_hybrid_recursive_direct(train_path, test_path, model_dir, forecast_dir, dataset_name, seasonality, max_steps):
    """Orquestra o treino e previsão do modelo híbrido."""
    train_df = pd.read_csv(train_path, parse_dates=['ds'])
    test_df = pd.read_csv(test_path, parse_dates=['ds'])
    horizon = len(test_df)
    target_column = 'passengers' if dataset_name == 'airline' else 'sunspot_count'
    freq = 'M' if dataset_name == 'airline' else 'AS-JAN'

    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    residuals_train = pd.Series(arima_model.resid())
    arima_forecasts = arima_model.predict(n_periods=horizon)

    residuals_train_df_fmt = pd.DataFrame({'ds': train_df['ds'][:len(residuals_train)], 'y': residuals_train, 'unique_id': f'{dataset_name}_residuals'})
    
    nbeats_residuals_model = [NBEATS(input_size=2 * horizon, h=horizon, loss=MAE(), max_steps=max_steps, random_seed=42)]
    nf_residuals = NeuralForecast(models=nbeats_residuals_model, freq=freq)
    nf_residuals.fit(df=residuals_train_df_fmt)
    
    residuals_forecasts_raw = nf_residuals.predict()
    residuals_forecasts = residuals_forecasts_raw['NBEATS'].values

    # --- LOG DE DEPURACAO ---
    print(f"\n[HYBRID DEBUG] Comprimento previsão ARIMA: {len(arima_forecasts)}")
    print(f"[HYBRID DEBUG] 5 primeiros valores ARIMA: \n{arima_forecasts[:5]}")
    print(f"\n[HYBRID DEBUG] Comprimento previsão Resíduos: {len(residuals_forecasts)}")
    print(f"[HYBRID DEBUG] 5 primeiros valores Resíduos: \n{residuals_forecasts[:5]}")
    # -------------------------
    
    final_forecast = arima_forecasts + residuals_forecasts
    df_out = create_standard_forecast_df(test_df, final_forecast, dataset_name, target_column)
    return save_forecasts(df_out, forecast_dir, "Hybrid_RecursiveDirect", dataset_name)