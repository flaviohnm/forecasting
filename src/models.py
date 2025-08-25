# VERSÃO DEFINITIVA E CORRIGIDA do models.py
import pandas as pd
import pmdarima as pm
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, LSTM, MLP, iTransformer, NHITS

# --- Funções Auxiliares ---
def prepare_df_dl(df_time, y_series, unique_id, ds_column='ds'):
    y_values = np.array(y_series)
    return pd.DataFrame({'ds': df_time[ds_column], 'y': y_values.astype('float32'), 'unique_id': unique_id})

# --- Modelos Clássicos e Baselines ---
def train_and_forecast_naive(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, **kwargs):
    last_value = train_df[target_column].iloc[-1]
    return np.full(len(test_df), fill_value=last_value)

def train_and_forecast_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, seasonality: int, target_column: str, **kwargs):
    model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=True, suppress_warnings=True, error_action='ignore')
    return model.predict(n_periods=len(test_df)).values

def train_and_forecast_ets(train_df: pd.DataFrame, test_df: pd.DataFrame, seasonality: int, target_column: str, freq: str, **kwargs):
    train_series = train_df.set_index('ds')[target_column].asfreq(freq)
    model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=seasonality).fit()
    return model.forecast(steps=len(test_df)).values

# =====================================================================================
# --- Lógica Centralizada e Corrigida para Modelos de Deep Learning ---
# =====================================================================================

def _filter_dl_params(all_params: dict):
    """
    Filtra os kwargs para separar os parâmetros do modelo e do treinador,
    removendo chaves internas do pipeline.
    """
    INTERNAL_KEYS = ['train_df', 'test_df', 'target_column', 'dataset_name', 'seasonality', 'is_mimo', 'model_cls', 'model_name', 'freq','activation','forecast_horizon']
    KNOWN_TRAINER_KEYS = [
        'gpus', 'accelerator', 'batch_size', 'num_workers',
        'val_check_steps', 'early_stop_patience_steps'
    ]
    
    trainer_params = {}
    model_params = {}

    for key, value in all_params.items():
        if key in KNOWN_TRAINER_KEYS:
            trainer_params[key] = value
        elif key not in INTERNAL_KEYS:
            model_params[key] = value
            
    return model_params, trainer_params

def run_deep_learning_model(model_cls, model_name, train_df, test_df, is_mimo, **kwargs):
    """
    Função genérica e robusta para treinar qualquer modelo do NeuralForecast.
    """
    target_column = kwargs['target_column']
    dataset_name = kwargs['dataset_name']
    freq = kwargs['freq']
    horizon = len(test_df)
    
    model_params, trainer_params = _filter_dl_params(kwargs)

    if is_mimo:
        train_df_fmt = prepare_df_dl(train_df, train_df[target_column], dataset_name)
        model_params.setdefault('input_size', 2 * horizon)
        
        models = [model_cls(h=horizon, **model_params)]
        nf = NeuralForecast(models=models, freq=freq, **trainer_params)
        nf.fit(df=train_df_fmt)
        return nf.predict()[model_name].values
    else:  # Estratégia Direta
        all_forecasts = []
        for h in range(1, horizon + 1):
            target_series = train_df[target_column].shift(-h).dropna()
            train_df_h = train_df.iloc[:len(target_series)]
            train_df_fmt_h = prepare_df_dl(train_df_h, target_series, f'{dataset_name}_h{h}')
            
            model_params_h = model_params.copy()
            model_params_h.setdefault('input_size', 2 * horizon)
            
            models = [model_cls(h=1, **model_params_h)]
            nf = NeuralForecast(models=models, freq=freq, **trainer_params.copy())
            nf.fit(df=train_df_fmt_h)
            
            forecast_h = nf.predict()[model_name].values[0]
            all_forecasts.append(forecast_h)
        return np.array(all_forecasts)

# --- Interfaces para os modelos ---

def train_and_forecast_mimo_mlp(train_df, test_df, **kwargs):
    return run_deep_learning_model(MLP, 'MLP', train_df, test_df, is_mimo=True, **kwargs)
def train_and_forecast_direct_mlp(train_df, test_df, **kwargs):
    return run_deep_learning_model(MLP, 'MLP', train_df, test_df, is_mimo=False, **kwargs)
def train_and_forecast_nbeats_mimo(train_df, test_df, **kwargs):
    return run_deep_learning_model(NBEATS, 'NBEATS', train_df, test_df, is_mimo=True, **kwargs)
def train_and_forecast_nbeats_direct(train_df, test_df, **kwargs):
    return run_deep_learning_model(NBEATS, 'NBEATS', train_df, test_df, is_mimo=False, **kwargs)
def train_and_forecast_nhits_mimo(train_df, test_df, **kwargs):
    return run_deep_learning_model(NHITS, 'NHITS', train_df, test_df, is_mimo=True, **kwargs)
def train_and_forecast_nhits_direct(train_df, test_df, **kwargs):
    return run_deep_learning_model(NHITS, 'NHITS', train_df, test_df, is_mimo=False, **kwargs)
def train_and_forecast_lstm_mimo(train_df, test_df, **kwargs):
    return run_deep_learning_model(LSTM, 'LSTM', train_df, test_df, is_mimo=True, **kwargs)
def train_and_forecast_lstm_direct(train_df, test_df, **kwargs):
    return run_deep_learning_model(LSTM, 'LSTM', train_df, test_df, is_mimo=False, **kwargs)
def train_and_forecast_transformer_mimo(train_df, test_df, **kwargs):
    return run_deep_learning_model(iTransformer, 'iTransformer', train_df, test_df, is_mimo=True, **kwargs)

# --- Modelos Híbridos ---
def run_hybrid_strategy_wrapper(model_cls, model_name, is_mimo, train_df, test_df, **kwargs):
    horizon = len(test_df)
    seasonality = kwargs['seasonality']
    target_column = kwargs['target_column']
    
    print("   [Híbrido] Treinando modelo ARIMA...")
    arima_model = pm.auto_arima(train_df[target_column], m=seasonality, seasonal=True, trace=False, suppress_warnings=True, error_action='ignore')
    arima_forecasts = arima_model.predict(n_periods=horizon)
    residuals_train = pd.Series(arima_model.resid()).astype('float32')
    
    print(f"   [Híbrido] Treinando modelo {model_name} nos resíduos...")
    residuals_train_df = pd.DataFrame({'ds': train_df['ds'].iloc[:len(residuals_train)], 'residuals': residuals_train})
    
    hybrid_kwargs = kwargs.copy()
    hybrid_kwargs['target_column'] = 'residuals'
    hybrid_kwargs['dataset_name'] = f"{kwargs['dataset_name']}_residuals"
    
    residuals_forecasts = run_deep_learning_model(
        model_cls=model_cls, model_name=model_name,
        train_df=residuals_train_df, test_df=test_df,
        is_mimo=is_mimo, **hybrid_kwargs
    )
    return arima_forecasts.values + residuals_forecasts

# --- Interfaces para Modelos Híbridos ---
def train_and_forecast_hybrid_mimo_mlp(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(MLP, 'MLP', True, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_direct_mlp(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(MLP, 'MLP', False, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_mimo_nbeats(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(NBEATS, 'NBEATS', True, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_direct_nbeats(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(NBEATS, 'NBEATS', False, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_mimo_nhits(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(NHITS, 'NHITS', True, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_direct_nhits(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(NHITS, 'NHITS', False, train_df, test_df, **kwargs)
def train_and_forecast_hybrid_recursive_lstm(train_df, test_df, **kwargs):
    return run_hybrid_strategy_wrapper(LSTM, 'LSTM', True, train_df, test_df, **kwargs)