# File: src/models/arima_model.py

import pmdarima as pm
import joblib
import os
import pandas as pd

def train_and_save_arima(series: pd.Series, model_path: str, arima_params: dict):
    """
    Treina um modelo auto-ARIMA e o salva em disco.
    """
    print("Treinando o modelo ARIMA...")
    model = pm.auto_arima(
        series,
        start_p=arima_params['start_p'],
        start_q=arima_params['start_q'],
        max_p=arima_params['max_p'],
        max_q=arima_params['max_q'],
        d=arima_params.get('d'),
        seasonal=arima_params['seasonal'],
        m=arima_params.get('m', 1),
        trace=arima_params['trace'],
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print(f"Melhor modelo ARIMA encontrado: {model.order} {model.seasonal_order if arima_params['seasonal'] else ''}")
    
    joblib.dump(model, model_path)
    print(f"Modelo ARIMA salvo em: {model_path}")
    return model

def load_and_forecast_recursive(model_path: str, horizon: int):
    """
    Carrega um modelo ARIMA e faz uma previsão recursiva de H passos.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo ARIMA não encontrado em {model_path}")
    
    model = joblib.load(model_path)
    forecasts = model.predict(n_periods=horizon)
    return forecasts