# File: src/models/ets_model.py

import os
import pandas as pd
from statsmodels.tsa.api import ETSModel
from statsmodels.tsa.exponential_smoothing.ets import ETSResults

def train_and_save_ets(series: pd.Series, model_path: str, ets_params: dict, seasonal_periods: int):
    """
    Treina um modelo ETS e o salva em disco.
    """
    print("Treinando o modelo ETS...")
    
    # Instancia o modelo ETS. 'seasonal_periods' é crucial para séries sazonais.
    model = ETSModel(
        series,
        seasonal_periods=seasonal_periods if seasonal_periods > 1 else None,
        trend=ets_params.get('trend'),
        seasonal=ets_params.get('seasonal'),
        damped_trend=ets_params.get('damped_trend', False)
    )
    
    fitted_model = model.fit()
    print(f"Modelo ETS treinado: {fitted_model.summary()}")
    
    # O statsmodels tem seu próprio método para salvar, que é mais seguro que joblib/pickle
    fitted_model.save(model_path)
    print(f"Modelo ETS salvo em: {model_path}")

def load_and_forecast_ets(model_path: str, horizon: int):
    """
    Carrega um modelo ETS salvo e faz uma previsão.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo ETS não encontrado em '{model_path}'")
    
    # Carrega o modelo usando o método do statsmodels
    loaded_model_results = ETSResults.load(model_path)
    
    # Gera a previsão
    forecast = loaded_model_results.forecast(steps=horizon)
    return forecast