# File: src/models/ets_model.py

import os
import pandas as pd
from statsmodels.tsa.api import ETSModel
from statsmodels.tsa.exponential_smoothing.ets import ETSResults
import logging


def train_and_save_ets(series: pd.Series, model_path: str, ets_params: dict,
                       seasonal_periods: int):
    """
    Treina um modelo ETS e o salva em disco.
    """
    print("Treinando o modelo ETS...")

    # Converte 1 (ou 0) para None, que é o que statsmodels espera para não-sazonal.
    ets_seasonal_periods = seasonal_periods if seasonal_periods > 1 else None

    # Determina o componente sazonal ('add', 'mul', ou None)
    ets_seasonal = ets_params.get('seasonal')

    # Se os períodos forem None, o componente sazonal também DEVE ser None.
    if ets_seasonal_periods is None:
        if ets_seasonal is not None:
            # Apenas avisa se o trace estiver ativado ou se for um warning crítico
            if ets_params.get('trace', False):
                print(
                    f"  Aviso ETS: Período sazonal é 1. Forçando componente 'seasonal' para None (era '{ets_seasonal}')."
                )
        ets_seasonal = None
    elif ets_seasonal is None:
        ets_seasonal = 'add'

    try:
        model = ETSModel(series,
                         seasonal_periods=ets_seasonal_periods,
                         trend=ets_params.get('trend'),
                         seasonal=ets_seasonal,
                         damped_trend=ets_params.get('damped_trend', False))

        fitted_model = model.fit()

        # --- LÓGICA DE TRACE CONFIGURÁVEL ---
        if ets_params.get('trace', False):
            print(f"Modelo ETS treinado: {fitted_model.summary()}")

        fitted_model.save(model_path)
        print(f"Modelo ETS salvo em: {model_path}")

    except Exception as e:
        logging.error(f"Falha ao treinar modelo ETS para {series.name}: {e}")
        # Tenta um modelo mais simples (sem sazonalidade) como fallback
        if ets_seasonal_periods is not None:
            print("  Tentando fallback do ETS sem sazonalidade...")
            try:
                model_fallback = ETSModel(series,
                                          seasonal_periods=None,
                                          seasonal=None,
                                          trend=ets_params.get('trend'),
                                          damped_trend=ets_params.get(
                                              'damped_trend', False))
                fitted_model_fallback = model_fallback.fit()

                # Logica de trace para o fallback também
                if ets_params.get('trace', False):
                    print(
                        f"Modelo ETS (Fallback) treinado: {fitted_model_fallback.summary()}"
                    )

                fitted_model_fallback.save(model_path)
                print(f"Modelo ETS (Fallback) salvo em: {model_path}")
            except Exception as e_fallback:
                logging.error(f"Falha no fallback do ETS: {e_fallback}")
                raise e_fallback
        else:
            raise e


def load_and_forecast_ets(model_path: str, horizon: int):
    """
    Carrega um modelo ETS salvo e faz uma previsão.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo ETS não encontrado em '{model_path}'")

    loaded_model_results = ETSResults.load(model_path)

    forecast = loaded_model_results.forecast(steps=horizon)
    return forecast
