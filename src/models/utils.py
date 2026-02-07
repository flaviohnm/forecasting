import pandas as pd
import numpy as np
import logging

def get_fitted_values(model, df_train, horizon=1):
    """
    Recupera as previsões 'in-sample' (nos dados de treino).
    Essencial para calcular os resíduos que treinam o modelo híbrido.
    """
    # 1. StatsForecast (ARIMA, ETS, NAIVE, etc.)
    if 'StatsForecast' in str(type(model)):
        try:
            # --- FIX CRÍTICO PARA STATSFORECAST >= 1.7 ---
            # É obrigatório rodar forecast(fitted=True) antes de pedir os valores.
            # O retorno deste forecast é descartado (_), pois queremos apenas
            # que ele popule os valores internos (side-effect).
            _ = model.forecast(h=horizon, fitted=True)
            
            # Agora sim podemos recuperar os valores ajustados com segurança
            fitted = model.forecast_fitted_values()
            
            # Tratamento de colunas e índices
            if 'unique_id' not in fitted.columns and fitted.index.name == 'unique_id':
                fitted = fitted.reset_index()

            # Identifica a coluna de predição (ex: 'AutoARIMA', 'AutoETS')
            model_col = [c for c in fitted.columns if c not in ['ds', 'unique_id', 'y']][0]
            fitted = fitted.rename(columns={model_col: 'y_hat'})
            
            return fitted[['unique_id', 'ds', 'y', 'y_hat']]
            
        except Exception as e:
            logging.error(f"Erro ao extrair fitted values do StatsForecast: {e}")
            raise e

    # 2. NeuralForecast (N-HiTS, N-BEATS)
    elif hasattr(model, 'predict_insample'):
        fitted = model.predict_insample()
        
        if 'unique_id' not in fitted.columns and fitted.index.name == 'unique_id':
            fitted = fitted.reset_index()
            
        model_col = [c for c in fitted.columns if c not in ['ds', 'unique_id', 'y']][0]
        fitted = fitted.rename(columns={model_col: 'y_hat'})
        return fitted[['unique_id', 'ds', 'y', 'y_hat']]
    
    else:
        # Tenta fallback genérico se não for nenhum dos tipos conhecidos
        raise ValueError(f"Modelo {type(model)} não suporta recuperação de fitted values (in-sample).")

def predict_wrapper(model, horizon, df_history=None):
    """
    Interface unificada para gerar previsões futuras (Out-of-Sample).
    """
    # 1. StatsForecast
    if 'StatsForecast' in str(type(model)):
        forecast = model.predict(h=horizon)
        
        # Garante unique_id como coluna
        if 'unique_id' not in forecast.columns:
            if forecast.index.name == 'unique_id':
                forecast = forecast.reset_index()
            else:
                forecast['unique_id'] = 'ETTh1' # Fallback seguro
                
        model_cols = [c for c in forecast.columns if c not in ['ds', 'unique_id']]
        pred_col = model_cols[0] 
        forecast = forecast.rename(columns={pred_col: 'y_hat'})
        
        return forecast[['unique_id', 'ds', 'y_hat']]

    # 2. NeuralForecast
    elif 'NeuralForecast' in str(type(model)):
        forecast = model.predict() 
        
        if 'unique_id' not in forecast.columns and forecast.index.name == 'unique_id':
            forecast = forecast.reset_index()
            
        model_cols = [c for c in forecast.columns if c not in ['ds', 'unique_id']]
        pred_col = model_cols[0]
        forecast = forecast.rename(columns={pred_col: 'y_hat'})
        return forecast[['unique_id', 'ds', 'y_hat']]

    else:
        raise TypeError(f"Tipo de modelo desconhecido para previsão: {type(model)}")