import pandas as pd
import logging
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, Informer
from neuralforecast.losses.pytorch import MAE

def train_dl_model(df_train_scaled, model_conf, horizon, freq, val_size):
    """
    Treina modelos de Deep Learning usando NeuralForecast.
    """
    model_type = model_conf['model_type']
    params = model_conf.get('params', {})
    
    # Recupera kwargs extras
    model_kwargs = params.get('model_kwargs', {})
    
    # Input size calculado dinamicamente no pipeline
    # Se não houver, usa o horizonte como default
    input_size = model_kwargs.get('input_size', horizon)

    # --- CORREÇÃO DO ERRO "MULTIPLE VALUES" ---
    # Criamos uma cópia limpa dos argumentos para remover 'input_size'
    # já que vamos passá-lo explicitamente logo abaixo.
    model_kwargs_clean = model_kwargs.copy()
    if 'input_size' in model_kwargs_clean:
        del model_kwargs_clean['input_size']

    logging.info(f"Inicializando {model_type} com Input={input_size}, Horizon={horizon}")

    # --- SELEÇÃO DE MODELOS ---
    if model_type == 'NBEATS':
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            max_steps=params.get('max_steps', 1000),
            learning_rate=params.get('learning_rate', 1e-3),
            **model_kwargs_clean  # Usamos a versão limpa
        )
    
    elif model_type == 'NHITS':
        model = NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=params.get('max_steps', 1000),
            learning_rate=params.get('learning_rate', 1e-3),
            **model_kwargs_clean # Usamos a versão limpa
        )

    elif model_type == 'Informer':
        model = Informer(
            h=horizon,
            input_size=input_size,
            max_steps=params.get('max_steps', 1000),
            learning_rate=params.get('learning_rate', 1e-4),
            scaler_type=None, # Importante: Desligado pois já escalamos fora
            **model_kwargs_clean # Usamos a versão limpa
        )
        
    else:
        raise ValueError(f"Modelo Neural {model_type} não implementado em deep_learning.py")

    # Configuração do NeuralForecast
    nf = NeuralForecast(
        models=[model],
        freq=freq
    )

    # Treinamento
    nf.fit(df=df_train_scaled, val_size=val_size)
    
    return nf