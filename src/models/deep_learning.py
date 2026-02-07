import pandas as pd
import logging
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MAE

def train_dl_model(df, model_conf, horizon, freq, val_size):
    """
    Treina modelos de Deep Learning (N-HiTS, N-BEATS) usando NeuralForecast.
    """
    # 1. Extração de Parâmetros do YAML
    params = model_conf.get('params', {})
    
    # Separa argumentos do Modelo (arquitetura) e do Treinador (epochs, learning rate)
    model_kwargs = params.get('model_kwargs', {}).copy()
    trainer_kwargs = params.get('trainer_kwargs', {}).copy() # <--- AQUI ESTÁ O SEGREDO
    
    # Definição do Modelo
    model_type = model_conf['model_type']
    
    # Configuração comum para todos os modelos
    # H = Horizonte de previsão
    # Input Size = Definido no YAML ou padrão 3xH
    input_size = model_kwargs.get('input_size', horizon * 3)
    
    # Lista de modelos suportados
    models_list = []
    
    if 'NHiTS' in model_type:
        # Removemos input_size do kwargs para passar explícito
        if 'input_size' in model_kwargs: del model_kwargs['input_size']
            
        model = NHITS(
            h=horizon,
            input_size=input_size,
            loss=MAE(),
            scaler_type='standard',
            **model_kwargs,    # Passa mlp_units, n_blocks, etc.
            **trainer_kwargs   # <--- CORREÇÃO: Passa max_steps, early_stop, etc.
        )
        models_list.append(model)
        
    elif 'NBEATS' in model_type:
        if 'input_size' in model_kwargs: del model_kwargs['input_size']
            
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            loss=MAE(),
            scaler_type='standard',
            **model_kwargs,
            **trainer_kwargs   # <--- CORREÇÃO: Passa max_steps aqui também
        )
        models_list.append(model)
        
    else:
        raise ValueError(f"Modelo Deep Learning desconhecido: {model_type}")

    # 2. Instancia o Orquestrador NeuralForecast
    nf = NeuralForecast(
        models=models_list,
        freq=freq
    )
    
    # 3. Treinamento
    # O NeuralForecast faz o split interno baseado no val_size se fornecido, 
    # mas aqui passamos o DF completo e ele treina conforme a configuração do modelo.
    # Para validar cross-validation, o ideal seria usar o cross_validation do próprio NF,
    # mas para manter compatibilidade com nossa pipeline sequencial, usamos fit direto.
    
    logging.info(f"Treinando {model_type} com h={horizon}, input={input_size}, steps={trainer_kwargs.get('max_steps', 'Default')}")
    
    nf.fit(df=df, val_size=val_size)
    
    return nf