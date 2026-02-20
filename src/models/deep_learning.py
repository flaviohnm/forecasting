import pandas as pd
import logging
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, Informer
from neuralforecast.losses.pytorch import MAE


def train_dl_model(df_train_scaled, model_conf, horizon, freq, val_size):
    """
    Treina modelos de Deep Learning usando NeuralForecast.
    """
    model_type = model_conf["model_type"]
    params = model_conf.get("params", {})

    model_kwargs = params.get("model_kwargs", {})

    input_size = model_kwargs.get("input_size", horizon)

    model_kwargs_clean = model_kwargs.copy()
    keys_to_remove = ["input_size", "learning_rate", "max_steps"]

    for key in keys_to_remove:
        model_kwargs_clean.pop(
            key, None
        )  # Remove se existir, sem dar erro se não existir

    # Transforma "Hybrid_NHITS" em "NHITS" e coloca em maiúsculo
    clean_model_name = model_type.upper().replace("HYBRID_", "")

    logging.info(
        f"Inicializando {clean_model_name} (Original: {model_type}) com Input={input_size}, Horizon={horizon}"
    )

    # --- SELEÇÃO DE MODELOS ---
    if clean_model_name == "NBEATS":
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            max_steps=params.get("max_steps", 1000),
            learning_rate=params.get("learning_rate", 1e-3),
            **model_kwargs_clean,
        )

    elif clean_model_name == "NHITS":
        model = NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=params.get("max_steps", 1000),
            learning_rate=params.get("learning_rate", 1e-3),
            **model_kwargs_clean,
        )

    elif clean_model_name == "INFORMER":
        model = Informer(
            h=horizon,
            input_size=input_size,
            max_steps=params.get("max_steps", 1000),
            learning_rate=params.get("learning_rate", 1e-4),
            scaler_type=None,
            **model_kwargs_clean,
        )

    else:
        raise ValueError(
            f"Modelo Neural '{clean_model_name}' não implementado em deep_learning.py"
        )

    # Configuração do NeuralForecast
    nf = NeuralForecast(models=[model], freq=freq)

    # Treinamento
    nf.fit(df=df_train_scaled, val_size=val_size)

    return nf
