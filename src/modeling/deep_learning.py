import logging

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, Informer

def train_dl_model(df_train_scaled, model_conf, horizon, freq, val_size):
    """
    Treina modelos de Deep Learning usando NeuralForecast (Univariado).
    Contém janelamento dinâmico (3x) e Early Stopping.
    """
    model_type = model_conf["model_type"]
    params = model_conf.get("params", {})
    model_kwargs = params.get("model_kwargs", {})
    
    # --- 1. JANELAMENTO ACADÊMICO (LOOKBACK) ---
    input_size = model_kwargs.get("input_size", int(3 * horizon))

    # --- 2. EARLY STOPPING E OVERFITTING ---
    max_steps = params.get("max_steps", 1000)
    early_stop_patience = 3  
    val_check_steps = max(10, max_steps // 10) 

    model_kwargs_clean = model_kwargs.copy()
    keys_to_remove = ["input_size", "learning_rate", "max_steps"]
    for key in keys_to_remove:
        model_kwargs_clean.pop(key, None)

    clean_model_name = model_type.upper().replace("HYBRID_", "").replace("BASE_", "")

    logging.info(f"Inicializando {clean_model_name} (Univariado) | Horizon: {horizon} | Input_size: {input_size}")

    # FIX 2: Injeção incondicional do modelo para evitar variável vazia
    if clean_model_name == "NBEATS":
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=params.get("learning_rate", 1e-3),
            early_stop_patience_steps=early_stop_patience,
            val_check_steps=val_check_steps,
            scaler_type=None, 
            **model_kwargs_clean,
        )

    elif clean_model_name == "NHITS":
        model = NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=params.get("learning_rate", 1e-3),
            early_stop_patience_steps=early_stop_patience,
            val_check_steps=val_check_steps,
            scaler_type=None,
            **model_kwargs_clean,
        )

    elif clean_model_name == "INFORMER":
        model = Informer(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=params.get("learning_rate", 1e-4),
            early_stop_patience_steps=early_stop_patience,
            val_check_steps=val_check_steps,
            scaler_type=None,
            **model_kwargs_clean,
        )
    else:
        raise ValueError(f"Modelo Neural '{clean_model_name}' não implementado.")

    nf = NeuralForecast(models=[model], freq=freq)

    logging.info(f"Iniciando treinamento com Early Stopping (val_size={val_size})...")
    nf.fit(df=df_train_scaled, val_size=val_size)

    return nf