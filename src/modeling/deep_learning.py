import logging

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATS, AutoNHITS  # Novos imports para HPO
from neuralforecast.models import NBEATS, NHITS, Informer


def train_dl_model(df_train_scaled, model_conf, horizon, freq, val_size):
    """
    Treina modelos de Deep Learning usando NeuralForecast.
    Suporta instanciação estática ou Otimização de Hiperparâmetros (Auto-Tune).
    """
    model_type = model_conf["model_type"]
    params = model_conf.get("params", {})
    model_kwargs = params.get("model_kwargs", {})

    # --- Configurações de HPO (Busca Automática) ---
    auto_tune = model_conf.get("auto_tune", False)
    num_samples = model_conf.get("num_samples", 10)  # Nº de arquiteturas a testar

    input_size = model_kwargs.get("input_size", horizon)

    # Limpeza de argumentos para os modelos estáticos
    model_kwargs_clean = model_kwargs.copy()
    keys_to_remove = ["input_size", "learning_rate", "max_steps"]

    for key in keys_to_remove:
        model_kwargs_clean.pop(key, None)  # Remove se existir, sem dar erro

    # Transforma "Hybrid_NHITS" em "NHITS" e coloca em maiúsculo
    clean_model_name = model_type.upper().replace("HYBRID_", "").replace("BASE_", "")

    logging.info(f"Inicializando {clean_model_name} (AutoTune: {auto_tune}) com Input={input_size}, Horizon={horizon}")

    model = None

    # ==========================================
    # MODO HPO (BUSCA AUTOMÁTICA)
    # ==========================================
    if auto_tune:
        if clean_model_name == "NBEATS":
            model = AutoNBEATS(h=horizon, num_samples=num_samples)
        elif clean_model_name == "NHITS":
            model = AutoNHITS(h=horizon, num_samples=num_samples)
        else:
            logging.warning(f"Auto-Tune não suportado nativamente para {clean_model_name}. Usando versão padrão.")
            auto_tune = False  # Força o fallback para o modo estático abaixo

    # ==========================================
    # MODO PADRÃO (ESTÁTICO)
    # ==========================================
    if not auto_tune:
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
            raise ValueError(f"Modelo Neural '{clean_model_name}' não implementado em deep_learning.py")

    # Configuração do NeuralForecast
    nf = NeuralForecast(models=[model], freq=freq)

    # Treinamento (No modo Auto, ele usa o val_size do seu dataset para escolher o melhor modelo)
    nf.fit(df=df_train_scaled, val_size=val_size)

    return nf
