import logging
import os

import joblib
import pandas as pd
import torch

try:
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        # Força weights_only=False incondicionalmente
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    logging.info(
        "⚠️  Patch de Segurança aplicado em TRAIN: torch.load(weights_only=False)"
    )
except Exception as e:
    logging.warning(f"Falha ao aplicar patch do PyTorch: {e}")
# ==============================================================================

# --- Imports do Projeto (Agora é seguro importar NeuralForecast) ---
from neuralforecast import NeuralForecast

from src.data_management.data_loader import load_dataset
from src.data_management.preprocessor import TimeSeriesScaler
from src.models.deep_learning import train_dl_model
from src.models.statistical import train_stats_model


def run(model_conf, dataset_conf, main_config, exec_name):
    logging.info(f"--- [TRAIN] Iniciando Pipeline: {exec_name} ---")

    save_path = main_config["results_paths"]["saved_models"]
    os.makedirs(save_path, exist_ok=True)

    # Caminhos
    file_path = os.path.join(save_path, f"{exec_name}.joblib")  # Para estatísticos
    neural_path = os.path.join(save_path, exec_name)  # Para neurais (pasta)
    scaler_path = os.path.join(save_path, f"{exec_name}_scaler.joblib")

    # Verifica se já existe (checa arquivo OU pasta)
    if (os.path.exists(file_path) or os.path.exists(neural_path)) and os.path.exists(
        scaler_path
    ):
        logging.info(f"Modelo {exec_name} já existe. Pulando treino.")
        return

    # 2. Carregamento
    df = load_dataset(dataset_conf)

    # 3. Input Size Dinâmico
    horizon = dataset_conf["forecast_horizon"]
    if "input_size_multiplier" in model_conf:
        multiplier = model_conf["input_size_multiplier"]
        calculated_input = int(horizon * multiplier)
        if "params" not in model_conf:
            model_conf["params"] = {}
        if "model_kwargs" not in model_conf["params"]:
            model_conf["params"]["model_kwargs"] = {}
        model_conf["params"]["model_kwargs"]["input_size"] = calculated_input

    # 4. Híbrido: Resíduos
    if model_conf.get("depends_on"):
        base_model_name = (
            f"{dataset_conf['name']}_{model_conf['depends_on']}_h{horizon}"
        )
        logging.info(f"Modo Híbrido. Calculando resíduos sobre: {base_model_name}")
        try:
            df_residuals = calculate_residuals(base_model_name, df, save_path, horizon)
            df = df_residuals
        except Exception as e:
            logging.error(f"Falha ao calcular resíduos: {e}")
            raise e

    # 4.5 Scaler
    logging.info("Aplicando normalização (StandardScaler)...")
    scaler = TimeSeriesScaler()
    df_scaled = scaler.fit_transform(df)

    # 5. Treinamento
    model_type = model_conf.get("model_type", "")
    group = model_conf.get("comparison_group", "")

    try:
        if "statistical" in group or model_type in [
            "ARIMA",
            "ETS",
            "NAIVE",
            "SEASONAL_NAIVE",
        ]:
            model_object = train_stats_model(
                df_scaled, model_conf, horizon, dataset_conf["freq"]
            )

            # SALVAMENTO ESTATÍSTICO (Joblib)
            logging.info(f"Salvando modelo estatístico em: {file_path}")
            joblib.dump(model_object, file_path)

        else:
            # Neural (NHiTS, NBEATS, Informer)
            val_size = dataset_conf.get("val_size", horizon)
            model_object = train_dl_model(
                df_scaled, model_conf, horizon, dataset_conf["freq"], val_size
            )

            # SALVAMENTO NEURAL (Nativo .save)
            logging.info(f"Salvando modelo neural em: {neural_path}")
            model_object.save(neural_path, overwrite=True)

        # Salva o scaler (sempre via joblib, ele é leve)
        logging.info(f"Salvando scaler em: {scaler_path}")
        joblib.dump(scaler, scaler_path)

    except Exception as e:
        logging.error(f"Erro fatal em {exec_name}: {e}")
        raise e


def calculate_residuals(base_name, df_original, save_path, horizon):
    """
    Carrega Base (suporta Neural ou Stat), inverte escala e calcula resíduo.
    """
    # Define caminhos possíveis
    model_path_joblib = os.path.join(save_path, f"{base_name}.joblib")
    model_path_neural = os.path.join(save_path, base_name)  # Pasta
    scaler_path = os.path.join(save_path, f"{base_name}_scaler.joblib")

    # CARREGAMENTO INTELIGENTE (Stat vs Neural)
    if os.path.exists(model_path_neural):
        # É NeuralForecast (Pasta)
        model = NeuralForecast.load(model_path_neural)
    elif os.path.exists(model_path_joblib):
        # É Estatístico (Arquivo)
        model = joblib.load(model_path_joblib)
    else:
        raise FileNotFoundError(f"Modelo base {base_name} não encontrado.")

    # Carrega Scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler da base {base_name} não encontrado.")
    base_scaler = joblib.load(scaler_path)

    from src.models.utils import get_fitted_values

    # Gera fitted values (escalados)
    df_fitted = get_fitted_values(model, df_original, horizon=horizon)

    # Inverte escala para voltar a MW/Graus reais
    df_fitted = base_scaler.inverse_transform(df_fitted)

    if "y" in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=["y"])

    df_merged = pd.merge(df_original, df_fitted, on=["ds", "unique_id"], how="inner")
    df_merged["y"] = df_merged["y"] - df_merged["y_hat"]

    return df_merged[["unique_id", "ds", "y"]]
