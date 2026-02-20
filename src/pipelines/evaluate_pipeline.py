import logging
import os

import joblib
import numpy as np
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
        "⚠️  Patch de Segurança aplicado em EVAL: torch.load(weights_only=False)"
    )
except Exception as e:
    logging.warning(f"Falha ao aplicar patch do PyTorch: {e}")
# ==============================================================================

# --- Imports do Projeto ---
from neuralforecast import NeuralForecast

from src.analysis.diagnostics import save_residual_diagnostics
from src.data_management.data_loader import load_dataset
from src.data_management.preprocessor import TimeSeriesScaler
from src.models.utils import predict_wrapper


def run(main_config, model_conf, dataset_conf, exec_name):
    logging.info(f"--- [EVAL] Iniciando Avaliação: {exec_name} ---")

    models_path = main_config["results_paths"]["saved_models"]
    metrics_path = main_config["results_paths"]["metrics"]
    forecasts_path = main_config["results_paths"]["forecasts"]
    diag_path = main_config["results_paths"].get(
        "diagnostics", os.path.join(os.path.dirname(metrics_path), "diagnostics")
    )

    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(forecasts_path, exist_ok=True)
    os.makedirs(diag_path, exist_ok=True)

    # 1. Dados
    df_full = load_dataset(dataset_conf)

    # 2. Carregar Modelo e Scaler
    scaler_path = os.path.join(models_path, f"{exec_name}_scaler.joblib")

    # Caminhos possíveis do modelo
    path_joblib = os.path.join(models_path, f"{exec_name}.joblib")
    path_neural = os.path.join(models_path, exec_name)

    # LÓGICA DE CARREGAMENTO MISTO
    if os.path.exists(path_neural):
        logging.info("Carregando modelo NeuralForecast (Nativo)...")
        model = NeuralForecast.load(path_neural)
    elif os.path.exists(path_joblib):
        logging.info("Carregando modelo Estatístico (Joblib)...")
        model = joblib.load(path_joblib)
    else:
        logging.error(f"Modelo {exec_name} não encontrado (nem .joblib nem pasta).")
        return

    # Carrega Scaler
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        logging.warning("Scaler não encontrado! Avaliação pode falhar.")

    # 3. Setup de Teste
    test_size = dataset_conf.get("test_size", 720)
    horizon = dataset_conf["forecast_horizon"]
    cutoff_index = len(df_full) - dataset_conf["forecast_horizon"]

    df_history = df_full.iloc[:cutoff_index].copy()
    df_test = df_full.iloc[cutoff_index:].copy()

    # 4. Escala e Previsão
    if scaler:
        df_history_scaled = scaler.transform(df_history)
    else:
        df_history_scaled = df_history

    try:
        # O modelo prevê (output escalado)
        y_hat_df = predict_wrapper(model, horizon, df_history=df_history_scaled)
    except Exception as e:
        logging.error(f"Erro na previsão: {e}")
        return

    # 5. Pós-Processamento
    y_hat_df = y_hat_df.sort_values("ds").reset_index(drop=True)

    if scaler:
        # Inverte escala (volta para MW/Graus)
        y_hat_df = scaler.inverse_transform(y_hat_df)

    # Alinha datas e IDs
    target_id = df_test["unique_id"].iloc[0]
    y_hat_df["unique_id"] = target_id

    # Merge seguro
    df_test = df_test.sort_values("ds").reset_index(drop=True)
    if len(y_hat_df) == len(df_test):
        df_test["y_hat"] = y_hat_df["y_hat"].values
    else:
        df_test = pd.merge(df_test, y_hat_df[["ds", "y_hat"]], on="ds", how="left")

    # 6. Lógica Híbrida (Reconstrução)
    if model_conf.get("depends_on"):
        base_name = f"{dataset_conf['name']}_{model_conf['depends_on']}_h{horizon}"
        base_forecast_path = os.path.join(forecasts_path, f"forecast_{base_name}.csv")

        if os.path.exists(base_forecast_path):
            df_base = pd.read_csv(base_forecast_path)
            # Garante que base também está ordenada
            df_base["ds"] = pd.to_datetime(df_base["ds"])
            df_base = df_base.sort_values("ds").reset_index(drop=True)

            if len(df_base) == len(df_test):
                df_test["y_hat_base"] = df_base["y_hat"].values
                df_test["y_hat_resid"] = df_test["y_hat"].values
                df_test["y_hat"] = df_test["y_hat_base"] + df_test["y_hat_resid"]
                logging.info(f"Híbrido reconstruído: Base + Resíduo")

    # 7. Métricas e Salvamento
    df_test = df_test.dropna(subset=["y", "y_hat"])
    if len(df_test) == 0:
        return

    mape = np.mean(np.abs((df_test["y"] - df_test["y_hat"]) / df_test["y"])) * 100

    # Cálculo MASE simples
    y_train = df_full.iloc[:-test_size]["y"].values
    scale = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    mae = np.mean(np.abs(df_test["y"] - df_test["y_hat"]))
    mase = mae / scale if scale != 0 else np.inf

    logging.info(f"Resultados {exec_name} -> MAPE: {mape:.4f} | MASE: {mase:.4f}")

    metrics_df = pd.DataFrame(
        [
            {
                "model": exec_name,
                "dataset": dataset_conf["name"],
                "horizon": horizon,
                "mape": mape,
                "mase": mase,
            }
        ]
    )
    metrics_df.to_csv(
        os.path.join(metrics_path, f"metrics_{exec_name}.csv"), index=False
    )
    df_test[["unique_id", "ds", "y", "y_hat"]].to_csv(
        os.path.join(forecasts_path, f"forecast_{exec_name}.csv"), index=False
    )
