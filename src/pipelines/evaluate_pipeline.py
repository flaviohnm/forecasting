import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch

try:
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    logging.info("⚠️  Patch de Segurança aplicado em EVAL: torch.load(weights_only=False)")
except Exception as e:
    logging.warning(f"Falha ao aplicar patch do PyTorch: {e}")
# ==============================================================================

from neuralforecast import NeuralForecast

from src.data_management.data_loader import load_dataset
from src.models.utils import predict_wrapper


def get_historical_residuals(base_name, df_hist, save_path, horizon):
    """
    Recupera os fitted_values do modelo base APENAS para a janela de histórico
    garantindo que o modelo híbrido receba a base correta para inferência.
    """
    model_path_joblib = os.path.join(save_path, f"{base_name}.joblib")
    model_path_neural = os.path.join(save_path, base_name)
    scaler_path = os.path.join(save_path, f"{base_name}_scaler.joblib")

    if os.path.exists(model_path_neural):
        model = NeuralForecast.load(model_path_neural)
    else:
        model = joblib.load(model_path_joblib)

    base_scaler = joblib.load(scaler_path)
    from src.models.utils import get_fitted_values

    df_fitted = get_fitted_values(model, df_hist, horizon=horizon)
    df_fitted = base_scaler.inverse_transform(df_fitted)

    if "y" in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=["y"])

    df_merged = pd.merge(df_hist, df_fitted, on=["ds", "unique_id"], how="inner")
    df_merged["y"] = df_merged["y"] - df_merged["y_hat"]
    return df_merged[["unique_id", "ds", "y"]]


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

    df_full = load_dataset(dataset_conf)
    scaler_path = os.path.join(models_path, f"{exec_name}_scaler.joblib")
    path_joblib = os.path.join(models_path, f"{exec_name}.joblib")
    path_neural = os.path.join(models_path, exec_name)

    if os.path.exists(path_neural):
        logging.info("Carregando modelo NeuralForecast (Nativo)...")
        model = NeuralForecast.load(path_neural)
    elif os.path.exists(path_joblib):
        logging.info("Carregando modelo Estatístico (Joblib)...")
        model = joblib.load(path_joblib)
    else:
        logging.error(f"Modelo {exec_name} não encontrado.")
        return

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    test_size = dataset_conf.get("test_size", 720)
    horizon = dataset_conf["forecast_horizon"]
    cutoff_index = len(df_full) - dataset_conf["forecast_horizon"]

    df_history_original = df_full.iloc[:cutoff_index].copy()
    df_test = df_full.iloc[cutoff_index:].copy()
    target_id = df_test["unique_id"].iloc[0]

    # --- LÓGICA ANTI-VAZAMENTO E RECONSTRUÇÃO HÍBRIDA ---
    if model_conf.get("depends_on"):
        base_name = f"{dataset_conf['name']}_{model_conf['depends_on']}_h{horizon}"
        logging.info("Calculando histórico de resíduos da base para inferência...")

        # O modelo híbrido precisa receber RESÍDUOS passados, não os dados originais!
        df_history = get_historical_residuals(base_name, df_history_original, models_path, horizon)
    else:
        df_history = df_history_original

    # Escala e Previsão
    df_history_scaled = scaler.transform(df_history) if scaler else df_history

    try:
        y_hat_df = predict_wrapper(model, horizon, df_history=df_history_scaled)
    except Exception as e:
        logging.error(f"Erro na previsão: {e}")
        return

    y_hat_df = y_hat_df.sort_values("ds").reset_index(drop=True)

    if scaler:
        y_hat_df = scaler.inverse_transform(y_hat_df)

    y_hat_df["unique_id"] = target_id

    # Alinhamento e Merge com Ground Truth
    df_test = df_test.sort_values("ds").reset_index(drop=True)
    if len(y_hat_df) == len(df_test):
        df_test["y_hat"] = y_hat_df["y_hat"].values
    else:
        df_test = pd.merge(df_test, y_hat_df[["ds", "y_hat"]], on="ds", how="left")

    # Reconstrução Final (Soma Base + Resíduo Previsto)
    if model_conf.get("depends_on"):
        base_forecast_path = os.path.join(forecasts_path, f"forecast_{base_name}.csv")
        if os.path.exists(base_forecast_path):
            df_base = pd.read_csv(base_forecast_path)
            df_base["ds"] = pd.to_datetime(df_base["ds"])
            df_base = df_base.sort_values("ds").reset_index(drop=True)

            if len(df_base) == len(df_test):
                df_test["y_hat_base"] = df_base["y_hat"].values
                df_test["y_hat_resid"] = df_test["y_hat"].values
                df_test["y_hat"] = df_test["y_hat_base"] + df_test["y_hat_resid"]
                logging.info("Híbrido reconstruído: Base + Resíduo")

    df_test = df_test.dropna(subset=["y", "y_hat"])
    if len(df_test) == 0:
        return

    # ==========================================
    # --- CÁLCULO DAS 4 MÉTRICAS DE AVALIAÇÃO ---
    # ==========================================

    # 1. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((df_test["y"] - df_test["y_hat"]) / df_test["y"])) * 100

    # 2. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(df_test["y"] - df_test["y_hat"]))

    # 3. MASE (Mean Absolute Scaled Error)
    y_train = df_full.iloc[:-test_size]["y"].values
    scale = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    mase = mae / scale if scale != 0 else np.inf

    # 4. sMAPE (Symmetric Mean Absolute Percentage Error)
    denom = np.abs(df_test["y"]) + np.abs(df_test["y_hat"])
    smape = np.mean((2.0 * np.abs(df_test["y"] - df_test["y_hat"])) / np.where(denom == 0, 1e-6, denom)) * 100

    logging.info(
        f"Resultados {exec_name} -> MAE: {mae:.2f} | sMAPE: {smape:.2f}% | MAPE: {mape:.2f}% | MASE: {mase:.4f}"
    )

    # Exportando as 4 métricas para o CSV
    metrics_df = pd.DataFrame(
        [
            {
                "model": exec_name,
                "dataset": dataset_conf["name"],
                "horizon": horizon,
                "mae": mae,  # Adicionado
                "smape": smape,  # Adicionado
                "mape": mape,
                "mase": mase,
            }
        ]
    )

    metrics_df.to_csv(os.path.join(metrics_path, f"metrics_{exec_name}.csv"), index=False)

    df_test[["unique_id", "ds", "y", "y_hat"]].to_csv(
        os.path.join(forecasts_path, f"forecast_{exec_name}.csv"), index=False
    )
