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

from src.preparation.data_loader import load_dataset
from src.modeling.utils import predict_wrapper


def run(main_config, model_conf, dataset_conf, exec_name):
    metrics_path = main_config["results_paths"]["metrics"]
    forecasts_path = main_config["results_paths"]["forecasts"]
    save_path = main_config["results_paths"]["saved_models"]

    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(forecasts_path, exist_ok=True)

    horizon = dataset_conf.get("forecast_horizon", dataset_conf.get("horizon", 96))
    df_full = load_dataset(dataset_conf)

    # --- ANCORAGEM DE AVALIAÇÃO ESTÁTICA ---
    max_horizon = dataset_conf.get("max_horizon", 720)
    cutoff_test = len(df_full) - max_horizon

    # O conjunto de teste completo e intocável
    df_test_full = df_full.iloc[cutoff_test:].copy()

    # Como este modelo só previu 'horizon' passos, avaliamos apenas esse escopo!
    df_test = df_test_full.iloc[:horizon].copy()

    # Carrega o scaler exclusivo desta execução
    scaler_path = os.path.join(save_path, f"{exec_name}_scaler.joblib")
    scaler = joblib.load(scaler_path)

    base_dep = model_conf.get("depends_on")

    if base_dep:
        # --- RECONSTRUÇÃO HÍBRIDA ---
        base_model_name = f"{dataset_conf['name']}_{base_dep}_h{horizon}"

        # 1. Recupera Previsão Base
        model_base = joblib.load(os.path.join(save_path, f"{base_model_name}.joblib"))
        scaler_base = joblib.load(os.path.join(save_path, f"{base_model_name}_scaler.joblib"))

        pred_base_scaled = predict_wrapper(model_base, horizon)
        pred_base = scaler_base.inverse_transform(pred_base_scaled)

        # 2. Recupera Previsão dos Resíduos (Rede Neural)
        model_dl = NeuralForecast.load(os.path.join(save_path, exec_name))
        pred_dl_scaled = predict_wrapper(model_dl, horizon)
        pred_dl = scaler.inverse_transform(pred_dl_scaled)

        # 3. Combinação Linear
        df_pred = pd.merge(pred_base[["ds", "y_hat"]], pred_dl[["ds", "y_hat"]], on="ds", suffixes=("_base", "_res"))
        df_pred["y_hat"] = df_pred["y_hat_base"] + df_pred["y_hat_res"]

    else:
        # --- MODELOS PUROS ---
        model_path_joblib = os.path.join(save_path, f"{exec_name}.joblib")
        model_path_neural = os.path.join(save_path, exec_name)

        if os.path.exists(model_path_neural):
            model = NeuralForecast.load(model_path_neural)
        else:
            model = joblib.load(model_path_joblib)

        pred_scaled = predict_wrapper(model, horizon)
        df_pred = scaler.inverse_transform(pred_scaled)

    # Alinhamento final com os valores reais
    df_eval = pd.merge(df_test, df_pred[["ds", "y_hat"]], on="ds", how="inner")

    # Cálculos das Métricas
    y = df_eval["y"].values
    y_hat = df_eval["y_hat"].values

    # Correção para divisão por zero
    y_safe = np.where(y == 0, 1e-8, y)
    denom = np.abs(y) + np.abs(y_hat)
    denom_safe = np.where(denom == 0, 1e-8, denom)

    mape = np.mean(np.abs((y - y_hat) / y_safe)) * 100
    mae = np.mean(np.abs(y - y_hat))

    y_train = df_full.iloc[:cutoff_test]["y"].values
    scale = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    mase = mae / scale if scale != 0 else np.inf
    smape = np.mean((2.0 * np.abs(y - y_hat)) / denom_safe) * 100

    logging.info(f"Resultados {exec_name} -> MAE: {mae:.2f} | sMAPE: {smape:.2f}% | MAPE: {mape:.2f}%")

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": dataset_conf["name"],
                "model": exec_name,
                "horizon": horizon,
                "mae": mae,
                "mase": mase,
                "mape": mape,
                "smape": smape,
            }
        ]
    )

    metrics_df.to_csv(os.path.join(metrics_path, f"metrics_{exec_name}.csv"), index=False)
    df_eval.to_csv(os.path.join(forecasts_path, f"forecast_{exec_name}.csv"), index=False)
