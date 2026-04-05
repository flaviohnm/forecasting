import logging
import os
import shutil

import joblib
import pandas as pd
import torch

try:
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    logging.info("⚠️  Patch de Segurança aplicado em TRAIN: torch.load(weights_only=False)")
except Exception as e:
    logging.warning(f"Falha ao aplicar patch do PyTorch: {e}")
# ==============================================================================

from neuralforecast import NeuralForecast

from src.preparation.data_loader import load_dataset
from src.preparation.preprocessor import TimeSeriesScaler
from src.modeling.deep_learning import train_dl_model
from src.modeling.statistical import train_stats_model
from src.modeling.utils import get_fitted_values


def calculate_residuals(base_name, df_original, save_path, horizon):
    """Calcula resíduos e injeta de volta na coluna 'y' para a rede neural."""
    model_path_joblib = os.path.join(save_path, f"{base_name}.joblib")

    if os.path.exists(model_path_joblib):
        model = joblib.load(model_path_joblib)
    else:
        raise FileNotFoundError(f"Modelo base {base_name} não encontrado.")

    # Recupera in-sample predictions
    df_fitted = get_fitted_values(model, df_original, horizon)

    # Merge de segurança para alinhar as datas
    df_merged = pd.merge(df_original, df_fitted[["unique_id", "ds", "y_hat"]], on=["unique_id", "ds"], how="inner")

    # A MÁGICA: y agora é o resíduo (Real - Previsto)
    df_merged["y"] = df_merged["y"] - df_merged["y_hat"]

    return df_merged[["unique_id", "ds", "y"]]


def run(main_config, model_conf, dataset_conf, exec_name):
    save_path = main_config["results_paths"]["saved_models"]
    os.makedirs(save_path, exist_ok=True)

    file_path_joblib = os.path.join(save_path, f"{exec_name}.joblib")
    neural_path = os.path.join(save_path, exec_name)
    scaler_path = os.path.join(save_path, f"{exec_name}_scaler.joblib")

    horizon = dataset_conf.get("forecast_horizon", dataset_conf.get("horizon", 96))
    freq = dataset_conf.get("freq", "H")

    # 1. Carrega o dado univariado limpo
    df_full = load_dataset(dataset_conf)

    # --- ANCORAGEM ACADÊMICA ESTÁTICA ---
    max_horizon = dataset_conf.get("max_horizon", 720)
    val_size = dataset_conf.get("val_size", max_horizon)

    cutoff_test = len(df_full) - max_horizon
    cutoff_val = cutoff_test - val_size

    df_train = df_full.iloc[:cutoff_val].copy()
    df_val = df_full.iloc[cutoff_val:cutoff_test].copy()
    df_test = df_full.iloc[cutoff_test:].copy()

    df_train_val = pd.concat([df_train, df_val]).reset_index(drop=True)

    logging.info(f"Split Estático -> Treino: {len(df_train)} | Val: {len(df_val)} | Teste Oculto: {len(df_test)}")

    # 3. Transformação Híbrida (Extração de Resíduos)
    base_dep = model_conf.get("depends_on")
    if base_dep:
        base_model_name = f"{dataset_conf['name']}_{base_dep}_h{horizon}"
        logging.info(f"Híbrido detectado. Extraindo resíduos do histórico de: {base_model_name}")
        df_train_val = calculate_residuals(base_model_name, df_train_val, save_path, horizon)

    # 4. Scaler independente (aprende a escala dos dados ou dos resíduos)
    logging.info("Aplicando TimeSeriesScaler no conjunto de Treino/Validação...")
    scaler = TimeSeriesScaler()
    df_scaled = scaler.fit_transform(df_train_val)
    joblib.dump(scaler, scaler_path)

    model_type = model_conf.get("model_type", "")
    group = model_conf.get("comparison_group", "")

    # 5. Treinamento
    if "statistical" in group or model_type in ["ARIMA", "ETS", "NAIVE", "SEASONAL_NAIVE"]:
        model_object = train_stats_model(df_scaled, model_conf, horizon, freq)
        joblib.dump(model_object, file_path_joblib)
    else:
        model_object = train_dl_model(df_scaled, model_conf, horizon, freq, val_size)

        if os.path.exists(neural_path):
            shutil.rmtree(neural_path)
        model_object.save(neural_path, overwrite=True)

    logging.info(f"Treinamento finalizado de forma isolada e segura: {exec_name}")
