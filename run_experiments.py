# /forecasting/run_experiments.py (VERSÃO FINAL E COMPLETA)
from src.reporting import evaluate_forecasts, generate_final_report
from src.models import (
    train_and_forecast_arima,
    train_and_forecast_nbeats_direct,
    train_and_forecast_nbeats_mimo,
    train_and_forecast_nhits_direct,
    train_and_forecast_nhits_mimo,
    train_and_forecast_lstm_direct,
    train_and_forecast_lstm_mimo,
    train_and_forecast_mlp_direct,
    train_and_forecast_mlp_mimo,
    train_and_forecast_ets,
    train_and_forecast_transformer_mimo,
    train_and_forecast_hybrid_direct_nbeats,
    train_and_forecast_hybrid_mimo_nbeats,
    train_and_forecast_hybrid_arima_mlp,
    train_and_forecast_hybrid_recursive_lstm,
    train_and_forecast_hybrid_direct_nhits,
    train_and_forecast_hybrid_mimo_nhits
)
from src.data_processing import fetch_airline_data, process_data_fixed_origin
import time
import os
os.environ['NIXTLA_ID_AS_COL'] = '1'

# --- MUDANÇA 1: Importar as novas funções ---

# ==============================================================================
# PAINEL DE CONTROLE DOS EXPERIMENTOS
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline", "fetch_func": fetch_airline_data,
    "target_column": "passengers", "seasonality": 12, "forecast_horizon": 10,
}

# --- MUDANÇA 2: Adicionar os novos modelos à lista ---
EXPERIMENTS = {
    # Modelos Estatísticos
    "ARIMA": {"func": train_and_forecast_arima, "params": {}},
    "ETS": {"func": train_and_forecast_ets, "params": {}},
    # Modelos DL Puros
    "MLP_Direct": {"func": train_and_forecast_mlp_direct, "params": {"max_steps": 100}},
    "MLP_MIMO": {"func": train_and_forecast_mlp_mimo, "params": {"max_steps": 100}},
    "NBEATS_Direct": {"func": train_and_forecast_nbeats_direct, "params": {"max_steps": 100}},
    "NBEATS_MIMO": {"func": train_and_forecast_nbeats_mimo, "params": {"max_steps": 100}},
    "NHiTS_Direct": {"func": train_and_forecast_nhits_direct, "params": {"max_steps": 100}},
    "NHiTS_MIMO": {"func": train_and_forecast_nhits_mimo, "params": {"max_steps": 100}},
    "Transformer_MIMO": {"func": train_and_forecast_transformer_mimo, "params": {"max_steps": 100}},
    "NHITS_MIMO": {"func": train_and_forecast_nhits_mimo, "params": {"max_steps": 100}},
    "LSTM_Direct": {"func": train_and_forecast_lstm_direct, "params": {"max_steps": 100}},
    "LSTM_MIMO": {"func": train_and_forecast_lstm_mimo, "params": {"max_steps": 100}},
    "Transformer_MIMO": {"func": train_and_forecast_transformer_mimo, "params": {"max_steps": 100}},
    # Modelos Híbridos
    "Hybrid_Direct_N-BEATS": {"func": train_and_forecast_hybrid_direct_nbeats, "params": {"max_steps": 100}},
    "Hybrid_MIMO_N-BEATS": {"func": train_and_forecast_hybrid_mimo_nbeats, "params": {"max_steps": 100}},
    "Hybrid_ARIMA_MLP": {"func": train_and_forecast_hybrid_arima_mlp, "params": {"max_steps": 100}},
    "Hybrid_Recursive_LSTM": {"func": train_and_forecast_hybrid_recursive_lstm, "params": {"max_steps": 100}},
    "Hybrid_Direct_N-HiTS": {"func": train_and_forecast_hybrid_direct_nhits, "params": {"max_steps": 100}},
    "Hybrid_MIMO_N-HiTS": {"func": train_and_forecast_hybrid_mimo_nhits, "params": {"max_steps": 100}},
}
# ==============================================================================


def run():
    start_time = time.time()
    dataset_name = DATASET_CONFIG["name"]
    print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name} ---")
    print("\n[ETAPA 1/3] Criando split de treino/teste de origem fixa...")
    raw_path = DATASET_CONFIG["fetch_func"](data_dir="data")
    processed_paths = process_data_fixed_origin(dataset_name=dataset_name, raw_path=raw_path,
                                                processed_dir=f"data/processed/{dataset_name}", forecast_horizon=DATASET_CONFIG["forecast_horizon"])
    print("Split de dados criado com sucesso.")
    print("\n[ETAPA 2/3] Executando experimentos de modelagem...")
    forecast_paths = {}
    for model_name, model_config in EXPERIMENTS.items():
        print(f"\n--- Rodando Modelo: {model_name} ---")
        params = {
            "seasonality": DATASET_CONFIG["seasonality"], **model_config.get("params", {})}
        forecast_file = model_config["func"](train_path=processed_paths["train_path"], test_path=processed_paths["test_path"],
                                             model_dir=f"results/models/{dataset_name}", forecast_dir=f"results/forecasts/{dataset_name}", dataset_name=dataset_name, **params)
        forecast_paths[model_name] = forecast_file
    print("\n[ETAPA 3/3] Calculando métricas de performance...")
    for model_name, forecast_file in forecast_paths.items():
        if forecast_file:
            evaluate_forecasts(forecast_path=forecast_file, train_path=processed_paths["train_path"],
                               results_dir=f"results/metrics/{dataset_name}", target_column=DATASET_CONFIG["target_column"], model_name=model_name)
    print("\nGerando relatório final...")
    generate_final_report(results_dir="results", dataset_name=dataset_name,
                          forecast_horizon=DATASET_CONFIG["forecast_horizon"], model_names=list(EXPERIMENTS.keys()))
    end_time = time.time()
    print(
        f"\n--- PIPELINE COMPLETA EM {end_time - start_time:.2f} SEGUNDOS ---")


if __name__ == "__main__":
    run()
