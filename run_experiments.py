# /forecasting/run_experiments.py (VERSÃO COM TODOS OS MODELOS)
import time
from src.data_processing import fetch_airline_data, process_data_fixed_origin
# --- MUDANÇA 1: Importar a função do Híbrido LSTM ---
from src.models import (
    train_and_forecast_arima,
    train_and_forecast_nbeats_direct,
    train_and_forecast_nbeats_mimo,
    train_and_forecast_lstm_direct,
    train_and_forecast_lstm_mimo,
    train_and_forecast_hybrid_direct_nbeats,
    train_and_forecast_hybrid_mimo_nbeats,
    train_and_forecast_hybrid_recursive_lstm # <-- Reintroduzido
)
from src.reporting import evaluate_forecasts, generate_final_report

# ==============================================================================
# PAINEL DE CONTROLE DOS EXPERIMENTOS
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline", "fetch_func": fetch_airline_data,
    "target_column": "passengers", "seasonality": 12, "forecast_horizon": 10,
}

# --- MUDANÇA 2: Adicionar o Híbrido LSTM de volta aos experimentos ---
EXPERIMENTS = {
    "ARIMA": {"func": train_and_forecast_arima, "params": {}},
    "NBEATS_Direct": {"func": train_and_forecast_nbeats_direct, "params": {"max_steps": 100}},
    "NBEATS_MIMO": {"func": train_and_forecast_nbeats_mimo, "params": {"max_steps": 100}},
    "LSTM_Direct": {"func": train_and_forecast_lstm_direct, "params": {"max_steps": 100}},
    "LSTM_MIMO": {"func": train_and_forecast_lstm_mimo, "params": {"max_steps": 100}},
    "Hybrid_Direct_N-BEATS": {"func": train_and_forecast_hybrid_direct_nbeats, "params": {"max_steps": 100}},
    "Hybrid_MIMO_N-BEATS": {"func": train_and_forecast_hybrid_mimo_nbeats, "params": {"max_steps": 100}},
    "Hybrid_Recursive_LSTM": {"func": train_and_forecast_hybrid_recursive_lstm, "params": {"max_steps": 100}},
}
# ==============================================================================

def run():
    # (O corpo da função run não precisa de alterações)
    start_time = time.time()
    dataset_name = DATASET_CONFIG["name"]
    print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name} ---")
    print("\n[ETAPA 1/4] Criando split de treino/teste de origem fixa...")
    raw_path = DATASET_CONFIG["fetch_func"](data_dir="data")
    processed_paths = process_data_fixed_origin(dataset_name=dataset_name, raw_path=raw_path, processed_dir=f"data/processed/{dataset_name}", forecast_horizon=DATASET_CONFIG["forecast_horizon"])
    print("Split de dados criado com sucesso.")
    print("\n[ETAPA 2/4] Executando experimentos de modelagem...")
    forecast_paths = {}
    for model_name, model_config in EXPERIMENTS.items():
        print(f"\n--- Rodando Modelo: {model_name} ---")
        params = {"seasonality": DATASET_CONFIG["seasonality"], **model_config.get("params", {})}
        forecast_file = model_config["func"](train_path=processed_paths["train_path"], test_path=processed_paths["test_path"], model_dir=f"results/models/{dataset_name}", forecast_dir=f"results/forecasts/{dataset_name}", dataset_name=dataset_name, **params)
        forecast_paths[model_name] = forecast_file
    print("\n[ETAPA 3/4] Calculando métricas de performance...")
    for model_name, forecast_file in forecast_paths.items():
        if forecast_file:
            evaluate_forecasts(forecast_path=forecast_file, train_path=processed_paths["train_path"], results_dir=f"results/metrics/{dataset_name}", target_column=DATASET_CONFIG["target_column"], model_name=model_name)
    print("\n[ETAPA 4/4] Gerando relatório final...")
    generate_final_report(results_dir="results", dataset_name=dataset_name, forecast_horizon=DATASET_CONFIG["forecast_horizon"], model_names=list(EXPERIMENTS.keys()))
    end_time = time.time()
    print(f"\n--- PIPELINE COMPLETA EM {end_time - start_time:.2f} SEGUNDOS ---")

if __name__ == "__main__":
    run()