# /forecasting/main.py (ATUALIZADO COM CHAMADA FINAL DE RELATÓRIO)
import time
from src.data_processing import fetch_airline_data, process_data_fixed_origin
from src.models import (
    train_and_forecast_arima,
    train_and_forecast_nbeats_direct,
    train_and_forecast_hybrid_recursive_direct
)
from src.reporting import evaluate_forecasts, generate_final_report

# ==============================================================================
# PAINEL DE CONTROLE DOS EXPERIMENTOS
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline",
    "fetch_func": fetch_airline_data,
    "target_column": "passengers",
    "seasonality": 12,
    "forecast_horizon": 10,
}

EXPERIMENTS = {
    "ARIMA": {"func": train_and_forecast_arima, "params": {}},
    "NBEATS_direct": {"func": train_and_forecast_nbeats_direct, "params": {"max_steps": 100}},
    "Hybrid_RecursiveDirect": {"func": train_and_forecast_hybrid_recursive_direct, "params": {"max_steps": 100}},
}
# ==============================================================================

def run():
    """Função principal para orquestrar a pipeline de experimentos."""
    start_time = time.time()
    dataset_name = DATASET_CONFIG["name"]
    print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name} ---")

    # 1. Etapa de Dados
    print("\n[ETAPA 1/4] Criando split de treino/teste de origem fixa...")
    raw_path = DATASET_CONFIG["fetch_func"](data_dir="data")
    processed_paths = process_data_fixed_origin(
        dataset_name=dataset_name,
        raw_path=raw_path,
        processed_dir=f"data/processed/{dataset_name}",
        forecast_horizon=DATASET_CONFIG["forecast_horizon"]
    )
    print("Split de dados criado com sucesso.")

    # 2. Etapa de Modelagem
    print("\n[ETAPA 2/4] Executando experimentos de modelagem...")
    forecast_paths = {}
    for model_name, model_config in EXPERIMENTS.items():
        print(f"\n--- Rodando Modelo: {model_name} ---")
        params = {"seasonality": DATASET_CONFIG["seasonality"], **model_config.get("params", {})}
        forecast_file = model_config["func"](
            train_path=processed_paths["train_path"],
            test_path=processed_paths["test_path"],
            model_dir=f"results/models/{dataset_name}",
            forecast_dir=f"results/forecasts/{dataset_name}",
            dataset_name=dataset_name,
            **params
        )
        forecast_paths[model_name] = forecast_file
    
    # 3. Etapa de Avaliação
    print("\n[ETAPA 3/4] Calculando métricas de performance...")
    for model_name, forecast_file in forecast_paths.items():
        evaluate_forecasts(
            forecast_path=forecast_file,
            train_path=processed_paths["train_path"],
            results_dir=f"results/metrics/{dataset_name}",
            target_column=DATASET_CONFIG["target_column"]
        )

    # --- MUDANÇA AQUI: Chamando a função de relatório com os parâmetros corretos ---
    print("\n[ETAPA 4/4] Gerando relatório final...")
    generate_final_report(
        results_dir="results",
        dataset_name=dataset_name,
        forecast_horizon=DATASET_CONFIG["forecast_horizon"],
        model_names=list(EXPERIMENTS.keys())
    )
    
    end_time = time.time()
    print(f"\n--- PIPELINE COMPLETA EM {end_time - start_time:.2f} SEGUNDOS ---")

if __name__ == "__main__":
    run()