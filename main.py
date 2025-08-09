# /forecasting/main.py (VERSÃO COM PASTAS DE RESULTADOS CENTRALIZADAS)
import time
from src.data_processing import fetch_airline_data, process_data_fixed_origin
from src.models import train_and_forecast_arima, train_and_forecast_nbeats_direct, train_and_forecast_hybrid_recursive_direct

# ==============================================================================
# PAINEL DE CONTROLE DOS EXPERIMENTOS
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline",
    "fetch_func": fetch_airline_data,
    "target_column": "passengers",
    "freq": "M",
    "seasonality": 12,
    "forecast_horizon": 10,
}

EXPERIMENTS = {
    "ARIMA": {
        "func": train_and_forecast_arima,
        "params": {}
    },
    "NBEATS_direct": {
        "func": train_and_forecast_nbeats_direct,
        "params": {
            "max_steps": 100
        }
    },
    "Hybrid_RecursiveDirect": {
        "func": train_and_forecast_hybrid_recursive_direct,
        "params": {
            "max_steps": 100
        }
    },
}
# ==============================================================================

def run():
    """Função principal para orquestrar a pipeline de experimentos."""
    start_time = time.time()
    dataset_name = DATASET_CONFIG["name"]
    print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name} ---")

    # 1. Etapa de Dados
    print("\n[ETAPA 1/3] Criando split de treino/teste de origem fixa...")
    raw_path = DATASET_CONFIG["fetch_func"](data_dir="data")
    processed_paths = process_data_fixed_origin(
        dataset_name=dataset_name,
        raw_path=raw_path,
        processed_dir=f"data/processed/{dataset_name}",
        forecast_horizon=DATASET_CONFIG["forecast_horizon"]
    )
    print("Split de dados criado com sucesso.")

    # 2. Etapa de Modelagem
    print("\n[ETAPA 2/3] Executando experimentos de modelagem...")
    for model_name, model_config in EXPERIMENTS.items():
        print(f"\n--- Rodando Modelo: {model_name} ---")
        model_start_time = time.time()
        
        # --- MUDANÇA AQUI: Centralizando todos os caminhos de saída ---
        params = {
            "seasonality": DATASET_CONFIG["seasonality"],
            **model_config["params"]
        }

        model_config["func"](
            train_path=processed_paths["train_path"],
            test_path=processed_paths["test_path"],
            # Todos os modelos salvos irão para 'results/models'
            model_dir=f"results/models/{dataset_name}",
            # Todas as previsões irão para 'results/forecasts'
            forecast_dir=f"results/forecasts/{dataset_name}",
            dataset_name=dataset_name,
            **params
        )
        
        model_end_time = time.time()
        print(f"--- Modelo {model_name} concluído em {model_end_time - model_start_time:.2f} segundos ---")
    
    # 3. Etapa de Avaliação
    print("\n[ETAPA 3/3] Avaliação dos resultados...")
    print("Etapa de avaliação a ser implementada.")
    
    end_time = time.time()
    print(f"\n--- PIPELINE COMPLETA EM {end_time - start_time:.2f} SEGUNDOS ---")

if __name__ == "__main__":
    run()