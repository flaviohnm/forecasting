# /forecasting/generate_report.py
import time
from src.reporting import generate_final_report

# ==============================================================================
# CONFIGURAÇÃO DO RELATÓRIO
# Precisamos informar ao script quais resultados ele deve procurar
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline",
    "forecast_horizon": 10,
}

MODEL_NAMES = [
    "ARIMA",
    "NBEATS",
    "Hybrid_Direct_N-BEATS",
    "Hybrid_MIMO_N-BEATS",
    "Hybrid_Recursive_LSTM",
]
# ==============================================================================

def run():
    """Função principal para gerar o relatório final a partir dos resultados salvos."""
    start_time = time.time()
    
    generate_final_report(
        results_dir="results",
        dataset_name=DATASET_CONFIG["name"],
        forecast_horizon=DATASET_CONFIG["forecast_horizon"],
        model_names=MODEL_NAMES
    )
    
    end_time = time.time()
    print(f"\n--- Relatório gerado em {end_time - start_time:.2f} segundos ---")


if __name__ == "__main__":
    run()