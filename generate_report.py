# /forecasting/generate_report.py (VERSÃO AJUSTADA)
import time
from src.reporting import generate_final_report

# ==============================================================================
# CONFIGURAÇÃO DO RELATÓRIO
# Informe para quais datasets e modelos você quer gerar o relatório.
# Você pode adicionar mais dicionários a esta lista para gerar relatórios
# para múltiplos datasets de uma vez.
# ==============================================================================
ALL_DATASET_RESULTS = [
    {
        "dataset_name": "airline",
        "forecast_horizon": 10,
        "model_names": [
            "Naive",
            "ARIMA",
            "ETS",
            "MLP_Direct_Relu",
            "MLP_MIMO_Relu",
            "NBEATS_Direct",
            "NBEATS_MIMO",
            "NHiTS_Direct",
            "NHiTS_MIMO",
            "LSTM_Direct",
            "LSTM_MIMO",
            "Transformer_MIMO",
            "Hybrid_Direct_N-BEATS",
            "Hybrid_MIMO_N-BEATS",
            "Hybrid_Recursive_LSTM",
            "Hybrid_Direct_N-HiTS",
            "Hybrid_MIMO_N-HiTS",
            "Hybrid_MLP_Direct_Relu",
            "Hybrid_MLP_MIMO_Relu"
        ]
    },
    # Exemplo: Se você também rodou os experimentos para o dataset 'daily_births'
    {
        "dataset_name": "daily_births",
        "forecast_horizon": 10,
        "model_names": [
            "Naive",
            "ARIMA",
            "ETS",
            "MLP_Direct_Relu",
            "MLP_MIMO_Relu",
            "NBEATS_Direct",
            "NBEATS_MIMO",
            "NHiTS_Direct",
            "NHiTS_MIMO",
            "LSTM_Direct",
            "LSTM_MIMO",
            "Transformer_MIMO",
            "Hybrid_Direct_N-BEATS",
            "Hybrid_MIMO_N-BEATS",
            "Hybrid_Recursive_LSTM",
            "Hybrid_Direct_N-HiTS",
            "Hybrid_MIMO_N-HiTS",
            "Hybrid_MLP_Direct_Relu",
            "Hybrid_MLP_MIMO_Relu"
        ]
    }
]
# ==============================================================================

def run():
    """Função principal para gerar o relatório final a partir dos resultados salvos."""
    print("Iniciando a geração do relatório a partir dos resultados existentes...")
    start_time = time.time()
    
    if not ALL_DATASET_RESULTS:
        print("AVISO: Nenhuma configuração de dataset foi encontrada. O relatório não será gerado.")
        return
        
    # A função agora recebe a lista de configurações, como esperado.
    generate_final_report(
        results_dir="results",
        all_dataset_results=ALL_DATASET_RESULTS
    )
    
    end_time = time.time()
    print(f"\n--- Relatório gerado em {end_time - start_time:.2f} segundos ---")


if __name__ == "__main__":
    run()