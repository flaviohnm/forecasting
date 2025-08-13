# /forecasting/run_experiments.py (VERSÃO REATORADA E COMPLETA)
import time
import os
from src.reporting import evaluate_forecasts, generate_final_report, create_and_save_forecast_df
from src.models import (
    train_and_forecast_arima,
    train_and_forecast_ets,
    train_and_forecast_mlp_direct,
    train_and_forecast_mlp_mimo,
    train_and_forecast_nbeats_direct,
    train_and_forecast_nbeats_mimo,
    train_and_forecast_nhits_direct,
    train_and_forecast_nhits_mimo,
    train_and_forecast_lstm_direct,
    train_and_forecast_lstm_mimo,
    train_and_forecast_transformer_mimo,
    train_and_forecast_hybrid_direct_nbeats,
    train_and_forecast_hybrid_mimo_nbeats,
    train_and_forecast_hybrid_arima_mlp,
    train_and_forecast_hybrid_recursive_lstm,
    train_and_forecast_hybrid_direct_nhits,
    train_and_forecast_hybrid_mimo_nhits
)
from src.data_processing import fetch_airline_data, process_data_fixed_origin, load_processed_data

os.environ['NIXTLA_ID_AS_COL'] = '1'

# ==============================================================================
# PAINEL DE CONTROLE DOS EXPERIMENTOS
# ==============================================================================
DATASET_CONFIG = {
    "name": "airline",
    "fetch_func": fetch_airline_data,
    "target_column": "passengers",
    "seasonality": 12,
    "forecast_horizon": 10,
    "freq": "ME"  # Frequência dos dados (ex: 'ME' para fim de mês)
}

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
    target_column = DATASET_CONFIG["target_column"]
    
    print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name} ---")

    # ETAPA 1: Obter e processar os dados brutos
    print("\n[ETAPA 1/4] Processando e dividindo os dados...")
    raw_path = DATASET_CONFIG["fetch_func"](data_dir="data")
    processed_paths = process_data_fixed_origin(
        dataset_name=dataset_name,
        raw_path=raw_path,
        processed_dir=f"data/processed/{dataset_name}",
        forecast_horizon=DATASET_CONFIG["forecast_horizon"]
    )
    print("Split de dados criado com sucesso.")
    
    # ETAPA 2: Carregar os dados processados para a memória
    print("\n[ETAPA 2/4] Carregando dados de treino/teste em memória...")
    data_dfs = load_processed_data(processed_paths)
    train_df = data_dfs["train_df"]
    test_df = data_dfs["test_df"]
    print("DataFrames carregados.")

    # ETAPA 3: Executar modelos e salvar previsões
    print("\n[ETAPA 3/4] Executando experimentos de modelagem...")
    forecast_paths = {}
    for model_name, model_config in EXPERIMENTS.items():
        print(f"\n--- Rodando Modelo: {model_name} ---")
        
        params = {
            "train_df": train_df,
            "test_df": test_df,
            "dataset_name": dataset_name,
            "seasonality": DATASET_CONFIG["seasonality"],
            "target_column": target_column,
            "freq": DATASET_CONFIG["freq"],
            **model_config.get("params", {})
        }
        
        # Executa o modelo para obter as previsões
        forecast_values = model_config["func"](**params)
        
        # Usa a função de reporting para salvar as previsões
        forecast_file = create_and_save_forecast_df(
            test_df=test_df,
            forecast_values=forecast_values,
            forecast_dir=f"results/forecasts/{dataset_name}",
            model_name=model_name,
            dataset_name=dataset_name,
            target_column=target_column
        )
        forecast_paths[model_name] = forecast_file

    # ETAPA 4: Avaliar previsões e gerar relatório final
    print("\n[ETAPA 4/4] Calculando métricas e gerando relatório...")
    for model_name, forecast_file in forecast_paths.items():
        if forecast_file:
            evaluate_forecasts(
                forecast_path=forecast_file,
                train_df=train_df,
                results_dir=f"results/metrics/{dataset_name}",
                target_column=target_column,
                model_name=model_name
            )
            
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