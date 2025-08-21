# /forecasting/run_experiments.py (VERSÃO FINAL COMPLETA)
import time
import os
import copy
import json
from torch.optim import Adam
from src.reporting import evaluate_forecasts, generate_final_report, create_and_save_forecast_df
from src.models import (
    train_and_forecast_arima, train_and_forecast_ets, train_and_forecast_naive,
    train_and_forecast_direct_mlp, train_and_forecast_mimo_mlp,
    train_and_forecast_nbeats_direct, train_and_forecast_nbeats_mimo,
    train_and_forecast_nhits_direct, train_and_forecast_nhits_mimo,
    train_and_forecast_lstm_direct, train_and_forecast_lstm_mimo,
    train_and_forecast_transformer_mimo,
    train_and_forecast_hybrid_direct_nbeats,
    train_and_forecast_hybrid_mimo_nbeats,
    train_and_forecast_hybrid_direct_mlp, train_and_forecast_hybrid_mimo_mlp,
    train_and_forecast_hybrid_recursive_lstm,
    train_and_forecast_hybrid_direct_nhits,
    train_and_forecast_hybrid_mimo_nhits)
from src.data_processing import fetch_airline_data, fetch_daily_births_data, process_data_fixed_origin, load_processed_data

os.environ['NIXTLA_ID_AS_COL'] = '1'

# ==============================================================================
# PAINEL DE CONTROLE DOS DATASETS
# ==============================================================================
ALL_DATASET_CONFIGS = [
    {
        "name": "airline",
        "fetch_func": fetch_airline_data,
        "target_column": "passengers",
        "seasonality": 12,
        "forecast_horizon": 10,
        "freq": "MS"
    },
    {
        "name": "daily_births",
        "fetch_func": fetch_daily_births_data,
        "target_column": "value",
        "seasonality": 7,
        "forecast_horizon": 10,
        "freq": "D"
    }
]

# ==============================================================================
# CARREGAMENTO DINÂMICO DOS EXPERIMENTOS
# ==============================================================================

def load_experiments_from_config(config_path='models_config.json'):
    """
    Carrega a configuração dos modelos a partir de um arquivo JSON,
    montando o dicionário de experimentos dinamicamente.
    """
    MODEL_FUNCTIONS = {
        "Naive": train_and_forecast_naive, "ARIMA": train_and_forecast_arima, "ETS": train_and_forecast_ets,
        "NBEATS_Direct": train_and_forecast_nbeats_direct, "NBEATS_MIMO": train_and_forecast_nbeats_mimo,
        "NHiTS_Direct": train_and_forecast_nhits_direct, "NHiTS_MIMO": train_and_forecast_nhits_mimo,
        "LSTM_Direct": train_and_forecast_lstm_direct, "LSTM_MIMO": train_and_forecast_lstm_mimo,
        "Transformer_MIMO": train_and_forecast_transformer_mimo,
        "Hybrid_Direct_N-BEATS": train_and_forecast_hybrid_direct_nbeats,
        "Hybrid_MIMO_N-BEATS": train_and_forecast_hybrid_mimo_nbeats,
        "Hybrid_Recursive_LSTM": train_and_forecast_hybrid_recursive_lstm,
        "Hybrid_Direct_N-HiTS": train_and_forecast_hybrid_direct_nhits,
        "Hybrid_MIMO_N-HiTS": train_and_forecast_hybrid_mimo_nhits
    }

    with open(config_path, 'r') as f:
        config = json.load(f)

    experiments = {}

    # Processa modelos estáticos
    for model_conf in config['models']:
        if model_conf.get('enabled', False):
            model_name = model_conf['name']
            params = model_conf.get('params', {})
            
            if any(k in model_name for k in ["NBEATS", "NHiTS", "LSTM", "Transformer", "Hybrid"]):
                params["optimizer"] = Adam
            
            if 'mlp_units' in params and isinstance(params['mlp_units'], str):
                params['mlp_units'] = json.loads(params['mlp_units'])

            experiments[model_name] = {
                "func": MODEL_FUNCTIONS[model_name],
                "params": params
            }

    # Processa modelos MLP dinâmicos
    if config.get('dynamic_mlp_models', {}).get('enabled', False):
        mlp_config = config['dynamic_mlp_models']
        mlp_base_params = mlp_config.get('base_params', {})
        mlp_base_params["optimizer"] = Adam
        
        dynamic_mlp_functions = {
            "MLP_MIMO": train_and_forecast_mimo_mlp,
            "MLP_Direct": train_and_forecast_direct_mlp,
            "Hybrid_MLP_MIMO": train_and_forecast_hybrid_mimo_mlp,
            "Hybrid_MLP_Direct": train_and_forecast_hybrid_direct_mlp,
        }

        for model_key, model_func in dynamic_mlp_functions.items():
            for activation_func in mlp_config.get('activations', []):
                model_name = f"{model_key}_{activation_func.capitalize()}"
                current_params = copy.deepcopy(mlp_base_params)
                current_params["activation"] = activation_func
                experiments[model_name] = {"func": model_func, "params": current_params}

    print(f"--- {len(experiments)} experimentos carregados do arquivo de configuração ---")
    if not experiments:
        print("AVISO: Nenhum experimento foi habilitado no arquivo 'models_config.json'. A execução será encerrada.")
    return experiments


# ==============================================================================

def run():
    """
    Função principal que orquestra todo o pipeline de experimentação.
    """
    print("Iniciando a execucao dos experimentos...")
    start_time = time.time()
    
    EXPERIMENTS = load_experiments_from_config()
    
    if not EXPERIMENTS:
        return

    all_dataset_results = []

    for dataset_config in ALL_DATASET_CONFIGS:
        dataset_name = dataset_config["name"]
        target_column = dataset_config["target_column"]
        
        print(f"\n{'='*60}")
        print(f"--- INICIANDO PIPELINE PARA O DATASET: {dataset_name.upper()} ---")
        print(f"{'='*60}\n")

        print("\n[ETAPA 1/4] Processando e dividindo os dados...")
        raw_path = dataset_config["fetch_func"](data_dir="data")
        processed_paths = process_data_fixed_origin(
            dataset_name=dataset_name, raw_path=raw_path,
            processed_dir=f"data/processed/{dataset_name}",
            forecast_horizon=dataset_config["forecast_horizon"])
        print("Split de dados criado com sucesso.")

        print("\n[ETAPA 2/4] Carregando dados de treino/teste em memória...")
        data_dfs = load_processed_data(processed_paths)
        train_df, test_df = data_dfs["train_df"], data_dfs["test_df"]
        print("DataFrames carregados.")

        print("\n[ETAPA 3/4] Executando experimentos de modelagem...")
        forecast_paths = {}
        for model_name, model_config in EXPERIMENTS.items():
            print(f"\n--- Rodando Modelo: {model_name} ---")
            params = {
                "train_df": train_df, "test_df": test_df,
                "dataset_name": dataset_name, "target_column": target_column,
                "seasonality": dataset_config["seasonality"],
                "freq": dataset_config["freq"], **model_config.get("params", {})}
            
            try:
                forecast_values = model_config["func"](**params)
                forecast_file = create_and_save_forecast_df(
                    test_df=test_df, forecast_values=forecast_values,
                    forecast_dir=f"results/forecasts/{dataset_name}",
                    model_name=model_name, dataset_name=dataset_name,
                    target_column=target_column)
                forecast_paths[model_name] = forecast_file
            except Exception as e:
                print(f"!!!!!! ERRO AO EXECUTAR O MODELO {model_name}: {e} !!!!!!")
                forecast_paths[model_name] = None
        
        print(f"\n[ETAPA 4/4] Calculando métricas para {dataset_name}...")
        for model_name, forecast_file in forecast_paths.items():
            if forecast_file:
                evaluate_forecasts(
                    forecast_path=forecast_file, train_df=train_df,
                    results_dir=f"results/metrics/{dataset_name}",
                    target_column=target_column, model_name=model_name)
        
        all_dataset_results.append({
            "dataset_name": dataset_name,
            "forecast_horizon": dataset_config["forecast_horizon"],
            "model_names": list(EXPERIMENTS.keys())
        })

    print("\n\n[ETAPA FINAL] Gerando relatório consolidado para todos os datasets...")
    generate_final_report(
        results_dir="results",
        all_dataset_results=all_dataset_results
    )

    end_time = time.time()
    print(f"\n--- PIPELINE COMPLETA EM {end_time - start_time:.2f} SEGUNDOS ---")

if __name__ == "__main__":
    run()