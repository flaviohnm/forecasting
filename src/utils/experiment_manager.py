import os
import logging
from src.utils.config_loader import load_and_organize_strategies

def run_custom_pipeline(main_config, yaml_model_path, target_datasets=None, force_rerun=False):
    """
    Orquestrador com DEBUG para encontrar os arquivos .joblib
    """
    # 1. Carrega estratégias
    base_queue, hybrid_queue = load_and_organize_strategies(yaml_model_path)
    
    # 2. Configura Datasets
    raw_datasets = main_config.get('datasets', {})
    datasets_dict = raw_datasets if isinstance(raw_datasets, dict) else {ds: {} for ds in raw_datasets}
    if target_datasets:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in target_datasets}

    metrics_path = main_config['results_paths']['metrics']
    saved_models_path = main_config['results_paths']['saved_models']
    
    # --- DEBUG: O QUE O PYTHON ESTÁ VENDO? ---
    print(f"\n--- [DEBUG DIAGNÓSTICO] ---")
    print(f"Pasta de modelos configurada: {saved_models_path}")
    if os.path.exists(saved_models_path):
        arquivos = os.listdir(saved_models_path)
        print(f"Total de arquivos na pasta: {len(arquivos)}")
        print(f"Exemplo dos primeiros 5 arquivos: {arquivos[:5]}")
        if not arquivos:
            print("AVISO: A pasta existe mas está VAZIA!")
    else:
        print(f"ERRO CRÍTICO: A pasta {saved_models_path} NÃO EXISTE no disco!")
    print("---------------------------\n")

    successful_runs = []

    # --- LOOP 1: DATASETS ---
    for ds_name, ds_conf in datasets_dict.items():
        horizons = ds_conf.get('forecast_horizon', [24])
        if not isinstance(horizons, list): horizons = [horizons]

        # --- LOOP 2: HORIZONTES ---
        for horizon in horizons:
            print(f"\n>>> PROCESSANDO: {ds_name} | H={horizon} <<<")
            
            current_ds_conf = ds_conf.copy()
            current_ds_conf['forecast_horizon'] = horizon
            current_ds_conf['name'] = ds_name

            # --- LOOP 3: MODELOS BASE ---
            for m_conf in base_queue:
                exec_name = f"{ds_name}_{m_conf['unique_exec_name']}_h{horizon}"
                if not force_rerun and check_metrics_exists(metrics_path, exec_name):
                    successful_runs.append(exec_name)
                    continue
                success = execute_step(main_config, m_conf, current_ds_conf, exec_name)
                if success: successful_runs.append(exec_name)

            # --- LOOP 4: MODELOS HÍBRIDOS ---
            # Se a lista hybrid_queue estiver vazia, o problema é no config_loader
            if not hybrid_queue:
                print("ERRO: A fila de híbridos está vazia! Verifique seu model_params.yaml")
                
            for m_conf in hybrid_queue:
                base_dep = m_conf['depends_on']
                
                # Monta os nomes esperados
                base_exec_name = f"{ds_name}_{base_dep}_h{horizon}"
                exec_name = f"{ds_name}_{m_conf['unique_exec_name']}_h{horizon}"
                
                # Nome do arquivo físico esperado
                base_filename = f"{base_exec_name}.joblib"
                base_model_path = os.path.join(saved_models_path, base_filename)

                # Verifica Checkpoint do Híbrido
                if not force_rerun and check_metrics_exists(metrics_path, exec_name):
                    successful_runs.append(exec_name)
                    continue

                # --- DEBUG DE DEPENDÊNCIA ---
                if os.path.exists(base_model_path):
                    print(f"  [OK] Base encontrada: {base_filename}. Iniciando treino do Híbrido...")
                    success = execute_step(main_config, m_conf, current_ds_conf, exec_name)
                    if success: successful_runs.append(exec_name)
                else:
                    print(f"  [FALHA] Não encontrei a base para o híbrido.")
                    print(f"          Procurando por: {base_model_path}")
                    print(f"          Dica: Verifique se o nome '{base_dep}' bate com o arquivo na pasta.")

    return successful_runs

def check_metrics_exists(path, name):
    return os.path.exists(os.path.join(path, f"metrics_{name}.csv"))

def execute_step(config, m_conf, d_conf, name):
    from src.pipelines import train_pipeline, evaluate_pipeline
    try:
        train_pipeline.run(m_conf, d_conf, config, name)
        evaluate_pipeline.run(config, m_conf, d_conf, name)
        return True
    except Exception as e:
        logging.error(f"  [ERRO] Falha em {name}: {e}")
        return False