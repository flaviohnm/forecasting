import logging
import os
import traceback

from src.utils.config_loader import load_and_organize_strategies


def run_custom_pipeline(main_config, yaml_model_path, target_datasets=None, force_rerun=False):
    base_queue, hybrid_queue = load_and_organize_strategies(yaml_model_path)

    raw_datasets = main_config.get("datasets", {})
    datasets_dict = raw_datasets if isinstance(raw_datasets, dict) else {ds: {} for ds in raw_datasets}
    if target_datasets:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in target_datasets}

    metrics_path = main_config["results_paths"]["metrics"]
    saved_models_path = main_config["results_paths"]["saved_models"]

    print("\n--- [DEBUG DIAGNÓSTICO] ---")
    print(f"Pasta de modelos configurada: {saved_models_path}")
    if os.path.exists(saved_models_path):
        arquivos = os.listdir(saved_models_path)
        print(f"Total de arquivos na pasta: {len(arquivos)}")
    else:
        print("AVISO: Pasta de modelos não existe ainda.")
    print("---------------------------\n")

    successful_runs = []

    for ds_name, current_ds_conf in datasets_dict.items():
        logging.info(f"=== Dataset: {ds_name} ===")
        current_ds_conf["name"] = ds_name

        raw_horizons = current_ds_conf.get("forecast_horizon", current_ds_conf.get("horizon", [96]))
        horizons = raw_horizons if isinstance(raw_horizons, list) else [raw_horizons]

        for horizon in horizons:
            logging.info(f">>> Iniciando bateria para o Horizonte: {horizon} <<<")

            run_ds_conf = current_ds_conf.copy()
            run_ds_conf["forecast_horizon"] = horizon
            run_ds_conf["horizon"] = horizon

            # --- EXECUÇÃO DOS MODELOS BASE ---
            for m_conf in base_queue:
                model_type = m_conf.get("model_type", "unknown").lower()

                # LÊ A CHAVE CORRETA DO SEU YAML
                yaml_name = m_conf.get("model_name", m_conf.get("name"))
                if yaml_name:
                    exec_name = f"{ds_name}_{yaml_name}_h{horizon}"
                else:
                    exec_name = f"{ds_name}_base_{model_type}_h{horizon}"

                if not force_rerun and check_metrics_exists(metrics_path, exec_name):
                    successful_runs.append(exec_name)
                    continue

                success = execute_step(main_config, m_conf, run_ds_conf, exec_name)
                if success:
                    successful_runs.append(exec_name)

            # --- EXECUÇÃO DOS MODELOS HÍBRIDOS ---
            for m_conf in hybrid_queue:
                model_type = m_conf.get("model_type", "unknown").lower()
                base_dep_raw = m_conf.get("depends_on", "")
                base_dep = base_dep_raw.replace("base_", "").lower()

                # LÊ A CHAVE CORRETA DO SEU YAML
                yaml_name = m_conf.get("model_name", m_conf.get("name"))
                if yaml_name:
                    exec_name = f"{ds_name}_{yaml_name}_h{horizon}"
                else:
                    clean_type = model_type.replace("hybrid_", "")
                    exec_name = f"{ds_name}_hybrid_{clean_type}_on_{base_dep}_h{horizon}"

                if base_dep:
                    base_exec_name = f"{ds_name}_base_{base_dep}_h{horizon}"
                    base_filename = f"{base_exec_name}.joblib"
                    base_model_path = os.path.join(saved_models_path, base_filename)

                    if not force_rerun and check_metrics_exists(metrics_path, exec_name):
                        successful_runs.append(exec_name)
                        continue

                    if os.path.exists(base_model_path):
                        logging.info(f"  [OK] Base encontrada: {base_filename}. Iniciando Híbrido...")
                        success = execute_step(main_config, m_conf, run_ds_conf, exec_name)
                        if success:
                            successful_runs.append(exec_name)
                    else:
                        logging.warning(
                            f"  [FALHA] Base não encontrada para o híbrido. Procurando por: {base_model_path}"
                        )

    return successful_runs


def check_metrics_exists(path, name):
    return os.path.exists(os.path.join(path, f"metrics_{name}.csv"))


def execute_step(config, m_conf, d_conf, name):
    from src.pipelines import evaluator, trainer

    try:
        trainer.run(config, m_conf, d_conf, name)
        evaluator.run(config, m_conf, d_conf, name)
        return True
    except Exception as e:
        logging.error(f"  [ERRO CRÍTICO] O modelo {name} falhou durante a execução:")
        logging.error(traceback.format_exc())
        return False
