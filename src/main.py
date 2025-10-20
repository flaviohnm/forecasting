# File: src/main.py

import os
import logging
# Esconde as mensagens
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NIXTLA_ID_AS_COL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR) # Suprime logs do PL

import argparse
import yaml
import random
import numpy as np
import tensorflow as tf
import pytorch_lightning as pl

from src.data_management import downloader
from src.pipelines import train_pipeline, evaluate_pipeline
from src.visualization import plotter
from src.analysis import statistical_tests, comparison_tests
from src.reporting import reporter

SEED = 42

def get_executions_to_run(main_config, model_params, args):
    # ... (sem alterações) ...
    strategy_name = args.strategy
    if strategy_name not in model_params['strategies']:
        raise ValueError(f"Estratégia '{strategy_name}' não encontrada.")
    strategy_steps = model_params['strategies'][strategy_name]
    if 'all' not in args.model:
        model_names_in_strategy = {s['model_name'] for s in strategy_steps}
        requested_models = set(args.model)
        valid_models = requested_models.intersection(model_names_in_strategy)
        invalid_models = requested_models.difference(model_names_in_strategy)
        if invalid_models: print(f"AVISO: Modelos não encontrados na estratégia: {', '.join(invalid_models)}")
        if not valid_models: print(f"Nenhum modelo válido encontrado."); return []
        strategy_steps = [s for s in strategy_steps if s['model_name'] in valid_models]
    available_datasets = {ds['name']: ds for ds in main_config['datasets']}
    datasets_to_run = []
    if 'all' in args.dataset:
        datasets_to_run = list(available_datasets.values())
    else:
        requested_datasets = set(args.dataset)
        valid_datasets = {name: available_datasets[name] for name in requested_datasets if name in available_datasets}
        invalid_datasets = requested_datasets.difference(valid_datasets.keys())
        if invalid_datasets: print(f"AVISO: Datasets não encontrados: {', '.join(invalid_datasets)}")
        if not valid_datasets: print(f"Nenhum dataset válido encontrado."); return []
        datasets_to_run = list(valid_datasets.values())
    return [(ds, ms) for ds in datasets_to_run for ms in strategy_steps]


def run_pipeline_step(step_name, main_config, executions_to_run):
    # ... (sem alterações, mas retorna successful_executions) ...
    pipeline_func = train_pipeline.run if step_name == 'train' else evaluate_pipeline.run
    total_runs = len(executions_to_run)
    current_run = 0
    successful_executions = [] 
    for dataset_conf, model_conf in executions_to_run:
        current_run += 1
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        print(f"\n--- Executando {step_name.upper()} ({current_run}/{total_runs}) ---")
        print(f"  Dataset: {dataset_conf['name']} | Modelo: {model_conf['model_name']}")
        try:
            pipeline_func(main_config, model_conf, dataset_conf, execution_name)
            if step_name == 'evaluate': # Só adiciona à lista se a avaliação for bem-sucedida
                 successful_executions.append((dataset_conf, model_conf))
        except Exception as e:
            import traceback
            print(f"ERRO na execução '{execution_name}': {e}\n{traceback.format_exc()}")
            continue
    return successful_executions


def main():
    """Ponto de entrada principal que orquestra a pipeline."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    parser = argparse.ArgumentParser(description="Pipeline de Forecasting.")
    parser.add_argument('--step', default='full_run',
                        choices=['full_run', 'download', 'train', 'evaluate',
                                 'plot', 'report', 'testes', 'comparison_tests'], # Renomeado dm_test
                        help="Etapa a executar.")
    parser.add_argument('--strategy', default='full_comparison', help="Estratégia a ser executada.")
    parser.add_argument('--dataset', nargs='+', default=['all'], help="Datasets.")
    parser.add_argument('--model', nargs='+', default=['all'], help="Modelos.")
    args = parser.parse_args()

    with open('./config/main_config.yaml', 'r') as f: main_config = yaml.safe_load(f)
    with open('./config/model_params.yaml', 'r') as f: model_params = yaml.safe_load(f)

    # Cria pastas
    for path in main_config['data_paths'].values(): os.makedirs(path, exist_ok=True)
    for path in main_config['results_paths'].values(): os.makedirs(path, exist_ok=True)
    os.makedirs(main_config['models_path'], exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("results", "statistical_tests"), exist_ok=True)
    os.makedirs(os.path.join("results", "comparison_tests"), exist_ok=True)

    executions = get_executions_to_run(main_config, model_params, args)
    if not executions: print("Nenhuma execução válida encontrada."); return

    evaluated_executions = []

    if args.step == 'full_run':
        print("--- INICIANDO EXECUÇÃO COMPLETA DA PIPELINE ---")
        print("\n" + "=" * 50 + "\n[ETAPA DE DOWNLOAD]\n" + "=" * 50)
        downloader.prepare_raw_data(main_config)
        print("\n" + "=" * 50 + "\n[ETAPA DE ANÁLISE ESTATÍSTICA]\n" + "=" * 50)
        # Passa apenas a lista de configs de dataset únicos
        unique_datasets = list({ds['name']: ds for ds, ms in executions}.values())
        statistical_tests.run_tests(main_config, unique_datasets) 
        print("\n" + "=" * 50 + "\n[ETAPA DE TREINAMENTO]\n" + "=" * 50)
        run_pipeline_step('train', main_config, executions) # Treina todos
        print("\n" + "=" * 50 + "\n[ETAPA DE AVALIAÇÃO]\n" + "=" * 50)
        evaluated_executions = run_pipeline_step('evaluate', main_config, executions) # Avalia e obtém lista de sucesso
        if evaluated_executions:
            print("\n" + "=" * 50 + "\n[ETAPA DE VISUALIZAÇÃO]\n" + "=" * 50)
            plotter.generate_plots(main_config, evaluated_executions)
            print("\n" + "=" * 50 + "\n[ETAPA DE TESTES DE COMPARAÇÃO]\n" + "=" * 50) # Nome genérico
            comparison_tests.run_tests(main_config, evaluated_executions) # Chama a função principal do módulo
            print("\n" + "=" * 50 + "\n[ETAPA DE RELATÓRIO]\n" + "=" * 50)
            reporter.generate_report(main_config, evaluated_executions)
        else: print("\nAVISO: Nenhuma avaliação bem-sucedida. Etapas seguintes puladas.")
        print("\n--- EXECUÇÃO COMPLETA CONCLUÍDA ---")

    elif args.step == 'download':
        downloader.prepare_raw_data(main_config)
    elif args.step == 'testes':
        unique_datasets = list({ds['name']: ds for ds, ms in executions}.values())
        statistical_tests.run_tests(main_config, unique_datasets)
    elif args.step == 'train':
        run_pipeline_step('train', main_config, executions)
    elif args.step == 'evaluate':
        run_pipeline_step('evaluate', main_config, executions)
    elif args.step == 'plot':
        # Para plotar, precisa saber quais tiveram sucesso. Idealmente, rodar evaluate antes.
        # Vamos tentar plotar todas as solicitadas, a função plotter deve lidar com arquivos ausentes.
        plotter.generate_plots(main_config, executions) 
    elif args.step == 'comparison_tests': # Nome atualizado
        print("\n" + "=" * 50 + "\n[ETAPA DE TESTES DE COMPARAÇÃO]\n" + "=" * 50)
        # Tenta rodar para todas as execuções solicitadas. A função interna filtra.
        comparison_tests.run_tests(main_config, executions) 
    elif args.step == 'report':
        # O relatório filtra internamente as execuções com métricas salvas.
        reporter.generate_report(main_config, executions)

if __name__ == '__main__':
    main()