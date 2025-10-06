import os
import logging
# Esconde as mensagens do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NIXTLA_ID_AS_COL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import yaml
from src.data_management import downloader
from src.pipelines import train_pipeline, evaluate_pipeline
from src.visualization import plotter
from src.analysis import statistical_tests
from src.reporting import reporter


def get_executions_to_run(main_config, model_params, args):
    """
    Função centralizada para filtrar quais execuções devem rodar
    com base nos argumentos da linha de comando.
    Retorna uma lista de tuplas (dataset_conf, model_conf).
    """
    strategy_name = args.strategy
    if strategy_name not in model_params['strategies']:
        raise ValueError(
            f"Estratégia '{strategy_name}' não encontrada em model_params.yaml."
        )
    strategy_steps = model_params['strategies'][strategy_name]

    if 'all' not in args.model:
        strategy_steps = [
            s for s in strategy_steps if s['model_name'] in args.model
        ]
        if not strategy_steps:
            print(
                f"Nenhum modelo correspondente a '{args.model}' encontrado na estratégia '{strategy_name}'."
            )
            return []

    available_datasets = {ds['name']: ds for ds in main_config['datasets']}
    datasets_to_run = []
    if 'all' in args.dataset:
        datasets_to_run = list(available_datasets.values())
    else:
        datasets_to_run = [
            available_datasets[name] for name in args.dataset
            if name in available_datasets
        ]
        if not datasets_to_run:
            print(
                f"Nenhum dataset correspondente a '{args.dataset}' foi encontrado."
            )
            return []

    return [(ds, ms) for ds in datasets_to_run for ms in strategy_steps]


def run_pipeline_step(step_name, main_config, executions_to_run):
    """
    Função genérica que executa a etapa de treino ou avaliação.
    Recebe a lista de execuções já filtrada.
    """
    pipeline_func = train_pipeline.run if step_name == 'train' else evaluate_pipeline.run

    total_runs = len(executions_to_run)
    current_run = 0
    for dataset_conf, model_conf in executions_to_run:
        current_run += 1
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"

        print(
            f"\n--- Executando {step_name.upper()} ({current_run}/{total_runs}) ---"
        )
        print(
            f"  Dataset: {dataset_conf['name']} | Modelo: {model_conf['model_name']}"
        )

        try:
            pipeline_func(main_config, model_conf, dataset_conf,
                          execution_name)
        except Exception as e:
            print(f"ERRO na execução '{execution_name}': {e}")
            break


def main():
    """Ponto de entrada principal que orquestra a pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Forecasting para Mestrado.")

    parser.add_argument('--step',
                        default='full_run',
                        choices=[
                            'full_run', 'download', 'train', 'evaluate',
                            'plot', 'report', 'testes'
                        ],
                        help="Etapa a executar. Padrão: 'full_run'.")
    parser.add_argument(
        '--strategy',
        default='full_comparison',
        help="Estratégia a ser executada. Padrão: 'full_comparison'.")
    parser.add_argument('--dataset',
                        nargs='+',
                        default=['all'],
                        help="Um ou mais datasets. Padrão: ['all'].")
    parser.add_argument(
        '--model',
        nargs='+',
        default=['all'],
        help="Um ou mais modelos da estratégia. Padrão: ['all'].")

    args = parser.parse_args()

    # Carrega as configurações (mantendo seus caminhos customizados)
    with open('./config/main_config.yaml', 'r') as f:
        main_config = yaml.safe_load(f)
    with open('./config/model_params.yaml', 'r') as f:
        model_params = yaml.safe_load(f)

    # Cria pastas, se necessário
    os.makedirs(main_config['data_paths']['raw'], exist_ok=True)
    os.makedirs(main_config['results_paths']['metrics'], exist_ok=True)
    os.makedirs(main_config['results_paths']['plots'], exist_ok=True)
    os.makedirs(main_config['models_path'], exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("results", "statistical_tests"), exist_ok=True)

    # Lógica principal de execução
    if args.step == 'full_run':
        print("--- INICIANDO EXECUÇÃO COMPLETA DA PIPELINE ---")

        # Etapa 1: Download
        print("\n" + "=" * 50 + "\n[ETAPA DE DOWNLOAD]\n" + "=" * 50)
        downloader.download_all_datasets(main_config)

        # MUDANÇA: Etapa de Testes Estatísticos adicionada ao full_run
        print("\n" + "=" * 50 + "\n[ETAPA DE ANÁLISE ESTATÍSTICA]\n" +
              "=" * 50)
        available_datasets = {ds['name']: ds for ds in main_config['datasets']}
        datasets_to_run_tests = list(
            available_datasets.values())  # Roda para todos os datasets
        statistical_tests.run_tests(main_config, datasets_to_run_tests)

        # Etapas seguintes...
        executions = get_executions_to_run(main_config, model_params, args)

        print("\n" + "=" * 50 + "\n[ETAPA DE TREINAMENTO]\n" + "=" * 50)
        run_pipeline_step('train', main_config, executions)

        print("\n" + "=" * 50 + "\n[ETAPA DE AVALIAÇÃO]\n" + "=" * 50)
        run_pipeline_step('evaluate', main_config, executions)

        print("\n" + "=" * 50 + "\n[ETAPA DE VISUALIZAÇÃO]\n" + "=" * 50)
        plotter.generate_plots(main_config, executions)

        print("\n" + "=" * 50 + "\n[ETAPA DE RELATÓRIO]\n" + "=" * 50)
        reporter.generate_report(main_config, model_params, args)

        print("\n--- EXECUÇÃO COMPLETA CONCLUÍDA ---")

    elif args.step == 'download':
        downloader.download_all_datasets(main_config)

    elif args.step in ['train', 'evaluate']:
        executions = get_executions_to_run(main_config, model_params, args)
        run_pipeline_step(args.step, main_config, executions)

    elif args.step == 'plot':
        print("--- EXECUTANDO APENAS A ETAPA DE VISUALIZAÇÃO ---")
        executions = get_executions_to_run(main_config, model_params, args)
        plotter.generate_plots(main_config, executions)

    elif args.step == 'report':
        print("--- EXECUTANDO APENAS A ETAPA DE RELATÓRIO ---")
        reporter.generate_report(main_config, model_params, args)

    elif args.step == 'testes':
        print("--- EXECUTANDO APENAS A ETAPA DE ANÁLISE ESTATÍSTICA ---")

        available_datasets = {ds['name']: ds for ds in main_config['datasets']}
        datasets_to_run = []
        if 'all' in args.dataset:
            datasets_to_run = list(available_datasets.values())
        else:
            datasets_to_run = [
                available_datasets[name] for name in args.dataset
                if name in available_datasets
            ]
            if not datasets_to_run:
                print(
                    f"Nenhum dataset correspondente a '{args.dataset}' foi encontrado."
                )

        if datasets_to_run:
            statistical_tests.run_tests(main_config, datasets_to_run)


if __name__ == '__main__':
    main()
