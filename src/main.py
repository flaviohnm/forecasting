import logging
import argparse
import os
import glob
import warnings

from src.utils.config_loader import load_config
from src.utils.experiment_manager import run_custom_pipeline
from src.reporting import reporter
from src.analysis.statistical_tests import run_significance_analysis

# --- FILTRO DE AVISOS (Manter logs limpos) ---
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*'H' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*'T' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*'S' is deprecated.*")

# Ignora aviso sobre construtor 'df' do StatsForecast
warnings.filterwarnings(
    "ignore", message=".*The `df` argument of the StatsForecast constructor.*")

# --- NOVO FILTRO: Ignora aviso sobre NIXTLA_ID_AS_COL ---
warnings.filterwarnings("ignore", message=".*NIXTLA_ID_AS_COL.*")
# -------------------------------------------------------


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def scan_existing_results(metrics_path):
    """
    Escaneia a pasta de resultados para identificar modelos já treinados.
    Útil quando rodamos apenas o modo 'report'.
    """
    if not os.path.exists(metrics_path):
        return []

    csv_files = glob.glob(os.path.join(metrics_path, "metrics_*.csv"))
    found_runs = []
    for f in csv_files:
        filename = os.path.basename(f)
        # Remove prefixo 'metrics_' e sufixo '.csv'
        exec_name = filename.replace("metrics_", "").replace(".csv", "")
        found_runs.append(exec_name)

    return found_runs


def main():
    setup_logging()

    # --- 1. Configuração de Argumentos (CLI) ---
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting Pipeline")

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'train', 'report'],
        help="Escolha a etapa: 'train' (só treina), 'report' (só relatórios) ou 'all' (ambos)."
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help="Se usado, força o re-treinamento mesmo se o modelo já existir.")

    args = parser.parse_args()

    # --- 2. Carregar Configurações ---
    config_path = "./config/main_config.yaml"
    model_path = "./config/model_params.yaml"

    try:
        main_config = load_config(config_path)
    except FileNotFoundError:
        logging.error("ERRO CRÍTICO: main_config.yaml não encontrado.")
        return

    successful_runs = []

    # --- 3. ETAPA DE TREINAMENTO ---
    if args.mode in ['all', 'train']:
        logging.info(">>> MODO: EXECUÇÃO (Treino & Avaliação) <<<")
        # Pode parametrizar isso no argparse futuramente
        target_datasets = ['ETTh1','ETTh2','ETTm1']

        successful_runs = run_custom_pipeline(main_config,
                                              model_path,
                                              target_datasets=target_datasets,
                                              force_rerun=args.force)

    # --- 4. ETAPA DE RELATÓRIOS ---
    if args.mode in ['all', 'report']:
        logging.info(">>> MODO: REPORTING (Gráficos & Estatísticas) <<<")

        # Se pulamos o treino, precisamos descobrir o que existe no disco
        if not successful_runs:
            metrics_path = main_config['results_paths']['metrics']
            logging.info("Escaneando disco por resultados existentes...")
            successful_runs = scan_existing_results(metrics_path)

        if successful_runs:
            logging.info(
                f"Gerando artefatos para {len(successful_runs)} modelos.")

            # Gráficos
            try:
                reporter.generate_plots(main_config, successful_runs)
            except Exception as e:
                logging.error(f"Erro nos gráficos: {e}")

            # Relatório Markdown
            try:
                reporter.generate_report(main_config, successful_runs)
            except Exception as e:
                logging.error(f"Erro no relatório: {e}")
        else:
            logging.warning(
                "Nenhum resultado encontrado para gerar relatórios.")


if __name__ == "__main__":
    main()
