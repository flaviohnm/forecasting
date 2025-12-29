# File: src/data_management/downloader.py

import os
import requests
import pandas as pd
import statsmodels.api as sm
import logging
import traceback
import time
import urllib.error

MAX_RETRIES = 3


def download_from_url(url: str, file_path: str):
    """Baixa um arquivo de uma URL com retry básico."""
    print(f"Baixando de {url}...")
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Verificação básica: o arquivo é um CSV válido?
            pd.read_csv(file_path, nrows=5)
            print(f"Arquivo salvo e verificado em: {file_path}")
            return
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5 * (attempt + 1)
                logging.warning(
                    f"Erro ao baixar {url}. Tentando novamente em {wait}s... ({e})")
                time.sleep(wait)
            else:
                logging.error(f"Falha definitiva no download de {url}")
                raise


def load_from_statsmodels(dataset_name: str, file_path: str):
    """(Sua lógica atual permanece a mesma, omitida aqui para brevidade)"""
    # ... (mantenha todo o código da sua função load_from_statsmodels aqui)
    pass


def prepare_raw_data(main_config: dict):
    """
    Orquestra a preparação (download ou carga) de todos os datasets definidos no config.
    """
    raw_data_path = main_config['data_paths']['raw']
    os.makedirs(raw_data_path, exist_ok=True)

    if 'datasets' not in main_config or not main_config['datasets']:
        print("Nenhum dataset definido em './config/main_config.yaml'.")
        return

    print("Iniciando preparação dos datasets...")

    for dataset_conf in main_config['datasets']:
        if not dataset_conf or 'name' not in dataset_conf:
            continue

        file_path = os.path.join(raw_data_path, dataset_conf['filename'])

        if os.path.exists(file_path):
            print(f"Arquivo '{dataset_conf['filename']}' já existe. Pulado.")
            continue

        source = dataset_conf.get('source', 'url')

        try:
            print(
                f"\nPreparando dataset: {dataset_conf['name']} (Fonte: {source})")

            # Ajuste aqui para aceitar 'url' ou 'external'
            if source in ['url', 'external']:
                if 'url' not in dataset_conf:
                    raise ValueError(
                        f"Dataset '{dataset_conf['name']}' sem URL definida.")
                download_from_url(dataset_conf['url'], file_path)

            elif source == 'statsmodels':
                load_from_statsmodels(dataset_conf['name'], file_path)

            else:
                logging.warning(
                    f"Fonte '{source}' desconhecida para '{dataset_conf['name']}'.")

        except Exception as e:
            logging.error(
                f"Falha crítica ao preparar '{dataset_conf['name']}': {e}")
            continue

    print("\nPreparação concluída.")
