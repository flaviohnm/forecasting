import logging
import os

import pandas as pd
import requests


def load_dataset(dataset_conf):
    """
    Carrega o dataset e implementa cache do arquivo limpo (apenas Data e OT).
    Retorna: DataFrame com colunas ['unique_id', 'ds', 'y']
    """
    raw_file_path = dataset_conf["path"]
    url = dataset_conf.get("download_url")

    # Define o nome do arquivo limpo (ex: data/ETTh1_clean.csv)
    clean_file_path = raw_file_path.replace(".csv", "_clean.csv")

    # 1. CACHE: Se o arquivo tratado já existe, carrega ele direto (muito mais rápido)
    if os.path.exists(clean_file_path):
        logging.info(f"Dataset limpo encontrado. Carregando cache univariado de: {clean_file_path}")
        df_processed = pd.read_csv(clean_file_path)
        df_processed["ds"] = pd.to_datetime(df_processed["ds"])
        return df_processed

    # 2. RAW: Se não existe o limpo, garante que o arquivo bruto original exista
    _ensure_data_exists(raw_file_path, url)

    # 3. PROCESSAMENTO: Lê o bruto e faz a faxina
    try:
        logging.info(f"Lendo arquivo bruto de: {raw_file_path}")
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        raise IOError(f"Falha ao ler o arquivo CSV bruto em {raw_file_path}: {e}")

    target_col = dataset_conf.get("target_column", "OT")
    date_col = dataset_conf.get("date_column", "date")

    if target_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"Colunas '{date_col}' ou '{target_col}' não encontradas no arquivo bruto.")

    # Isola estritamente as colunas de Data e OT (Univariado)
    df_processed = df[[date_col, target_col]].copy()
    df_processed.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    df_processed["ds"] = pd.to_datetime(df_processed["ds"])

    dataset_name = dataset_conf.get("name", "series_1")
    df_processed["unique_id"] = dataset_name

    # Reordena e limpa índices
    cols = ["unique_id", "ds", "y"]
    df_processed = df_processed[cols].sort_values("ds").reset_index(drop=True)

    # 4. SALVAMENTO: Grava o arquivo limpo no disco para as próximas execuções
    df_processed.to_csv(clean_file_path, index=False)
    logging.info(f"Dataset limpo e salvo com sucesso em: {clean_file_path}. Shape: {df_processed.shape}")

    return df_processed


def _ensure_data_exists(file_path, url):
    """Verifica se o arquivo bruto existe; se não, baixa da fonte oficial."""
    if os.path.exists(file_path):
        return

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not url:
        raise FileNotFoundError(f"Arquivo bruto {file_path} não encontrado e nenhuma URL fornecida.")

    logging.info(f"Arquivo bruto não encontrado. Baixando de: {url}")
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        logging.info("Download do arquivo bruto concluído com sucesso.")
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar dataset bruto: {e}")
