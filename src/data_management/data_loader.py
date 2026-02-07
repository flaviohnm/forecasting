import os
import logging
import requests
import pandas as pd

def load_dataset(dataset_conf):
    """
    Carrega, valida e padroniza o dataset para o formato Long (Nixtla).
    Retorna: DataFrame com colunas ['unique_id', 'ds', 'y']
    """
    file_path = dataset_conf['path']
    url = dataset_conf.get('download_url')
    
    # 1. Garante que o arquivo existe (Download automático)
    _ensure_data_exists(file_path, url)

    # 2. Leitura do CSV
    try:
        logging.info(f"Carregando dataset de: {file_path}")
        df = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Falha ao ler o arquivo CSV em {file_path}: {e}")

    # 3. Validação de Colunas
    target_col = dataset_conf.get('target_column', 'OT')
    date_col = dataset_conf.get('date_column', 'date')

    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no CSV. Disponíveis: {df.columns.tolist()}")
    
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada no CSV.")

    # 4. Padronização para Formato Nixtla (unique_id, ds, y)
    # Selecionamos apenas as colunas necessárias para análise univariada/híbrida
    df_processed = df[[date_col, target_col]].copy()
    
    # Renomeia para padrão
    df_processed.columns = ['ds', 'y']
    
    # Converte para datetime (Critical para Time Series)
    try:
        df_processed['ds'] = pd.to_datetime(df_processed['ds'])
    except Exception as e:
        raise ValueError(f"Erro ao converter coluna de data para datetime: {e}")

    # Cria unique_id (Necessário para NeuralForecast lidar com múltiplas séries se fosse o caso)
    # Usamos o nome do dataset (ex: 'ETTh1') como ID
    dataset_name = dataset_conf.get('name', 'series_1')
    df_processed['unique_id'] = dataset_name
    
    # Reordena colunas e linhas
    df_processed = df_processed[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)

    logging.info(f"Dataset processado com sucesso. Shape: {df_processed.shape}. Range: {df_processed['ds'].min()} até {df_processed['ds'].max()}")
    
    return df_processed

def _ensure_data_exists(file_path, url):
    """Verifica se o arquivo existe; se não, tenta baixar."""
    if os.path.exists(file_path):
        return

    # Cria diretório data/ se não existir
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not url:
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado e nenhuma URL de download fornecida.")

    logging.info(f"Arquivo não encontrado. Baixando de: {url}")
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status() # Garante que não houve erro 404/500
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logging.info("Download concluído com sucesso.")
        
    except Exception as e:
        raise ConnectionError(f"Falha ao baixar o dataset de {url}: {e}")