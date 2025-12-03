# File: src/data_management/downloader.py

import os
import requests
import pandas as pd
import statsmodels.api as sm
import logging
import traceback
import time
import urllib.error

MAX_RETRIES = 3 # Número máximo de tentativas para cada download

def download_from_url(url: str, file_path: str):
    """Baixa um arquivo de uma URL."""
    print(f"Baixando de {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo salvo em: {file_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Falha no download de '{os.path.basename(file_path)}': {e}")
        raise

def load_from_statsmodels(dataset_name: str, file_path: str):
    """
    Carrega um dataset da biblioteca statsmodels, com lógica de re-tentativa robusta
    para evitar erros de 'Too Many Requests', e salva como CSV padronizado.
    Aproveita o cache interno da statsmodels.
    """
    print(f"Tentando carregar '{dataset_name}' da statsmodels (pode precisar baixar)...")

    for attempt in range(MAX_RETRIES):
        series = None
        data_loaded = False # Flag para saber se o carregamento foi bem-sucedido
        try:
            # --- Tenta carregar/baixar os dados ---
            if dataset_name == 'AirPassengers':
                df = sm.datasets.get_rdataset("AirPassengers", cache=True).data
                series = pd.Series(df['value'].values,
                                   index=pd.date_range(start='1949-01-01',
                                   periods=len(df),
                                   freq='MS'),
                                   name="AirPassengers"
                )
            elif dataset_name == 'co2':
                data = sm.datasets.co2.load_pandas().data # Geralmente já vem com statsmodels
                data.index = pd.to_datetime(data.index)
                series = data['co2'].resample('MS').mean().ffill().rename("CO2")
            elif dataset_name == 'nottem':
                df = sm.datasets.get_rdataset("nottem", cache=True).data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1920-01-01', periods=len(df), freq='MS'), name="NottinghamTemp")
            elif dataset_name == 'JohnsonJohnson':
                df = sm.datasets.get_rdataset("JohnsonJohnson", cache=True).data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1960-03-31', periods=len(df), freq='QE'), name="JohnsonJohnson")
            elif dataset_name == 'UKgas':
                df = sm.datasets.get_rdataset("UKgas", cache=True).data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1960-03-31', periods=len(df), freq='QE'), name="UKGas")
            elif dataset_name == 'sunspots':
                df = sm.datasets.sunspots.load_pandas().data # Geralmente já vem com statsmodels
                # CORREÇÃO: Usa a coluna 'YEAR' para criar o índice
                series = pd.Series(df['SUNACTIVITY'].values, index=pd.to_datetime(df['YEAR'], format='%Y'), name="Sunspots")
                series.index.freq = 'YS-JAN'
            elif dataset_name == 'Nile':
                df = sm.datasets.nile.load_pandas().data # Geralmente já vem com statsmodels
                # CORREÇÃO: Usa a coluna 'year' para criar o índice
                series = pd.Series(df['volume'].values, index=pd.to_datetime(df['year'], format='%Y'), name="Nile")
                series.index.freq = 'YS-JAN'
            elif dataset_name == 'ukdriverdeaths':
                df = sm.datasets.get_rdataset("UKDriverDeaths", cache=True).data
                series = pd.Series(df['value'].values, index=pd.date_range(start='1969-01-01', periods=len(df), freq='MS'), name="UKDriverDeaths")
            else:
                raise ValueError(f"Dataset '{dataset_name}' de statsmodels não reconhecido.")

            data_loaded = True # Marca que o carregamento deu certo

            # --- Salva o arquivo CSV ---
            if series is None:
                 raise ValueError(f"Falha ao criar a série para '{dataset_name}'.")

            df_to_save = series.reset_index()
            df_to_save.columns = ['date', series.name]
            # Converte a data para string ANTES de salvar (mais seguro)
            df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d')
            df_to_save.to_csv(file_path, index=False)
            print(f"Dataset '{dataset_name}' salvo com sucesso em: {file_path}")
            return # Sai da função após sucesso

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # --- BACKOFF EXPONENCIAL ---
                backoff_time = 15 * (2 ** attempt) # 15s, 30s, 60s
                logging.warning(f"Erro 429 para '{dataset_name}'. Tentativa {attempt + 1}/{MAX_RETRIES}. Aguardando {backoff_time}s...")
                time.sleep(backoff_time)
            else:
                logging.error(f"Erro HTTP inesperado ao carregar '{dataset_name}': {e}\n{traceback.format_exc()}")
                raise e # Falha imediatamente para outros erros HTTP
        except Exception as e:
            logging.error(f"Erro ao processar '{dataset_name}': {e}\n{traceback.format_exc()}")
            raise e # Falha imediatamente para outros erros

    # Se o loop terminar sem sucesso após todas as tentativas
    if not data_loaded:
        raise Exception(f"Falha ao carregar o dataset '{dataset_name}' da statsmodels após {MAX_RETRIES} tentativas devido ao erro 429.")

def prepare_raw_data(main_config: dict):
    """
    Orquestra a preparação (download ou carga) de todos os datasets definidos no config.
    """
    raw_data_path = main_config['data_paths']['raw']
    os.makedirs(raw_data_path, exist_ok=True)

    if 'datasets' not in main_config or not main_config['datasets']:
        print("Nenhum dataset definido em './config/main_config.yaml'.")
        return

    print("Iniciando preparação dos datasets. A primeira execução pode ser lenta devido a downloads e limites de taxa do servidor statsmodels.")
    print("Execuções subsequentes usarão o cache local e serão mais rápidas.")

    for dataset_conf in main_config['datasets']:
        if not dataset_conf or 'name' not in dataset_conf:
            continue

        file_path = os.path.join(raw_data_path, dataset_conf['filename'])

        if os.path.exists(file_path):
            print(f"\nArquivo '{dataset_conf['filename']}' ({dataset_conf['name']}) já existe. Pulado.")
            continue

        source = dataset_conf.get('source', 'url')

        try:
            print(f"\nPreparando dataset: {dataset_conf['name']} (Fonte: {source})")
            if source == 'url':
                if 'url' not in dataset_conf:
                     raise ValueError(f"Dataset '{dataset_conf['name']}' com source 'url', mas chave 'url' está ausente.")
                download_from_url(dataset_conf['url'], file_path)
            elif source == 'statsmodels':
                # A lógica de espera/re-tentativa está DENTRO de load_from_statsmodels
                load_from_statsmodels(dataset_conf['name'], file_path)
                # Pequeno delay adicional opcional após sucesso, pode ajudar
                # print("  Aguardando 1 segundo extra...")
                # time.sleep(1)
            else:
                logging.warning(f"Fonte '{source}' desconhecida para '{dataset_conf['name']}'. Pulando.")
        except Exception as e:
             # Loga o erro mas continua para outros datasets
             logging.error(f"Falha crítica ao preparar '{dataset_conf['name']}': {e}")
             continue

    print("\nPreparação de todos os datasets solicitados concluída (ou tentativas esgotadas).")