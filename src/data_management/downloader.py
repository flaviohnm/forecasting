# File: src/data_management/downloader.py

import os
import requests
import pandas as pd
import statsmodels.api as sm


def download_from_url(url: str, file_path: str):
    """Baixa um arquivo de uma URL."""
    print(f"Baixando de {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo salvo em: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Falha no download de '{os.path.basename(file_path)}': {e}")
        raise


def load_from_statsmodels(dataset_name: str, file_path: str):
    """
    Carrega um dataset da biblioteca statsmodels e o salva como um CSV.
    """
    print(f"Carregando dataset '{dataset_name}' da biblioteca statsmodels...")
    series = None
    try:
        if dataset_name == 'AirPassengers':
            df = sm.datasets.get_rdataset("AirPassengers").data
            series = pd.Series(df['value'].values,
                               index=pd.date_range(start='1949-01-01',
                                                   periods=len(df),
                                                   freq='MS'),
                               name="AirPassengers")
        elif dataset_name == 'co2_sm':  # Renomeado para não conflitar com o 'co2' da URL
            data = sm.datasets.co2.load_pandas().data
            series = data['co2'].resample('MS').mean().ffill().rename(
                "CO2_sm")  # Agrupado por mês
        elif dataset_name == 'nottem':
            df = sm.datasets.get_rdataset("nottem").data
            series = pd.Series(df['value'].values,
                               index=pd.date_range(start='1920-01-01',
                                                   periods=len(df),
                                                   freq='MS'),
                               name="NottinghamTemp")
        elif dataset_name == 'JohnsonJohnson':
            df = sm.datasets.get_rdataset("JohnsonJohnson").data
            series = pd.Series(df['value'].values,
                               index=pd.date_range(start='1960-01-01',
                                                   periods=len(df),
                                                   freq='QE'),
                               name="JohnsonJohnson")
        elif dataset_name == 'UKgas':
            df = sm.datasets.get_rdataset("UKgas").data
            series = pd.Series(df['value'].values,
                               index=pd.date_range(start='1960-01-01',
                                                   periods=len(df),
                                                   freq='QE'),
                               name="UKGas")
        elif dataset_name == 'sunspots_sm':  # Renomeado para não conflitar com o 'sunspot' da URL
            df = sm.datasets.sunspots.load_pandas().data
            series = pd.Series(df['SUNACTIVITY'].values,
                               index=pd.to_datetime(df['YEAR'].dt.year,
                                                    format='%Y'),
                               name="Sunspots_sm")
        elif dataset_name == 'Nile':
            df = sm.datasets.nile.load_pandas().data.reset_index()
            series = pd.Series(df['volume'].values,
                               index=pd.to_datetime(df['year'], format='%Y'),
                               name="Nile")
        elif dataset_name == 'ukdriverdeaths':
            df = sm.datasets.get_rdataset("UKDriverDeaths").data
            series = pd.Series(df['value'].values,
                               index=pd.date_range(start='1969-01-01',
                                                   periods=len(df),
                                                   freq='MS'),
                               name="UKDriverDeaths")
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' de statsmodels não reconhecido.")

        # Converte a série para um DataFrame com uma coluna de data explícita
        df_to_save = series.reset_index()
        df_to_save.columns = ['date', series.name]

        df_to_save.to_csv(file_path, index=False)
        print(f"Dataset salvo como CSV em: {file_path}")

    except Exception as e:
        import traceback
        print(
            f"Erro ao carregar o dataset '{dataset_name}' de statsmodels: {e}\n{traceback.format_exc()}"
        )
        raise


def prepare_raw_data(main_config: dict):
    """
    Orquestra a preparação de todos os datasets definidos no config.
    Ele verifica a fonte ('url' ou 'statsmodels') e chama a função apropriada.
    """
    raw_data_path = main_config['data_paths']['raw']
    os.makedirs(raw_data_path, exist_ok=True)

    if 'datasets' not in main_config or not main_config['datasets']:
        print("Nenhum dataset definido em './config/main_config.yaml'.")
        return

    for dataset_conf in main_config['datasets']:
        file_path = os.path.join(raw_data_path, dataset_conf['filename'])

        if os.path.exists(file_path):
            print(
                f"Arquivo '{dataset_conf['filename']}' já existe. Preparação pulada."
            )
            continue

        source = dataset_conf.get('source', 'url')

        if source == 'url':
            download_from_url(dataset_conf['url'], file_path)
        elif source == 'statsmodels':
            load_from_statsmodels(dataset_conf['name'], file_path)
        else:
            print(
                f"AVISO: Fonte '{source}' desconhecida para o dataset '{dataset_conf['name']}'. Pulando."
            )
