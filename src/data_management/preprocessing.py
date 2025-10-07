# File: src/data_management/preprocessing.py

import pandas as pd
import numpy as np
import os

def load_and_prepare_data(main_config: dict, dataset_conf: dict):
    """
    Carrega, processa e divide os dados em treino/teste.
    """
    file_path = os.path.join(main_config['data_paths']['raw'], dataset_conf['filename'])
    
    df = pd.read_csv(file_path)
    
    date_col = dataset_conf['date_column']
    df[date_col] = pd.to_datetime(df[date_col], format=dataset_conf.get('date_format'))
    df = df.set_index(date_col)

    series = df[dataset_conf['target_column']].astype(float)
    series.index.freq = pd.infer_freq(series.index)

    if dataset_conf.get('log_transform', False):
        series = np.log(series)

    series = series.dropna()

    split_date = dataset_conf['split_date']
    train_series = series.loc[series.index < split_date]
    test_series = series.loc[series.index >= split_date]

    print(f"Dataset '{dataset_conf['name']}' carregado e processado.")
    return train_series, test_series

def create_direct_forecast_datasets(series: pd.Series, input_lags: int, forecast_horizon: int):
    """
    Cria H datasets para a abordagem Direta (um modelo por horizonte).
    """
    datasets = {}
    X, y_all = [], []
    
    for i in range(len(series) - input_lags - forecast_horizon + 1):
        X.append(series.iloc[i : i + input_lags].values)
        y_all.append(series.iloc[i + input_lags : i + input_lags + forecast_horizon].values)
    
    X = np.array(X)
    y_all = np.array(y_all)
    
    for h in range(1, forecast_horizon + 1):
        y_h = y_all[:, h-1]
        datasets[h] = (X, y_h)
        
    return datasets

def create_mimo_forecast_dataset(series: pd.Series, input_lags: int, forecast_horizon: int):
    """
    Cria um único dataset para a abordagem MIMO (Multi-Input Multi-Output).
    """
    X, y = [], []
    for i in range(len(series) - input_lags - forecast_horizon + 1):
        X.append(series.iloc[i : i + input_lags].values)
        y.append(series.iloc[i + input_lags : i + input_lags + forecast_horizon].values)
    
    return np.array(X), np.array(y)

def create_recursive_forecast_dataset(series: pd.Series, input_lags: int):
    """
    Cria um único dataset para a abordagem Recursiva (prevê apenas 1 passo à frente).
    """
    X, y = [], []
    # O horizonte é sempre 1 para o alvo na estratégia recursiva
    forecast_horizon = 1
    for i in range(len(series) - input_lags - forecast_horizon + 1):
        X.append(series.iloc[i : i + input_lags].values)
        y.append(series.iloc[i + input_lags])
    
    return np.array(X), np.array(y)