# /forecasting/src/data_processing.py (COM NOVO DATASET 'DAILY BIRTHS')
import pandas as pd
from pathlib import Path
import urllib.request

def fetch_airline_data(data_dir="data"):
    """
    Verifica se o dataset 'airline.csv' existe localmente.
    Se não existir, baixa de uma URL estável e o formata corretamente.
    """
    raw_path = Path(data_dir) / "raw" / "airline.csv"
    
    if raw_path.exists():
        print(f"Utilizando o dataset local 'airline.csv' encontrado em: {raw_path}")
        return str(raw_path)

    print(f"Arquivo local 'airline.csv' não encontrado. Baixando...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, raw_path)
        df = pd.read_csv(raw_path)
        df.rename(columns={"Month": "ds", "Passengers": "passengers"}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df.to_csv(raw_path, index=False)
        print(f"Dataset 'airline.csv' baixado e formatado com sucesso em {raw_path}")
        return str(raw_path)
    except Exception as e:
        raise ConnectionError(f"Falha ao baixar o arquivo 'airline.csv'. Erro: {e}")

def fetch_daily_births_data(data_dir="data"):
    """
    Verifica se o dataset 'daily_births.csv' existe localmente.
    Se não existir, baixa o arquivo do repositório de Jason Brownlee.
    """
    raw_path = Path(data_dir) / "raw" / "daily_births.csv"
    if raw_path.exists():
        print(f"Utilizando o dataset local 'daily_births.csv' encontrado em: {raw_path}")
        return str(raw_path)

    print(f"Arquivo local 'daily_births.csv' não encontrado. Baixando...")
    
    # URL estável e de formato simples
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, raw_path)
        print(f"Dataset 'daily-total-female-births.csv' baixado com sucesso em {raw_path}")

        # Renomeia as colunas para o padrão do nosso projeto ('ds', 'value')
        df = pd.read_csv(raw_path)
        df.rename(columns={"Date": "ds", "Births": "value"}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        # Salva o arquivo já formatado, sobrescrevendo o original
        df.to_csv(raw_path, index=False)
        print(f"Arquivo final 'daily_births.csv' formatado e salvo.")
        
        return str(raw_path)
    except Exception as e:
        raise ConnectionError(f"Falha ao baixar ou processar o arquivo 'daily_births.csv'. Erro: {e}")


def process_data_fixed_origin(dataset_name, raw_path, processed_dir, forecast_horizon):
    """Cria e salva um único split de treino/teste (origem fixa)."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(raw_path, parse_dates=['ds'])
    
    # Remove linhas com valores ausentes que podem existir em alguns datasets
    df.dropna(inplace=True)
    
    train_df = df.iloc[:-forecast_horizon]
    test_df = df.iloc[-forecast_horizon:]
    
    train_path = processed_path / "train.csv"
    test_path = processed_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Dados divididos: Treino ({len(train_df)} obs), Teste ({len(test_df)} obs)")
    return {"train_path": str(train_path), "test_path": str(test_path)}


def load_processed_data(processed_paths: dict):
    """Carrega os arquivos de treino e teste em DataFrames pandas."""
    train_df = pd.read_csv(processed_paths["train_path"], parse_dates=['ds'])
    test_df = pd.read_csv(processed_paths["test_path"], parse_dates=['ds'])
    return {"train_df": train_df, "test_df": test_df}