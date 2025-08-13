# /forecasting/src/data_processing.py (VERSÃO HÍBRIDA E ROBUSTA)
import pandas as pd
from pathlib import Path
import urllib.request

def fetch_airline_data(data_dir="data"):
    """
    Verifica se o dataset 'airline.csv' existe localmente.
    Se não existir, baixa de uma URL estável e o formata corretamente.
    """
    raw_path = Path(data_dir) / "raw" / "airline.csv"

    # 1. Tenta usar o arquivo local primeiro
    if raw_path.exists():
        print(f"Utilizando o dataset local encontrado em: {raw_path}")
        return str(raw_path)

    # 2. Se o arquivo local não for encontrado, baixa da internet
    print(f"Arquivo local não encontrado. Tentando baixar de uma fonte alternativa...")

    # URL estável para o dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

    # Garante que o diretório 'data/raw' exista
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Realiza o download
        urllib.request.urlretrieve(url, raw_path)
        print(f"Download concluído com sucesso! Arquivo salvo em: {raw_path}")

        # 3. Formata o arquivo baixado para o padrão do projeto
        # O arquivo baixado tem colunas "Month" e "Passengers"
        df = pd.read_csv(raw_path)

        # Renomeia as colunas para o padrão esperado ('ds', 'passengers')
        df.rename(columns={"Month": "ds", "Passengers": "passengers"}, inplace=True)
        
        # Garante que a coluna 'ds' seja do tipo datetime
        df['ds'] = pd.to_datetime(df['ds'])

        # Salva o arquivo já formatado, sobrescrevendo o original
        df.to_csv(raw_path, index=False)
        print("Arquivo formatado e salvo corretamente.")

        return str(raw_path)

    except Exception as e:
        raise ConnectionError(
            f"Falha ao baixar o arquivo de {url}. Erro: {e}\n"
            "Verifique sua conexão com a internet ou se a URL ainda é válida."
        )

def process_data_fixed_origin(dataset_name, raw_path, processed_dir, forecast_horizon):
    """Cria e salva um único split de treino/teste (origem fixa)."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path, parse_dates=['ds'])
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