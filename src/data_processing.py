# /forecasting/src/data_processing.py (COM NOME DE FUNÇÃO CORRIGIDO)
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

# --- MUDANÇA AQUI: Renomeando a função ---
def fetch_airline_data(data_dir="data"):
    """Baixa o dataset 'airlines' do OpenML e o salva localmente."""
    raw_path = Path(data_dir) / "raw" / "airline.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print("Baixando dados do 'airlines' do OpenML...")
        airline_data = fetch_openml(name="airlines", version=1, as_frame=True, parser='auto')
        df = airline_data.frame
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=1)
        df.rename(columns={'time': 'ds', 'value': 'passengers'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df.to_csv(raw_path, index=False)
    
    return str(raw_path)

def process_data_fixed_origin(dataset_name, raw_path, processed_dir, forecast_horizon):
    """Cria e salva um único split de treino/teste (origem fixa)."""
    print(f"Criando split de origem fixa para o dataset '{dataset_name}'...")
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