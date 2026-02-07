import logging
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

# Mapeamento seguro para evitar erros de importação se a versão mudar
try:
    from statsforecast.models import ETS
except ImportError:
    ETS = None 

def train_stats_model(df, model_conf, horizon, freq):
    """
    Treina modelos estatísticos (ARIMA, ETS, Naive).
    Retorna o objeto StatsForecast ajustado.
    """
    model_type = model_conf['model_type']
    params = model_conf.get('params', {})
    
    # --- FIX CRÍTICO: Sanitização do DataFrame ---
    # Garante que unique_id é uma coluna e não o índice
    df = df.copy()
    if 'unique_id' not in df.columns:
        if df.index.name == 'unique_id':
            df = df.reset_index()
        else:
            raise ValueError(f"DataFrame recebido para treino do {model_type} não possui coluna 'unique_id'. Colunas: {df.columns.tolist()}")
    
    # Garante que ds é datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 1. Seleção do Modelo
    if model_type == 'ARIMA':
        model = AutoARIMA(
            season_length=params.get('m', 24),
            d=params.get('d', None),
            D=params.get('D', None),
            max_p=params.get('max_p', 5),
            max_q=params.get('max_q', 5),
            trace=params.get('trace', False)
        )
        
    elif model_type == 'ETS':
        # AutoETS é a implementação moderna no StatsForecast >= 1.0
        model = AutoETS(
            season_length=params.get('m', 24),
            model=params.get('model', 'ZZZ')
        )
        
    elif model_type == 'SEASONAL_NAIVE':
        model = SeasonalNaive(season_length=params.get('m', 24))
        
    elif model_type == 'NAIVE':
        model = Naive()
        
    else:
        raise ValueError(f"Modelo estatístico desconhecido: {model_type}")

    logging.info(f"Treinando {model_type} com horizonte {horizon}...")

    # 2. Instanciação do Wrapper StatsForecast
    # fallback_model: garante que se o ARIMA falhar em convergir, use Naive em vez de quebrar
    sf = StatsForecast(
        models=[model], 
        freq=freq, 
        n_jobs=1,
        fallback_model=Naive() 
    )

    # 3. Treinamento (Fit)
    try:
        sf.fit(df)
    except Exception as e:
        logging.error(f"Erro interno no .fit() do StatsForecast para {model_type}. Columns: {df.columns}. Dtypes: {df.dtypes}")
        raise e
    
    return sf