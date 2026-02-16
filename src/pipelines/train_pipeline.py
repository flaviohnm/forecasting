import os
import logging
import pandas as pd
import joblib

from src.data_management.data_loader import load_dataset
from src.models.statistical import train_stats_model
from src.models.deep_learning import train_dl_model
# --- NOVO IMPORT ---
from src.data_management.preprocessor import TimeSeriesScaler

def run(model_conf, dataset_conf, main_config, exec_name):
    """
    Orquestra o treinamento de um único modelo (Base ou Híbrido) com Normalização.
    """
    logging.info(f"--- [TRAIN] Iniciando Pipeline: {exec_name} ---")

    # 1. Configuração de Caminhos
    save_path = main_config['results_paths']['saved_models']
    os.makedirs(save_path, exist_ok=True)
    
    file_path = os.path.join(save_path, f"{exec_name}.joblib")
    scaler_path = os.path.join(save_path, f"{exec_name}_scaler.joblib") # Caminho do Scaler

    # Verifica se Modelo E Scaler já existem
    if os.path.exists(file_path) and os.path.exists(scaler_path):
        logging.info(f"Modelo e Scaler {exec_name} já existem em disco. Pulando treino.")
        return

    # 2. Carregamento de Dados
    df = load_dataset(dataset_conf)
    
    # 3. Cálculo Dinâmico do Input Size (Mantido sua lógica original)
    horizon = dataset_conf['forecast_horizon']
    if 'input_size_multiplier' in model_conf:
        multiplier = model_conf['input_size_multiplier']
        calculated_input = int(horizon * multiplier)
        
        if 'params' not in model_conf: model_conf['params'] = {}
        if 'model_kwargs' not in model_conf['params']: model_conf['params']['model_kwargs'] = {}
        
        model_conf['params']['model_kwargs']['input_size'] = calculated_input
        logging.info(f"Input Size Dinâmico calculado: {calculated_input} (H={horizon} x {multiplier})")

    # 4. Lógica Híbrida: Cálculo de Resíduos (Mantido sua lógica original)
    if model_conf.get('depends_on'):
        base_model_name = f"{dataset_conf['name']}_{model_conf['depends_on']}_h{horizon}"
        logging.info(f"Modo Híbrido detectado. Calculando resíduos sobre: {base_model_name}")
        
        try:
            # Carrega a base e faz a previsão IN-SAMPLE
            df_residuals = calculate_residuals(base_model_name, df, save_path, horizon)
            
            # Substitui o target original 'y' pelos resíduos
            df = df_residuals
        except Exception as e:
            logging.error(f"Falha ao calcular resíduos da base {base_model_name}: {e}")
            raise e

    # --- 4.5 APLICAÇÃO DO SCALER (NOVO BLOCO) ---
    # Normaliza os dados (seja dado bruto ou resíduo híbrido) antes de treinar
    logging.info("Aplicando normalização (StandardScaler)...")
    scaler = TimeSeriesScaler()
    
    # O Scaler aprende e transforma o dataframe atual
    df_scaled = scaler.fit_transform(df)

    # 5. Despacho para Treinamento (Usando df_scaled)
    model_type = model_conf.get('model_type', '')
    group = model_conf.get('comparison_group', '')
    
    try:
        if 'statistical' in group or model_type in ['ARIMA', 'ETS', 'NAIVE', 'SEASONAL_NAIVE']:
            # Passamos o df_scaled
            model_object = train_stats_model(df_scaled, model_conf, horizon, dataset_conf['freq'])
        else:
            # Deep Learning (NHiTS, NBEATS)
            val_size = dataset_conf.get('val_size', horizon)
            # Passamos o df_scaled
            model_object = train_dl_model(df_scaled, model_conf, horizon, dataset_conf['freq'], val_size)
            
        # 6. Salvar Modelo E Scaler
        logging.info(f"Salvando modelo em: {file_path}")
        joblib.dump(model_object, file_path)
        
        logging.info(f"Salvando scaler em: {scaler_path}")
        joblib.dump(scaler, scaler_path)

    except Exception as e:
        logging.error(f"Erro fatal durante o treinamento de {exec_name}: {e}")
        raise e

def calculate_residuals(base_name, df_original, save_path, horizon):
    """
    Carrega o modelo base e subtrai suas previsões do valor real.
    """
    model_path = os.path.join(save_path, f"{base_name}.joblib")
    scaler_path = os.path.join(save_path, f"{base_name}_scaler.joblib") # Carrega o scaler da base também

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo base não encontrado: {model_path}")

    model = joblib.load(model_path)
    
    # Tenta carregar o scaler da base (se existir)
    # Isso é crucial: Para calcular o resíduo correto, precisamos inverter a previsão da base
    # para a escala real antes de subtrair do df_original (que está em escala real).
    base_scaler = None
    if os.path.exists(scaler_path):
        base_scaler = joblib.load(scaler_path)

    from src.models.utils import get_fitted_values
    
    # Recupera valores ajustados ( fitted values costumam vir na escala do modelo)
    # Precisamos tratar isso com cuidado no get_fitted_values ou aqui.
    # Assumindo que o get_fitted_values retorna o output cru do modelo (escalado):
    df_fitted = get_fitted_values(model, df_original, horizon=horizon)
    
    # Se tínhamos um scaler na base, revertemos a previsão para a escala real
    if base_scaler:
        # Precisamos da classe TimeSeriesScaler importada aqui ou garantida no topo
        # O método inverse_transform aceita dataframe com colunas 'y'/'y_hat'
        df_fitted = base_scaler.inverse_transform(df_fitted)

    # --- Lógica de Merge Original ---
    if 'y' in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=['y'])

    # Merge para alinhar datas
    df_merged = pd.merge(df_original, df_fitted, on=['ds', 'unique_id'], how='inner')
    
    # Cálculo do Resíduo: Real - Previsto
    df_merged['y'] = df_merged['y'] - df_merged['y_hat']
    
    return df_merged[['unique_id', 'ds', 'y']]