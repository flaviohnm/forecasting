import os
import logging
import pandas as pd
import joblib

from src.data_management.data_loader import load_dataset
from src.models.statistical import train_stats_model
from src.models.deep_learning import train_dl_model

def run(model_conf, dataset_conf, main_config, exec_name):
    """
    Orquestra o treinamento de um único modelo (Base ou Híbrido).
    """
    logging.info(f"--- [TRAIN] Iniciando Pipeline: {exec_name} ---")

    # 1. Configuração de Caminhos
    save_path = main_config['results_paths']['saved_models']
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{exec_name}.joblib")

    if os.path.exists(file_path):
        logging.info(f"Modelo {exec_name} já existe em disco. Pulando treino.")
        return

    # 2. Carregamento de Dados
    df = load_dataset(dataset_conf)
    
    # 3. Cálculo Dinâmico do Input Size
    horizon = dataset_conf['forecast_horizon']
    if 'input_size_multiplier' in model_conf:
        multiplier = model_conf['input_size_multiplier']
        calculated_input = int(horizon * multiplier)
        
        if 'params' not in model_conf: model_conf['params'] = {}
        if 'model_kwargs' not in model_conf['params']: model_conf['params']['model_kwargs'] = {}
        
        model_conf['params']['model_kwargs']['input_size'] = calculated_input
        logging.info(f"Input Size Dinâmico calculado: {calculated_input} (H={horizon} x {multiplier})")

    # 4. Lógica Híbrida: Cálculo de Resíduos
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

    # 5. Despacho para Treinamento
    model_type = model_conf.get('model_type', '')
    group = model_conf.get('comparison_group', '')
    
    try:
        if 'statistical' in group or model_type in ['ARIMA', 'ETS', 'NAIVE', 'SEASONAL_NAIVE']:
            model_object = train_stats_model(df, model_conf, horizon, dataset_conf['freq'])
        else:
            # Deep Learning (NHiTS, NBEATS)
            val_size = dataset_conf.get('val_size', horizon)
            model_object = train_dl_model(df, model_conf, horizon, dataset_conf['freq'], val_size)
            
        # 6. Salvar Modelo
        logging.info(f"Salvando modelo em: {file_path}")
        joblib.dump(model_object, file_path)

    except Exception as e:
        logging.error(f"Erro fatal durante o treinamento de {exec_name}: {e}")
        raise e

def calculate_residuals(base_name, df_original, save_path, horizon):
    """
    Carrega o modelo base e subtrai suas previsões do valor real.
    """
    model_path = os.path.join(save_path, f"{base_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo base não encontrado: {model_path}")

    model = joblib.load(model_path)
    
    from src.models.utils import get_fitted_values
    
    # Recupera valores ajustados
    df_fitted = get_fitted_values(model, df_original, horizon=horizon)
    
    # --- FIX CRÍTICO: Evitar duplicidade de colunas no merge ---
    # O df_fitted traz uma coluna 'y'. O df_original também.
    # Se fizermos o merge direto, o pandas cria 'y_x' e 'y_y', e a coluna 'y' some.
    # Removemos o 'y' do fitted para manter apenas o do original.
    if 'y' in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=['y'])

    # Merge para alinhar datas (Agora seguro)
    df_merged = pd.merge(df_original, df_fitted, on=['ds', 'unique_id'], how='inner')
    
    # Cálculo do Resíduo: Real - Previsto
    # Agora temos certeza que 'y' e 'y_hat' existem
    df_merged['y'] = df_merged['y'] - df_merged['y_hat']
    
    return df_merged[['unique_id', 'ds', 'y']]