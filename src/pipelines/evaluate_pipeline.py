import os
import logging
import joblib
import pandas as pd
import numpy as np
from src.data_management.data_loader import load_dataset
from src.models.utils import predict_wrapper

# --- [NOVO] Importação da ferramenta de diagnóstico ---
from src.analysis.diagnostics import save_residual_diagnostics

def run(main_config, model_conf, dataset_conf, exec_name):
    """
    Avalia o modelo treinado nos dados de teste e salva métricas.
    """
    logging.info(f"--- [EVAL] Iniciando Avaliação: {exec_name} ---")

    # 1. Setup de Caminhos (Agora lendo do seu config ajustado)
    models_path = main_config['results_paths']['saved_models']
    metrics_path = main_config['results_paths']['metrics']
    forecasts_path = main_config['results_paths']['forecasts']
    
    # Usa a chave 'diagnostics' que você criou. Se não existir, cria um padrão.
    diag_path = main_config['results_paths'].get('diagnostics', 'results/diagnostics/')
    
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(forecasts_path, exist_ok=True)
    os.makedirs(diag_path, exist_ok=True) # Garante que a pasta existe

    # 2. Carregar Dados Reais (Ground Truth)
    df_full = load_dataset(dataset_conf)
    
    # Definir conjunto de TESTE (Onde a mágica acontece)
    test_size = dataset_conf.get('test_size', 720)
    horizon = dataset_conf['forecast_horizon']
    
    # Pegamos os primeiros 'horizon' pontos do conjunto de teste para comparação
    train_end_index = len(df_full) - test_size
    df_test = df_full.iloc[train_end_index : train_end_index + horizon].copy()
    
    # 3. Carregar Modelo Principal e Prever
    model_path = os.path.join(models_path, f"{exec_name}.joblib")
    if not os.path.exists(model_path):
        # Mudei para logging.error para não travar o loop se um falhar
        logging.error(f"Modelo não encontrado para avaliação: {model_path}")
        return
    
    model = joblib.load(model_path)
    
    # Gera previsão (y_hat)
    try:
        y_hat_df = predict_wrapper(model, horizon)
    except Exception as e:
        logging.error(f"Erro na previsão de {exec_name}: {e}")
        return
    
    # Garante que as datas batem (reset index para merge seguro)
    y_hat_df = y_hat_df.sort_values('ds').reset_index(drop=True)
    df_test = df_test.sort_values('ds').reset_index(drop=True)
    
    # Alinha as previsões com o Real
    if len(y_hat_df) == len(df_test):
        df_test['y_hat'] = y_hat_df['y_hat'].values
    else:
        # Tenta merge por data
        df_test = pd.merge(df_test, y_hat_df[['ds', 'y_hat']], on='ds', how='left')

    # 4. Lógica Híbrida (Soma: Base + Resíduo) - SUA LÓGICA ORIGINAL
    if model_conf.get('depends_on'):
        base_name = model_conf['depends_on']
        base_exec_name = f"{dataset_conf['name']}_{base_name}_h{horizon}"
        logging.info(f"Reconstruindo sinal híbrido. Base: {base_exec_name}")
        
        base_forecast_path = os.path.join(forecasts_path, f"forecast_{base_exec_name}.csv")
        
        if os.path.exists(base_forecast_path):
            df_base = pd.read_csv(base_forecast_path)
            df_base['ds'] = pd.to_datetime(df_base['ds'])
            df_base = df_base.sort_values('ds').reset_index(drop=True)
            
            # Merge seguro para Híbridos
            if len(df_test) == len(df_base):
                 df_test['y_hat_base'] = df_base['y_hat'].values
                 df_test['y_hat_resid'] = df_test['y_hat'].values
                 df_test['y_hat'] = df_test['y_hat_base'] + df_test['y_hat_resid']
                 logging.info("Sinal reconstruído: Final = Base + Resíduo Estimado")
            else:
                 logging.warning("Tamanhos incompatíveis entre Base e Híbrido. Merge abortado.")
        else:
            logging.warning(f"Arquivo de previsão da base {base_forecast_path} não encontrado. Usando apenas resíduo (ERRADO).")

    # 5. Cálculo de Métricas
    # Remover NaNs antes do cálculo
    df_test = df_test.dropna(subset=['y', 'y_hat'])
    
    if len(df_test) == 0:
        logging.error(f"Dataframe vazio para {exec_name}. Pulando métricas.")
        return

    mape = calculate_mape(df_test['y'], df_test['y_hat'])
    mase = calculate_mase(df_test['y'], df_test['y_hat'], df_full.iloc[:-test_size]['y'], dataset_conf.get('seasonal_period', 1))
    
    logging.info(f"Resultados {exec_name} -> MAPE: {mape:.4f} | MASE: {mase:.4f}")

    # 6. Salvar Resultados
    metrics_df = pd.DataFrame([{
        'model': exec_name,
        'dataset': dataset_conf['name'],
        'horizon': horizon,
        'mape': mape,
        'mase': mase
    }])
    metrics_df.to_csv(os.path.join(metrics_path, f"metrics_{exec_name}.csv"), index=False)
    
    df_test[['unique_id', 'ds', 'y', 'y_hat']].to_csv(os.path.join(forecasts_path, f"forecast_{exec_name}.csv"), index=False)

    # --- [NOVO] Executar Diagnóstico ACF/PACF ---
    # Usa o caminho 'diag_path' que veio do seu config
    save_residual_diagnostics(
        y_true=df_test['y'],
        y_pred=df_test['y_hat'],
        model_name=exec_name,
        dataset_name=dataset_conf['name'],
        horizon=horizon,
        save_path=diag_path
    )

# --- Funções Auxiliares (Sua versão original) ---

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mase(y_true, y_pred, y_train, seasonality):
    """Mean Absolute Scaled Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_train = np.array(y_train)
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    if len(y_train) <= seasonality:
        return np.nan

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    
    return mae / scale if scale != 0 else np.inf