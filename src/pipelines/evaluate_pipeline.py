import os
import logging
import joblib
import pandas as pd
import numpy as np
from src.data_management.data_loader import load_dataset
from src.models.utils import predict_wrapper
from src.analysis.diagnostics import save_residual_diagnostics
# --- NOVO IMPORT ---
from src.data_management.preprocessor import TimeSeriesScaler

def run(main_config, model_conf, dataset_conf, exec_name):
    """
    Avalia o modelo com suporte a Normalização (Scaling) e Lógica Híbrida.
    """
    logging.info(f"--- [EVAL] Iniciando Avaliação: {exec_name} ---")

    # 1. Setup de Caminhos
    models_path = main_config['results_paths']['saved_models']
    metrics_path = main_config['results_paths']['metrics']
    forecasts_path = main_config['results_paths']['forecasts']
    
    # Caminho do Scaler
    scaler_path = os.path.join(models_path, f"{exec_name}_scaler.joblib")
    
    # Pasta Diagnostics
    diag_path = main_config['results_paths'].get('diagnostics', 'results/diagnostics/')
    
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(forecasts_path, exist_ok=True)
    os.makedirs(diag_path, exist_ok=True)

    # 2. Carregar Dados Reais (Ground Truth)
    df_full = load_dataset(dataset_conf)
    
    # Definir conjunto de TESTE
    test_size = dataset_conf.get('test_size', 720)
    horizon = dataset_conf['forecast_horizon']
    
    # Corte Temporal (Holdout)
    train_end_index = len(df_full) - test_size
    
    # df_history: O contexto que o modelo vê
    df_history = df_full.iloc[:train_end_index].copy()
    
    # df_test: O gabarito (Futuro)
    df_test = df_full.iloc[train_end_index : train_end_index + horizon].copy()
    
    # 3. Carregar Modelo e Scaler
    model_path = os.path.join(models_path, f"{exec_name}.joblib")
    if not os.path.exists(model_path):
        logging.error(f"Modelo não encontrado para avaliação: {model_path}")
        return
    
    model = joblib.load(model_path)
    
    # Tenta carregar o Scaler
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info("Scaler carregado. Aplicando normalização no histórico.")
    
    # --- APLICAÇÃO DO SCALER NO HISTÓRICO ---
    # Se o modelo foi treinado com dados escalados, precisamos escalar o input
    if scaler:
        df_history_scaled = scaler.transform(df_history)
    else:
        df_history_scaled = df_history

    # 4. Previsão
    try:
        # O modelo gera previsões na escala que foi treinado (Normalizada ou Real)
        y_hat_df = predict_wrapper(model, horizon, df_history=df_history_scaled)
    except Exception as e:
        logging.error(f"Erro na previsão de {exec_name}: {e}")
        return

    # Garante estrutura
    y_hat_df = y_hat_df.sort_values('ds').reset_index(drop=True)
    
    # --- INVERSÃO DO SCALER NA PREVISÃO ---
    if scaler:
        # Traz a previsão de volta para a escala real (ex: de 0.5 para 3000 MW)
        # O inverse_transform procura pela coluna 'y_hat' e a converte
        y_hat_df = scaler.inverse_transform(y_hat_df)

    # 5. Alinhamento com o Teste
    df_test = df_test.sort_values('ds').reset_index(drop=True)
    
    if len(y_hat_df) == len(df_test):
        df_test['y_hat'] = y_hat_df['y_hat'].values
    else:
        df_test = pd.merge(df_test, y_hat_df[['ds', 'y_hat']], on='ds', how='left')

    # 6. Lógica Híbrida (Reconstrução do Sinal)
    # Se este modelo previu resíduos, precisamos somar à previsão da base
    if model_conf.get('depends_on'):
        base_name = model_conf['depends_on']
        base_exec_name = f"{dataset_conf['name']}_{base_name}_h{horizon}"
        logging.info(f"Reconstruindo sinal híbrido. Base: {base_exec_name}")
        
        base_forecast_path = os.path.join(forecasts_path, f"forecast_{base_exec_name}.csv")
        
        if os.path.exists(base_forecast_path):
            df_base = pd.read_csv(base_forecast_path)
            df_base['ds'] = pd.to_datetime(df_base['ds'])
            df_base = df_base.sort_values('ds').reset_index(drop=True)
            
            if len(df_test) == len(df_base):
                 # Base já está em escala real (foi processada pelo evaluate da base)
                 df_test['y_hat_base'] = df_base['y_hat'].values
                 
                 # O y_hat atual é o resíduo (já desnormalizado pelo inverse_transform acima)
                 df_test['y_hat_resid'] = df_test['y_hat'].values 
                 
                 # Soma Final
                 df_test['y_hat'] = df_test['y_hat_base'] + df_test['y_hat_resid']
                 logging.info("Sinal reconstruído: Final = Base (Real) + Resíduo (Desnormalizado)")
            else:
                 logging.warning("Tamanhos incompatíveis entre Base e Híbrido.")
        else:
            logging.warning(f"Base {base_forecast_path} não encontrada.")

    # 7. Métricas e Salvamento
    df_test = df_test.dropna(subset=['y', 'y_hat'])
    
    if len(df_test) == 0:
        logging.error(f"Dataframe vazio para {exec_name}.")
        return

    mape = calculate_mape(df_test['y'], df_test['y_hat'])
    mase = calculate_mase(df_test['y'], df_test['y_hat'], df_full.iloc[:-test_size]['y'], dataset_conf.get('seasonal_period', 1))
    
    logging.info(f"Resultados {exec_name} -> MAPE: {mape:.4f} | MASE: {mase:.4f}")

    # Salvar CSVs
    metrics_df = pd.DataFrame([{
        'model': exec_name,
        'dataset': dataset_conf['name'],
        'horizon': horizon,
        'mape': mape,
        'mase': mase
    }])
    metrics_df.to_csv(os.path.join(metrics_path, f"metrics_{exec_name}.csv"), index=False)
    df_test[['unique_id', 'ds', 'y', 'y_hat']].to_csv(os.path.join(forecasts_path, f"forecast_{exec_name}.csv"), index=False)

    # Diagnóstico Visual
    save_residual_diagnostics(
        y_true=df_test['y'],
        y_pred=df_test['y_hat'],
        model_name=exec_name,
        dataset_name=dataset_conf['name'],
        horizon=horizon,
        save_path=diag_path
    )

# --- Helpers ---
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mase(y_true, y_pred, y_train, seasonality):
    y_train = np.array(y_train)
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_train) <= seasonality: return np.nan
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    return mae / scale if scale != 0 else np.inf