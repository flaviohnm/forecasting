# File: src/pipelines/train_pipeline.py

import os
import pandas as pd
import joblib

from src.data_management import preprocessing
from src.models import arima_model, deep_learning_model, ets_model

def run(main_config: dict, model_conf: dict, dataset_conf: dict, execution_name: str):
    """
    Executa o pipeline de treinamento para uma combinação de modelo e dataset.
    - model_conf: Configuração do passo do modelo (da Estratégia).
    - dataset_conf: Configuração do dataset.
    - execution_name: Nome único para salvar os artefatos (ex: airline_base_arima).
    """
    model_type = model_conf['model_type']
    models_path = main_config['models_path']
    
    # Carrega os dados, usando apenas a parte de treino
    train_series, _ = preprocessing.load_and_prepare_data(main_config, dataset_conf)

    # --- Lógica de Treinamento por Tipo de Modelo ---
    
    if model_type == 'ARIMA':
        model_path = os.path.join(models_path, f"{execution_name}_arima.joblib")
        # Prepara os parâmetros do arima, incluindo os sazonais do dataset
        arima_params = model_conf['arima_params'].copy()
        arima_params['seasonal'] = dataset_conf['seasonal_period'] > 1
        arima_params['m'] = dataset_conf['seasonal_period']
        arima_model.train_and_save_arima(train_series, model_path, arima_params)

    elif model_type == 'ETS':
        model_path = os.path.join(models_path, f"{execution_name}_ets.pkl")
        ets_model.train_and_save_ets(
            train_series, 
            model_path, 
            model_conf['ets_params'],
            dataset_conf['seasonal_period']
        )
        
    elif model_type == 'Hybrid':
        dependency_model_name = model_conf.get('depends_on')
        if not dependency_model_name:
            raise ValueError(f"Experimento Híbrido '{execution_name}' не define 'depends_on'.")

        # Carrega o modelo ARIMA pré-treinado
        dependency_execution_name = f"{dataset_conf['name']}_{dependency_model_name}"
        arima_model_path = os.path.join(models_path, f"{dependency_execution_name}_arima.joblib")
        
        print(f"Carregando modelo ARIMA dependente de: {arima_model_path}")
        if not os.path.exists(arima_model_path):
            raise FileNotFoundError(f"Modelo ARIMA dependente não encontrado. Execute o treino de '{dependency_execution_name}' primeiro.")
        
        arima_instance = joblib.load(arima_model_path)
        
        # Calcula os resíduos do treino para treinar os modelos N-BEATS
        linear_preds_in_sample = arima_instance.predict_in_sample()
        residuals_train = train_series - pd.Series(linear_preds_in_sample, index=train_series.index)
        print("Resíduos do treino calculados a partir do modelo ARIMA pré-treinado.")

        horizon = dataset_conf['forecast_horizon']
        input_lags = model_conf['nbeats_params']['input_lags']
        residual_datasets = preprocessing.create_direct_forecast_datasets(residuals_train, input_lags, horizon)

        # Treina um modelo N-BEATS para cada passo do horizonte
        for h in range(1, horizon + 1):
            X_res, y_res = residual_datasets[h]
            model_path = os.path.join(models_path, f"{execution_name}_h{h}.keras")
            deep_learning_model.train_and_save_nbeats(X_res, y_res, model_path, model_conf['nbeats_params'])

    else:
        print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de treino definida. Pulando.")