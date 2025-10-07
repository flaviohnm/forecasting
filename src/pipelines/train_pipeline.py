# File: src/pipelines/train_pipeline.py

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data_management import downloader, preprocessing
from src.models import arima_model, deep_learning_model, ets_model


def run(main_config: dict, model_conf: dict, dataset_conf: dict,
        execution_name: str):
    """
    Executa o pipeline de treinamento para uma combinação de modelo e dataset.
    """
    model_type = model_conf['model_type']
    models_path = main_config['models_path']

    train_series, _ = preprocessing.load_and_prepare_data(
        main_config, dataset_conf)

    if model_type == 'ARIMA':
        model_path = os.path.join(models_path, f"{execution_name}.joblib")
        arima_params = model_conf['arima_params'].copy()
        arima_params['seasonal'] = dataset_conf['seasonal_period'] > 1
        arima_params['m'] = dataset_conf['seasonal_period']
        arima_model.train_and_save_arima(train_series, model_path,
                                         arima_params)

    elif model_type == 'ETS':
        model_path = os.path.join(models_path, f"{execution_name}.pkl")
        ets_model.train_and_save_ets(train_series, model_path,
                                     model_conf['ets_params'],
                                     dataset_conf['seasonal_period'])

    elif model_type == 'NAIVE':
        print("INFO: Modelo NAIVE não requer treinamento.")
        pass

    elif model_type == 'MLP' or model_type == 'LSTM':
        print(f"[DEBUG] Entrou no bloco de treino {model_type} Standalone.")
        params_key = 'mlp_params' if model_type == 'MLP' else 'lstm_params'
        model_builder = deep_learning_model.build_mlp_model if model_type == 'MLP' else deep_learning_model.build_lstm_model

        # --- LÓGICA DE ESCALONAMENTO ADICIONADA ---
        scaler_type = dataset_conf.get('preprocessing', {}).get('scaler')
        scaled_train_series = train_series.copy()

        if scaler_type:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Tipo de scaler desconhecido: {scaler_type}")

            scaled_values = scaler.fit_transform(
                train_series.values.reshape(-1, 1)).flatten()
            scaled_train_series = pd.Series(scaled_values,
                                            index=train_series.index)

            scaler_path = os.path.join(models_path,
                                       f"{execution_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            print(
                f"[INFO] Scaler '{scaler_type}' da série salvo em: {scaler_path}"
            )
        else:
            print(
                "[INFO] Nenhum scaler especificado para modelo Keras. Treinando com dados originais."
            )

        horizon = dataset_conf['forecast_horizon']
        input_lags = model_conf[params_key]['input_lags']
        # Usa a série escalada para criar os datasets
        datasets = preprocessing.create_direct_forecast_datasets(
            scaled_train_series, input_lags, horizon)

        for h in range(1, horizon + 1):
            X_train, y_train = datasets[h]
            model_path = os.path.join(models_path,
                                      f"{execution_name}_h{h}.keras")
            deep_learning_model.train_and_save_keras_model(
                X_train,
                y_train,
                model_path,
                model_conf[params_key],
                model_builder=model_builder,
                output_shape=1)

    elif model_type in [
            'iTransformer', 'NHiTS', 'Hybrid_MIMO_NHITS',
            'Hybrid_Direct_NHITS', 'Hybrid_MIMO_NBEATS_NF',
            'Hybrid_Direct_NBEATS_NF'
    ]:
        print(
            f"INFO: Modelo {model_type} (baseado em NeuralForecast) não requer um passo de treino separado. O treino ocorrerá durante a avaliação."
        )
        pass

    elif model_type in [
            'Hybrid_MLP_Recursive', 'Hybrid_LSTM_Recursive', 'Hybrid_MLP_MIMO'
    ]:
        dependency_name = model_conf.get('depends_on')
        arima_model_path = os.path.join(
            models_path, f"{dataset_conf['name']}_{dependency_name}.joblib")
        if not os.path.exists(arima_model_path):
            raise FileNotFoundError(
                f"Modelo ARIMA dependente não encontrado: '{arima_model_path}'"
            )

        arima_instance = joblib.load(arima_model_path)
        residuals_train = train_series - pd.Series(
            arima_instance.predict_in_sample(), index=train_series.index)

        scaler_type = dataset_conf.get('preprocessing', {}).get('scaler')
        scaled_residuals_train = residuals_train.copy()

        if scaler_type:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Tipo de scaler desconhecido: {scaler_type}")

            scaled_residuals_train_values = scaler.fit_transform(
                residuals_train.values.reshape(-1, 1)).flatten()
            scaled_residuals_train = pd.Series(scaled_residuals_train_values,
                                               index=residuals_train.index)

            scaler_path = os.path.join(models_path,
                                       f"{execution_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            print(
                f"[INFO] Scaler '{scaler_type}' dos resíduos salvo em: {scaler_path}"
            )
        else:
            print(
                "[INFO] Nenhum scaler especificado. Treinando com resíduos originais."
            )

        if 'Recursive' in model_type:
            params_key = 'mlp_params' if 'MLP' in model_type else 'lstm_params'
            model_builder = deep_learning_model.build_mlp_model if 'MLP' in model_type else deep_learning_model.build_lstm_model

            input_lags = model_conf[params_key]['input_lags']
            X_res, y_res = preprocessing.create_recursive_forecast_dataset(
                scaled_residuals_train, input_lags)

            model_path = os.path.join(models_path, f"{execution_name}.keras")
            deep_learning_model.train_and_save_keras_model(
                X_res,
                y_res,
                model_path,
                model_conf[params_key],
                model_builder=model_builder,
                output_shape=1)

        elif model_type == 'Hybrid_MLP_MIMO':
            horizon = dataset_conf['forecast_horizon']
            input_lags = model_conf['mlp_params']['input_lags']
            X_res, y_res = preprocessing.create_mimo_forecast_dataset(
                scaled_residuals_train, input_lags, horizon)

            model_path = os.path.join(models_path, f"{execution_name}.keras")
            deep_learning_model.train_and_save_keras_model(
                X_res,
                y_res,
                model_path,
                model_conf['mlp_params'],
                model_builder=deep_learning_model.build_mlp_model,
                output_shape=horizon)
    else:
        print(
            f"AVISO: Tipo de modelo '{model_type}' não possui lógica de treino definida."
        )
