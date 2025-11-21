# File: src/pipelines/train_pipeline.py

import os
import pandas as pd
import joblib
import numpy as np
import logging
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import traceback 

from src.data_management import preprocessing
from src.models import arima_model, deep_learning_model, ets_model

# --- Funções Helper (sem underline) ---
def get_suggestion(trial: optuna.Trial, name: str, config: list):
    """Traduz uma configuração do YAML para uma sugestão do Optuna."""
    param_type = config[0]
    if param_type == 'int':
        return trial.suggest_int(name, low=config[1], high=config[2])
    if param_type == 'float':
        try:
            low_val = float(config[1])
            high_val = float(config[2])
        except ValueError as e:
            logging.error(f"ERRO DE CONFIGURAÇÃO: Não foi possível converter '{config[1]}' ou '{config[2]}' para float no 'hpo_space' do YAML.")
            raise e
        if len(config) == 4 and config[3] == 'log':
            return trial.suggest_float(name, low=low_val, high=high_val, log=True)
        else:
            return trial.suggest_float(name, low=low_val, high=high_val)
    if param_type == 'categorical':
        return trial.suggest_categorical(name, choices=config[1])
    raise ValueError(f"Tipo de HPO desconhecido: {param_type}")

def get_scaler(scaler_type: str):
    """Retorna um objeto scaler com base no nome."""
    if scaler_type == 'standard':
        return StandardScaler()
    if scaler_type == 'minmax':
        return MinMaxScaler()
    return None

def scale_series(series: pd.Series, scaler):
    """Aplica o fit_transform em uma série e a retorna."""
    if scaler is None:
        return series
    # Garante que o scaler seja (re)treinado
    scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(scaled_values, index=series.index, name=series.name)

# --- FUNÇÃO RUN ---
def run(main_config: dict, model_conf: dict, dataset_conf: dict,
        execution_name: str):
    """
    Executa o pipeline de treinamento para uma combinação de modelo e dataset.
    Agora inclui lógica de HPO para modelos Keras.
    """
    model_type = model_conf['model_type']
    models_path = main_config['models_path']

    train_series_full, _ = preprocessing.load_and_prepare_data(
        main_config, dataset_conf)

    # --- LÓGICA PARA MODELOS NÃO-DL ---
    if model_type == 'ARIMA':
        model_path = os.path.join(models_path, f"{execution_name}.joblib")
        arima_params = model_conf['arima_params'].copy()
        arima_params['seasonal'] = dataset_conf.get('seasonal_period', 1) > 1
        arima_params['m'] = dataset_conf.get('seasonal_period', 1)
        arima_params.setdefault('D', None) 
        arima_params.setdefault('seasonal_test', 'ocsb')
        arima_model.train_and_save_arima(train_series_full, model_path, arima_params)
        return

    elif model_type == 'ETS':
        model_path = os.path.join(models_path, f"{execution_name}.pkl")
        ets_model.train_and_save_ets(train_series_full, model_path,
                                     model_conf['ets_params'],
                                     dataset_conf.get('seasonal_period', 1))
        return

    elif model_type == 'NAIVE' or model_type == 'SEASONAL_NAIVE':
        print(f"INFO: Modelo {model_type} não requer treinamento.")
        return

    elif model_type in [
            'iTransformer', 'NHiTS', 'NBEATS_MIMO', 'NBEATS_Direct',
            'Hybrid_MIMO_NHITS', 'Hybrid_Direct_NHITS', 
            'Hybrid_MIMO_NBEATS_NF', 'Hybrid_Direct_NBEATS_NF'
    ]:
        print(f"INFO: O treino do modelo {model_type} (NeuralForecast) ocorrerá durante a avaliação.")
        return
    
    # --- LÓGICA PARA MODELOS DL (KERAS) COM HPO ---
    
    params_key = ""
    model_builder = None
    data_prep_func = None
    
    if model_type in ['MLP_Direct', 'Hybrid_MLP_Direct']:
        params_key = 'mlp_params'
        model_builder = deep_learning_model.build_mlp_model
        data_prep_func = preprocessing.create_direct_forecast_datasets
    elif model_type == 'LSTM':
        params_key = 'lstm_params'
        model_builder = deep_learning_model.build_lstm_model
        data_prep_func = preprocessing.create_direct_forecast_datasets
    elif model_type in ['MLP_MIMO', 'Hybrid_MLP_MIMO']:
        params_key = 'mlp_params'
        model_builder = deep_learning_model.build_mlp_mimo_model
        data_prep_func = preprocessing.create_mimo_forecast_dataset
    elif model_type == 'Hybrid_MLP_Recursive':
        params_key = 'mlp_params'
        model_builder = deep_learning_model.build_mlp_model
        data_prep_func = preprocessing.create_recursive_forecast_dataset
    elif model_type == 'Hybrid_LSTM_Recursive':
        params_key = 'lstm_params'
        model_builder = deep_learning_model.build_lstm_model
        data_prep_func = preprocessing.create_recursive_forecast_dataset
    else:
        print(f"AVISO: Tipo de modelo '{model_type}' não possui lógica de treino definida.")
        return

    print(f"[DEBUG] Entrou no bloco de treino para {model_type}.")
    
    data_to_train = train_series_full.copy()
    if model_type.startswith('Hybrid_'):
        dependency_name = model_conf.get('depends_on')
        arima_model_path = os.path.join(models_path, f"{dataset_conf['name']}_{dependency_name}.joblib")
        if not os.path.exists(arima_model_path):
            raise FileNotFoundError(f"Modelo ARIMA dependente '{arima_model_path}' não encontrado.")
        arima_instance = joblib.load(arima_model_path)
        data_to_train = train_series_full - pd.Series(arima_instance.predict_in_sample(), index=train_series_full.index)

    fixed_params = model_conf.get(params_key, {})
    hpo_space = fixed_params.get('hpo_space')
    
    if not hpo_space:
        raise ValueError(f"Nenhum 'hpo_space' definido para o modelo {execution_name} em model_params.yaml")

    hpo_config = main_config.get('hpo_config', {'n_trials': 20, 'validation_percentage': 0.2})
    n_trials = hpo_config.get('n_trials', 20)
    validation_percentage = hpo_config.get('validation_percentage', 0.2)
    horizon = dataset_conf['forecast_horizon']
    
    validation_size = int(len(data_to_train) * validation_percentage)
    
    if validation_size < horizon:
         validation_size = horizon
         if validation_size > len(data_to_train) * 0.5:
             logging.warning(f"Dataset {execution_name} muito pequeno. Ajustando split de validação.")
             validation_size = int(len(data_to_train) * 0.5)
             if validation_size < 1:
                 raise ValueError("Não há dados suficientes para o split de validação (menos de 2 pontos).")

    split_point = len(data_to_train) - validation_size
    if split_point <= 0:
         raise ValueError(f"Não há dados suficientes para o split de treino HPO (split_point={split_point}). "
                          f"Tamanho dados: {len(data_to_train)}, Tamanho Validação: {validation_size}")

    train_hpo_series = data_to_train.iloc[:split_point]
    val_hpo_series = data_to_train.iloc[split_point:]
    
    print(f"[DEBUG-HPO] Split HPO: Treino={len(train_hpo_series)}, Validação={len(val_hpo_series)}")

    scaler_type = dataset_conf.get('preprocessing', {}).get('scaler')
    scaler = get_scaler(scaler_type)
    
    scaled_train_hpo = scale_series(train_hpo_series, scaler)
    scaled_val_hpo = pd.Series(scaler.transform(val_hpo_series.values.reshape(-1, 1)).flatten(), index=val_hpo_series.index) if scaler else val_hpo_series

    def objective(trial: optuna.Trial):
        try:
            hpo_params = {name: get_suggestion(trial, name, config) for name, config in hpo_space.items()}
            
            n_layers = hpo_params.get('n_layers', 1)
            n_neurons = []
            for i in range(n_layers):
                 neuron_key = f'n_neurons_l{i+1}'
                 if neuron_key in hpo_params:
                     n_neurons.append(hpo_params[neuron_key])
                 elif i == 0:
                     n_neurons.append(32)

            # Parâmetros de construção do modelo
            builder_params = {
                'learning_rate': hpo_params['learning_rate'],
                'n_neurons': n_neurons
            }
            # Parâmetros de ajuste (fit)
            fit_params = {
                'epochs': fixed_params.get('epochs', 100),
                'batch_size': fixed_params.get('batch_size', 16)
            }
            
            input_lags = hpo_params['input_lags']
            
            if input_lags >= len(scaled_train_hpo):
                logging.info(f"  INFO (HPO Trial): Pulando trial. 'input_lags' sugerido ({input_lags}) é >= tamanho do treino HPO ({len(scaled_train_hpo)}).")
                return 1e9
            
            val_horizon = min(horizon, len(val_hpo_series))

            if model_type in ['MLP_Direct', 'LSTM', 'Hybrid_MLP_Direct']:
                val_errors = []
                val_horizon_direct = min(horizon, len(val_hpo_series) - input_lags)
                if val_horizon_direct < 1:
                    logging.info(f"  INFO (HPO Trial): Pulando trial. 'input_lags' ({input_lags}) não deixa dados para validação (val_size={len(val_hpo_series)}).")
                    return 1e9

                datasets_train = preprocessing.create_direct_forecast_datasets(scaled_train_hpo, input_lags, horizon)
                datasets_val = preprocessing.create_direct_forecast_datasets(scaled_val_hpo, input_lags, val_horizon_direct)
                
                for h in range(1, val_horizon_direct + 1):
                    X_train, y_train = datasets_train[h]
                    X_val, y_val = datasets_val[h]

                    if X_val.shape[0] == 0: continue 

                    model = model_builder(input_shape=(input_lags,), output_shape=1, **builder_params)
                    if 'LSTM' in model_type: X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    
                    model.fit(X_train, y_train, **fit_params, verbose=0)
                    
                    if 'LSTM' in model_type: X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                    preds = model.predict(X_val, verbose=0)
                    val_errors.append(mean_squared_error(y_val, preds))
                
                return np.mean(val_errors) if val_errors else 1e9

            elif model_type in ['MLP_MIMO', 'Hybrid_MLP_MIMO']:
                val_horizon_mimo = min(horizon, len(val_hpo_series) - input_lags)
                if val_horizon_mimo < 1: 
                    logging.info(f"  INFO (HPO Trial): Pulando trial. 'input_lags' ({input_lags}) não deixa dados para validação (val_size={len(val_hpo_series)}).")
                    return 1e9
                
                X_train, y_train = data_prep_func(scaled_train_hpo, input_lags, horizon)
                X_val, y_val = data_prep_func(scaled_val_hpo, input_lags, val_horizon_mimo)
                if X_val.shape[0] == 0: return 1e9
                
                y_train_subset = y_train[:, :val_horizon_mimo]
                
                model = model_builder(input_shape=(input_lags,), output_shape=val_horizon_mimo, **builder_params)
                model.fit(X_train, y_train_subset, **fit_params, verbose=0)
                preds = model.predict(X_val, verbose=0)
                return mean_squared_error(y_val, preds)

            elif model_type in ['Hybrid_MLP_Recursive', 'Hybrid_LSTM_Recursive']:
                if len(scaled_val_hpo) < input_lags + 1:
                    logging.info(f"  INFO (HPO Trial): Pulando trial. 'input_lags' ({input_lags}) não deixa dados para validação (val_size={len(scaled_val_hpo)}).")
                    return 1e9

                X_train, y_train = data_prep_func(scaled_train_hpo, input_lags)
                X_val, y_val = data_prep_func(scaled_val_hpo, input_lags)
                if X_val.shape[0] == 0: return 1e9
                
                is_lstm = 'LSTM' in model_type
                model = model_builder(input_shape=(input_lags,), output_shape=1, **builder_params)
                
                if is_lstm: X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                model.fit(X_train, y_train, **fit_params, verbose=0)
                
                last_window = scaled_train_hpo.values[-input_lags:].tolist()
                preds_recursive = []
                for _ in range(len(val_hpo_series)):
                    input_data = np.array(last_window).reshape(1, -1)
                    if is_lstm: input_data = input_data.reshape(1, input_lags, 1)
                    pred = model.predict(input_data, verbose=0)[0, 0]
                    preds_recursive.append(pred)
                    last_window.pop(0)
                    last_window.append(pred)
                
                return mean_squared_error(y_val, preds_recursive[:len(y_val)])

            return 1e9
        
        except Exception as e:
            logging.warning(f"Trial HPO falhou: {e}. Retornando erro alto.\n{traceback.format_exc()}")
            return 1e9
    # --- FIM DA FUNÇÃO OBJECTIVE ---

    print(f"Iniciando HPO para {execution_name} com {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params_hpo = study.best_params
    print(f"HPO concluído. Melhor MSE de validação: {study.best_value:.6f}")
    print(f"Melhores hiperparâmetros encontrados: {best_params_hpo}")
    
    # --- SALVAMENTO DOS MELHORES PARÂMETROS (CRUCIAL) ---
    import json
    params_save_path = os.path.join(models_path, f"{execution_name}_params.json")
    try:
        with open(params_save_path, 'w') as f:
            json.dump(best_params_hpo, f)
        print(f"[INFO] Melhores parâmetros salvos em: {params_save_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar parâmetros HPO: {e}")


    # --- TREINO FINAL DO MODELO COM OS MELHORES PARÂMETROS ---
    print(f"Treinando modelo final com os melhores parâmetros na série de treino completa...")

    final_n_layers = best_params_hpo.get('n_layers', 1)
    final_n_neurons = []
    for i in range(final_n_layers):
        neuron_key = f'n_neurons_l{i+1}'
        if neuron_key in best_params_hpo:
            final_n_neurons.append(best_params_hpo[neuron_key])
        elif i == 0:
            final_n_neurons.append(32)
    
    if 'learning_rate' not in best_params_hpo or 'input_lags' not in best_params_hpo:
        logging.error(f"HPO falhou em encontrar parâmetros válidos para {execution_name}. Todos os {n_trials} trials falharam.")
        logging.error("Usando parâmetros de fallback (podem não ser ótimos). Verifique os logs de HPO.")
        best_params_hpo.setdefault('learning_rate', float(hpo_space['learning_rate'][1]))
        best_params_hpo.setdefault('input_lags', hpo_space['input_lags'][1])

    # 1. Combina todos os parâmetros necessários para train_and_save_keras_model
    final_model_params = {
        'epochs': fixed_params.get('epochs', 100),
        'batch_size': fixed_params.get('batch_size', 16),
        'learning_rate': best_params_hpo['learning_rate'],
        'n_neurons': final_n_neurons
    }
    final_input_lags = best_params_hpo['input_lags']
    
    final_scaler = get_scaler(scaler_type)
    scaled_data_to_train = scale_series(data_to_train, final_scaler)
    
    if scaler:
        scaler_path = os.path.join(models_path, f"{execution_name}_scaler.joblib")
        joblib.dump(final_scaler, scaler_path)
        print(f"[INFO] Scaler final salvo em: {scaler_path}")

    # 3. Treina e salva os modelos finais
    if model_type in ['MLP_Direct', 'LSTM', 'Hybrid_MLP_Direct']:
        datasets_final = data_prep_func(scaled_data_to_train, final_input_lags, horizon)
        for h in range(1, horizon + 1):
            X_train_final, y_train_final = datasets_final[h]
            model_path_final = os.path.join(models_path, f"{execution_name}_h{h}.keras")
            
            # 2. Passa o model_builder original e o dicionário de parâmetros completo
            deep_learning_model.train_and_save_keras_model(
                X_train_final, y_train_final, model_path_final, 
                model_params=final_model_params, # Passa o dict unificado
                model_builder=model_builder,
                output_shape=1)
                
    elif model_type in ['MLP_MIMO', 'Hybrid_MLP_MIMO']:
        X_train_final, y_train_final = data_prep_func(scaled_data_to_train, final_input_lags, horizon)
        model_path_final = os.path.join(models_path, f"{execution_name}.keras")
        deep_learning_model.train_and_save_keras_model(
            X_train_final, y_train_final, model_path_final,
            model_params=final_model_params,
            model_builder=model_builder,
            output_shape=horizon)
            
    elif model_type in ['Hybrid_MLP_Recursive', 'Hybrid_LSTM_Recursive']:
        X_train_final, y_train_final = data_prep_func(scaled_data_to_train, final_input_lags)
        model_path_final = os.path.join(models_path, f"{execution_name}.keras")
        deep_learning_model.train_and_save_keras_model(
            X_train_final, y_train_final, model_path_final, 
            model_params=final_model_params,
            model_builder=model_builder,
            output_shape=1)

    print(f"Treinamento final para {execution_name} concluído.")