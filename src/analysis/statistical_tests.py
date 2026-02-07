import numpy as np
import pandas as pd
from scipy import stats
import os
import logging

def dm_test(actual, pred1, pred2, h=1, criterion="MAE", poly_degree=2):
    """
    Teste de Diebold-Mariano (1995).
    H0: A precisão preditiva dos dois modelos é igual.
    H1: A precisão é diferente.
    
    Retorna: DM statistic, p-value
    """
    e1 = np.array(actual) - np.array(pred1)
    e2 = np.array(actual) - np.array(pred2)
    
    T = len(e1)
    
    # Define a função de perda (Loss Function)
    if criterion == "MSE":
        d = (e1)**2 - (e2)**2
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MAPE":
        d = (np.abs(e1)/np.abs(actual)) - (np.abs(e2)/np.abs(actual))
        
    # Média e variância da diferença de perda
    d_mean = np.mean(d)
    
    # Autocovariância para correção de correlação serial (Kernel HAC)
    gamma0 = np.var(d)
    sum_gamma = 0
    
    # Para horizontes h > 1, usamos um estimador de kernel para a variância
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
        sum_gamma += gamma_lag
        
    var_d = (gamma0 + 2 * sum_gamma) / T
    
    # Estatística DM
    if var_d > 0:
        dm_stat = d_mean / np.sqrt(var_d)
    else:
        dm_stat = 0
        
    # P-value (Bilateral)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def run_significance_analysis(main_config, successful_runs):
    """
    Compara todos os modelos Híbridos contra suas Bases.
    """
    logging.info("--- [STATS] Iniciando Teste de Diebold-Mariano ---")
    
    forecasts_path = main_config['results_paths']['forecasts']
    comparison_path = main_config['results_paths']['comparison']
    os.makedirs(comparison_path, exist_ok=True)
    
    results = []

    # Identifica pares para comparação
    # Lógica: Se tem "hybrid_X_on_Y", comparamos com "standalone_Y"
    for run in successful_runs:
        if "_on_" in run:
            parts = run.split('_on_')
            # Ex: ETTh1_hybrid_nhits_on_base_arima_h720
            # Precisamos extrair o nome da base e o horizonte
            
            # Recupera horizonte do final da string
            horizon_str = run.split('_h')[-1]
            h = int(horizon_str)
            
            # Recupera dataset
            dataset = run.split('_')[0]
            
            # Recupera nome da base (ex: standalone_arima)
            # O nome da execução híbrida é: {dataset}_{hybrid_model}_on_{base_model}_h{h}
            # O nome da execução base é:    {dataset}_standalone_{base_model}_h{h}
            
            # Hack de string para pegar o nome da base corretamente
            # Remove o dataset e o horizonte
            middle = run.replace(f"{dataset}_", "").replace(f"_h{h}", "")
            hybrid_model, base_model_name = middle.split('_on_')
            
            # Reconstrói o nome do arquivo da base
            base_run = f"{dataset}_{base_model_name}_h{h}"
            
            file_hybrid = os.path.join(forecasts_path, f"forecast_{run}.csv")
            file_base = os.path.join(forecasts_path, f"forecast_{base_run}.csv")
            
            if os.path.exists(file_hybrid) and os.path.exists(file_base):
                df_h = pd.read_csv(file_hybrid)
                df_b = pd.read_csv(file_base)
                
                # Garante alinhamento
                merged = pd.merge(df_h, df_b, on=['unique_id', 'ds', 'y'], suffixes=('_hybrid', '_base'))
                
                if len(merged) > 0:
                    dm, p_val = dm_test(merged['y'], merged['y_hat_base'], merged['y_hat_hybrid'], h=h)
                    
                    is_significant = p_val < 0.05
                    improvement = (np.mean(np.abs(merged['y'] - merged['y_hat_base'])) - np.mean(np.abs(merged['y'] - merged['y_hat_hybrid']))) > 0
                    
                    results.append({
                        'Dataset': dataset,
                        'Horizon': h,
                        'Hybrid_Model': hybrid_model,
                        'Base_Model': base_model_name,
                        'DM_Statistic': round(dm, 4),
                        'P_Value': round(p_val, 6),
                        'Significant_0.05': is_significant,
                        'Hybrid_Wins': is_significant and improvement
                    })
                    logging.info(f"DM Test {run} vs {base_run}: p={p_val:.4f} | Sig={is_significant}")

    if results:
        df_res = pd.DataFrame(results)
        output_file = os.path.join(comparison_path, "diebold_mariano_results.csv")
        df_res.to_csv(output_file, index=False)
        logging.info(f"Resultados estatísticos salvos em {output_file}")
    else:
        logging.warning("Nenhum par Híbrido-Base encontrado para comparação estatística.")