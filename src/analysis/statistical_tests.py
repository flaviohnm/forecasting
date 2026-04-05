import collections
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

def dm_test(actual, pred_benchmark, pred_proposed):
    """
    Diebold-Mariano test (Baseado em Erro Absoluto Percentual - APE).
    Compara o modelo benchmark (a) contra o modelo proposto (b).
    """
    # Evita divisão por zero
    actual_safe = np.where(actual == 0, 1e-8, actual)
    
    e1 = np.abs((actual_safe - pred_benchmark) / actual_safe)
    e2 = np.abs((actual_safe - pred_proposed) / actual_safe)
    d = e1 - e2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    
    den = var_d / len(d)
    if den <= 0 or np.isnan(den):
        return 0.0, 1.0
        
    stat = mean_d / np.sqrt(den)
    # p-value bicaudal
    p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p_value

def run_significance_analysis(main_config, metrics_df=None):
    stats_path = main_config["results_paths"]["statistical"]
    os.makedirs(stats_path, exist_ok=True)
    
    logging.info("--- [STATS] Iniciando Teste de Diebold-Mariano (DM) Agregado ---")

    forecasts_path = main_config["results_paths"]["forecasts"]
    files = glob.glob(os.path.join(forecasts_path, "forecast_*.csv"))

    if not files:
        logging.warning("Nenhum arquivo de forecast encontrado para DM.")
        return
    
    # Agrupa os DataFrames por (Dataset, Modelo_Core) empilhando os horizontes
    model_forecasts = collections.defaultdict(list)
    
    for f in files:
        basename = os.path.basename(f).replace("forecast_", "").replace(".csv", "")
        model_core = re.sub(r'_h\d+$', '', basename)
        ds = basename.split("_")[0]
        
        df = pd.read_csv(f).sort_values(["unique_id", "ds"]).reset_index(drop=True)
        model_forecasts[(ds, model_core)].append(df)
        
    aggregated_forecasts = {}
    for (ds, m_core), dfs in model_forecasts.items():
        df_concat = pd.concat(dfs, ignore_index=True)
        aggregated_forecasts[(ds, m_core)] = df_concat
        
    results = []
    
    # Identifica os modelos híbridos para testar
    for (ds, m_core) in aggregated_forecasts.keys():
        if "hybrid_" in m_core:
            df_hyb = aggregated_forecasts[(ds, m_core)]
            actual = df_hyb["y"].values
            pred_hyb = df_hyb["y_hat"].values
            
            for (other_ds, other_core) in aggregated_forecasts.keys():
                if other_ds == ds and other_core != m_core and "hybrid_" not in other_core:
                    df_other = aggregated_forecasts[(other_ds, other_core)]
                    
                    if len(actual) == len(df_other):
                        pred_other = df_other["y_hat"].values
                        stat_val, p_val = dm_test(actual, pred_other, pred_hyb)
                        
                        hyb_display = m_core.replace(f"{ds}_hybrid_", "").replace("_on_", " + ").upper()
                        bench_display = other_core.replace(f"{ds}_base_", "").upper()
                        
                        results.append({
                            "Dataset": ds.upper(),
                            "Proposed Hybrid": hyb_display,
                            "Model": bench_display,
                            "DM Value": stat_val,
                            "p-Value": p_val
                        })
                        
    if results:
        df_dm = pd.DataFrame(results)
        df_dm = df_dm.sort_values(by="DM Value", ascending=False)
        
        # Formatação Padrão IEEE/Elsevier
        df_dm["DM Value"] = df_dm["DM Value"].apply(lambda x: f"{x:.3f}")
        df_dm["p-Value"] = df_dm["p-Value"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
        
        df_dm.to_csv(os.path.join(stats_path, "dm_test_results.csv"), index=False)
        logging.info("Resultados do Teste DM salvos em CSV com sucesso.")