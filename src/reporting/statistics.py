# File: src/reporting/statistics.py
import numpy as np
import pandas as pd
from scipy import stats
import math

def calculate_pd(mape_base, mape_comp):
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(mape_comp) or mape_base == 0:
        return np.nan
    return 100 * (mape_comp - mape_base) / mape_base

def calculate_dm_statistic(actual, pred_base, pred_comp):
    try:
        e_base = np.abs(actual - pred_base)
        e_comp = np.abs(actual - pred_comp)
        d = e_base - e_comp
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        if var_d == 0: return np.nan, np.nan
        n = len(d)
        dm_stat = mean_d / np.sqrt(var_d / n)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        return dm_stat, p_value
    except:
        return np.nan, np.nan

def get_nemenyi_cd(k, N):
    """
    Calcula a Diferença Crítica (CD) para o teste de Nemenyi (alpha=0.05).
    Usado pelo CD Diagram.
    """
    # Valores críticos q_alpha para k modelos (Demšar, 2006)
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
        16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }
    # Aproximação para k > 20
    qa = q_alpha_05.get(k, 3.544 + (k-20)*0.02)
    cd = qa * np.sqrt((k * (k + 1)) / (6 * N))
    return cd