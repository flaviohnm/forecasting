import collections
import glob
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
    HAS_SP = True
except ImportError:
    HAS_SP = False
    logging.warning("scikit-posthocs não instalado. O CD Diagram exige este pacote. Use: poetry add scikit-posthocs")


def run_significance_analysis(main_config, metrics_df=None):
    stats_path = main_config["results_paths"]["statistical"]
    os.makedirs(stats_path, exist_ok=True)
    
    logging.info(f"--- [STATS] Iniciando Testes de Friedman e Nemenyi (Padrão CD Diagram) ---")

    forecasts_path = main_config["results_paths"]["forecasts"]
    files = glob.glob(os.path.join(forecasts_path, "forecast_*.csv"))

    if not files:
        logging.warning("Nenhum arquivo de forecast encontrado para análise estatística.")
        return

    # Agrupamos as análises por Dataset e Horizonte (Rigidez Matemática)
    groups = collections.defaultdict(list)
    
    for f in files:
        basename = os.path.basename(f).replace("forecast_", "").replace(".csv", "")
        ds_name = basename.split("_")[0]
        match = re.search(r'_h(\d+)$', basename)
        horizon = match.group(1) if match else "unknown"
        group_key = f"{ds_name}_h{horizon}"
        groups[group_key].append((basename, f))

    for group_key, runs in groups.items():
        model_errors = {}
        for exec_name, fpath in runs:
            try:
                # O reset_index garante o alinhamento perfeito das linhas sem falhar no merge
                df = pd.read_csv(fpath).sort_values(["unique_id", "ds"]).reset_index(drop=True)
                
                # Limpa o nome para o gráfico ficar bonito
                display_name = exec_name.replace(f"{group_key.split('_')[0]}_", "")
                display_name = re.sub(r'_h\d+$', '', display_name).upper()
                
                # Para o Friedman, extraímos o Erro Absoluto de cada ponto de tempo
                model_errors[display_name] = np.abs(df["y"] - df["y_hat"])
            except Exception as e:
                logging.error(f"Erro ao processar {fpath} para estatística: {e}")

        df_errors = pd.DataFrame(model_errors).dropna()

        # O Teste de Friedman exige no mínimo 3 modelos concorrentes
        if df_errors.empty or len(df_errors.columns) < 3:
            logging.info(f"[{group_key}] Modelos insuficientes (< 3) para Friedman. Pulando.")
            continue

        # 1. Teste Não-Paramétrico de Friedman
        stat, p_val = stats.friedmanchisquare(*[df_errors[col] for col in df_errors.columns])
        logging.info(f"[{group_key}] Friedman p-value: {p_val:.4e}")

        # Se p < 0.05, a diferença global não é obra do acaso, e disparamos o Post-hoc
        if HAS_SP and p_val < 0.05:
            # 2. Teste Post-Hoc de Nemenyi
            p_values = sp.posthoc_nemenyi_friedman(df_errors.values)
            p_values.columns = df_errors.columns
            p_values.index = df_errors.columns
            
            # Ranking médio de cada modelo
            ranks = df_errors.rank(axis=1).mean()
            
            # 3. Desenho do Critical Difference (CD) Diagram
            height = max(4.0, len(df_errors.columns) * 0.4)
            plt.figure(figsize=(10, height))
            
            try:
                sp.critical_difference_diagram(ranks, p_values)
                plt.title(f"Critical Difference Diagram (Nemenyi) - {group_key.upper()}\n(Friedman p-val: {p_val:.4e})")
                plt.tight_layout()
                plt.savefig(os.path.join(stats_path, f"CD_Diagram_{group_key}.png"), dpi=150, bbox_inches='tight')
                plt.close()
            except AttributeError:
                # Fallback caso a versão da biblioteca seja muito antiga
                sns.heatmap(p_values, annot=True, cmap="RdBu_r", vmin=0, vmax=0.1)
                plt.title(f"Nemenyi Post-hoc P-values - {group_key.upper()}")
                plt.tight_layout()
                plt.savefig(os.path.join(stats_path, f"CD_Diagram_{group_key}.png"))
                plt.close()
        elif HAS_SP:
            logging.info(f"[{group_key}] p-value > 0.05. Modelos empatados estatisticamente. Pulando CD Diagram.")