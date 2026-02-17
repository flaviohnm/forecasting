import numpy as np
import pandas as pd
from scipy import stats
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def dm_test(actual, pred1, pred2, h=1, criterion="MAE"):
    """
    Teste de Diebold-Mariano (Robust version).
    """
    e1 = np.array(actual) - np.array(pred1)
    e2 = np.array(actual) - np.array(pred2)
    T = len(e1)
    
    # Critério de Perda
    if criterion == "MSE":
        d = (e1)**2 - (e2)**2
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MAPE":
        safe_actual = np.where(actual == 0, 1e-6, actual)
        d = (np.abs(e1)/np.abs(safe_actual)) - (np.abs(e2)/np.abs(safe_actual))
    
    # Proteção: Modelos idênticos
    if np.allclose(d, 0, atol=1e-8):
        return 0.0, 1.0 # P=1.0 (Sem diferença)

    d_mean = np.mean(d)
    
    # Autocovariância
    gamma0 = np.var(d, ddof=1)
    sum_gamma = 0
    
    if h > 1 and T > h:
        for lag in range(1, h):
            try:
                cov = np.cov(d[lag:], d[:-lag])[0, 1]
                sum_gamma += cov
            except: pass
                
    var_d = (gamma0 + 2 * sum_gamma) / T
    
    if var_d <= 1e-16: return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)
    
    # Correção HLN
    correction = ((T + 1 - 2*h + (h*(h-1))/T) / T) ** 0.5
    dm_stat_corrected = dm_stat * correction
    
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=T-1))
    
    if np.isnan(p_value): return 0.0, 1.0
        
    return dm_stat_corrected, p_value

def plot_dm_heatmap(df_results, output_path, dataset_name):
    """
    Gera o Heatmap clássico da literatura para testes estatísticos.
    """
    if df_results.empty: return

    # Filtra apenas o dataset atual
    df = df_results[df_results['Dataset'] == dataset_name].copy()
    if df.empty: return

    # Prepara a matriz: Pivota os dados
    # Queremos uma matriz onde Linha=Model_A, Coluna=Model_B, Valor=DM_Stat
    # Se DM_Stat > 0 (e sig), Model_B (Coluna) ganha.
    # Se DM_Stat < 0 (e sig), Model_A (Linha) ganha.
    
    pivot_table = df.pivot(index='Model_A', columns='Model_B', values='DM_Statistic')
    p_values = df.pivot(index='Model_A', columns='Model_B', values='P_Value')
    
    plt.figure(figsize=(12, 10))
    
    # Máscara para o triângulo superior (opcional, mas comum se for todos contra todos)
    # mask = np.triu(np.ones_like(pivot_table, dtype=bool))
    
    # Customização da cor:
    # Verde: Diferença positiva significativa
    # Vermelho: Diferença negativa significativa
    # Cinza: Não significativo
    
    def annotate_heatmap(val, p):
        if pd.isna(val) or pd.isna(p): return ""
        sig = "*" if p < 0.05 else "ns"
        return f"{val:.2f}\n({sig})"

    # Cria anotações personalizadas
    annotations = np.vectorize(annotate_heatmap)(pivot_table.values, p_values.values)
    
    sns.heatmap(pivot_table, annot=annotations, fmt="", cmap="RdBu_r", center=0,
                linewidths=.5, cbar_kws={'label': 'Estatística DM (Positivo = Coluna Vence)'})
    
    plt.title(f'Diebold-Mariano Test Heatmap - {dataset_name}\n(* = p < 0.05)', fontsize=14)
    plt.tight_layout()
    
    filename = os.path.join(output_path, f"DM_Heatmap_{dataset_name}.png")
    plt.savefig(filename)
    plt.close()
    logging.info(f"Heatmap DM salvo em: {filename}")

def run_significance_analysis(main_config, metrics_df=None):
    """
    Executa testes e gera visualizações na pasta dedicada.
    """
    # --- NOVO: Pasta Dedicada ---
    base_results = os.path.dirname(main_config['results_paths']['metrics'])
    stats_path = os.path.join(base_results, "statistical") # Nova pasta
    os.makedirs(stats_path, exist_ok=True)
    
    logging.info(f"--- [STATS] Iniciando DM Test (Saída em: {stats_path}) ---")
    
    forecasts_path = main_config['results_paths']['forecasts']
    results = []
    
    # (Lógica de carregar métricas igual ao anterior...)
    if metrics_df is None:
        files = glob.glob(os.path.join(forecasts_path, "*.csv"))
        metrics_df = pd.DataFrame({'dataset': [os.path.basename(f).split('_')[1] for f in files]})

    datasets = metrics_df['dataset'].unique()

    for ds in datasets:
        ds_files = glob.glob(os.path.join(forecasts_path, f"forecast_{ds}_*.csv"))
        models_map = {}
        for f in ds_files:
            # Ex: forecast_ETTh1_Informer_h96.csv -> Informer_h96
            # Removemos o prefixo fixo e a extensão
            clean_name = os.path.basename(f).replace(f"forecast_{ds}_", "").replace(".csv", "")
            models_map[clean_name] = f
            
        model_names = list(models_map.keys())
        
        # Comparação "Todos contra Todos" (Para gerar o Heatmap completo)
        # Isso é pesado se tiver muitos modelos, mas gera o gráfico mais bonito da literatura
        for i, mod_a in enumerate(model_names):
            for j, mod_b in enumerate(model_names):
                if i >= j: continue # Evita duplicidade e auto-comparação
                
                try:
                    df_a = pd.read_csv(models_map[mod_a])
                    df_b = pd.read_csv(models_map[mod_b])
                    
                    merged = pd.merge(df_a, df_b, on=['unique_id', 'ds', 'y'], suffixes=('_A', '_B'))
                    if len(merged) > 10:
                        h = 1
                        if '_h' in mod_a: 
                            try: h = int(mod_a.split('_h')[-1]) 
                            except: pass
                        
                        dm, p_val = dm_test(merged['y'], merged['y_hat_B'], merged['y_hat_A'], h=h)
                        
                        results.append({
                            'Dataset': ds,
                            'Model_A': mod_a,
                            'Model_B': mod_b,
                            'DM_Statistic': dm,
                            'P_Value': p_val,
                            'Significant': p_val < 0.05
                        })
                except Exception:
                    pass

    if results:
        df_res = pd.DataFrame(results)
        # Salva CSV na nova pasta
        df_res.to_csv(os.path.join(stats_path, "dm_results_full.csv"), index=False)
        
        # Gera o Gráfico por Dataset
        for ds in datasets:
            try:
                plot_dm_heatmap(df_res, stats_path, ds)
            except Exception as e:
                logging.error(f"Erro ao plotar heatmap para {ds}: {e}")
                
    else:
        logging.warning("Nenhum resultado estatístico gerado.")