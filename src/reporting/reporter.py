# File: src/reporting/reporter.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib
# Garante backend não interativo para plotagem
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    logging.warning("Matplotlib não encontrado. Gráficos não serão gerados.")


def calculate_pd(mape_base, mape_comp):
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(mape_comp) or mape_base == 0:
        return np.nan
    return 100 * (mape_comp - mape_base) / mape_base

# --- FUNÇÃO 1: BOXPLOT DE RANKINGS (GLOBAL) ---
def generate_rank_boxplot(summary_df, output_dir):
    if not PLOTTING_ENABLED or summary_df.empty: return None

    # Calcula Ranking por Dataset
    df_ranked = summary_df.copy()
    df_ranked['rank'] = df_ranked.groupby('dataset')['MAPE'].rank(ascending=True, method='min')

    # Ordena modelos pela mediana do ranking
    model_stats = df_ranked.groupby('model_type')['rank'].median().sort_values()
    sorted_models = model_stats.index.tolist()

    data_to_plot = []
    for model in sorted_models:
        ranks = df_ranked[df_ranked['model_type'] == model]['rank'].values
        data_to_plot.append(ranks)

    plt.figure(figsize=(14, 8))
    box = plt.boxplot(data_to_plot, patch_artist=True, showfliers=True)
    
    colors = ['#ADD8E6'] * len(data_to_plot)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(range(1, len(sorted_models) + 1), sorted_models, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Posição no Ranking (1 = Melhor)", fontsize=12)
    plt.xlabel("Modelos", fontsize=12)
    plt.title("Distribuição dos Rankings de Cada Modelo (Consistência Global)", fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    max_rank = df_ranked['rank'].max()
    plt.yticks(np.arange(1, max_rank + 2, 1))
    plt.ylim(0.5, max_rank + 1.5)
    plt.tight_layout()

    filename = "boxplot_rankings_global.png"
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
        return filename
    except:
        plt.close()
        return None

# --- FUNÇÃO 2: GRÁFICO DE PD (PERCENTAGE DIFFERENCE) ---
def generate_pd_plot(metrics_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or metrics_df.empty: return None
    if 'MAPE' not in metrics_df.columns: return None
    
    df = metrics_df.copy()
    best_mape = df['MAPE'].min()
    df['PD'] = 100 * (df['MAPE'] - best_mape) / best_mape
    df_sorted = df.sort_values('PD')
    
    group_colors = {
        'benchmark_statistical': '#1f77b4',
        'benchmark_standalone_dl': '#ff7f0e',
        'hybrid_recursive': '#2ca02c',
        'hybrid_direct': '#d62728',
        'hybrid_mimo': '#9467bd',
        'other': '#7f7f7f'
    }
    bar_colors = [group_colors.get(g, group_colors['other']) for g in df_sorted['comparison_group']]

    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(df_sorted))
    bars = plt.bar(x_pos, df_sorted['PD'], color=bar_colors, edgecolor='black', alpha=0.8)
    
    plt.title(f"Diferença Percentual (PD) do MAPE - {dataset_name}", fontsize=14, pad=15)
    plt.ylabel("PD (%) - Pior que o Campeão", fontsize=11)
    plt.xticks(x_pos, df_sorted['model_type'], rotation=45, ha='right', fontsize=9)
    
    present_groups = df_sorted['comparison_group'].unique()
    legend_handles = [mpatches.Patch(color=group_colors.get(g, 'gray'), label=g) for g in present_groups]
    plt.legend(handles=legend_handles, title="Grupo", loc='best', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0.1:
            plt.text(bar.get_x() + bar.get_width()/2., height + (height*0.01), f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        elif height == 0:
            plt.text(bar.get_x() + bar.get_width()/2., 0, '★', ha='center', va='bottom', fontsize=12, color='gold')

    plt.tight_layout()
    filename = f"pd_plot_{dataset_name}.png"
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=120)
        plt.close()
        return filename
    except:
        plt.close()
        return None

# --- FUNÇÃO 3: PLOT PROPOSTA VS REFERÊNCIA ---
def generate_proposal_vs_reference_plot(forecasts_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or forecasts_df.empty: return None
    
    PROPOSED_MODEL = "Hybrid_MIMO_NBEATS_NF"
    REFERENCE_MODEL = "Hybrid_Direct_NBEATS_NF"
    
    subset_df = forecasts_df[forecasts_df['model_type'].isin([PROPOSED_MODEL, REFERENCE_MODEL])].copy()
    if subset_df.empty: return None
    
    subset_df = subset_df.sort_values('date_index')
    
    # Dados Reais
    if not subset_df[subset_df['model_type'] == PROPOSED_MODEL].empty:
        base_model = PROPOSED_MODEL
    else:
        base_model = REFERENCE_MODEL
    real_data = subset_df[subset_df['model_type'] == base_model][['date_index', 'real']].drop_duplicates().set_index('date_index')
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_data.index, real_data['real'], color='black', linestyle='--', linewidth=2, label='Valor Real', alpha=0.6)
    
    # Referência
    ref_data = subset_df[subset_df['model_type'] == REFERENCE_MODEL].set_index('date_index')
    if not ref_data.empty:
        plt.plot(ref_data.index, ref_data['previsao'], color='#ff7f0e', marker='x', markersize=6, linestyle='-', label=f'Ref: {REFERENCE_MODEL}', alpha=0.9)

    # Proposta
    prop_data = subset_df[subset_df['model_type'] == PROPOSED_MODEL].set_index('date_index')
    if not prop_data.empty:
        plt.plot(prop_data.index, prop_data['previsao'], color='#1f77b4', marker='o', markersize=5, linestyle='-', label=f'Prop: {PROPOSED_MODEL}', alpha=0.9)

    plt.title(f"Comparativo: Proposta vs Referência - {dataset_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    filename = f"proposal_vs_ref_{dataset_name}.png"
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=120)
        plt.close()
        return filename
    except:
        plt.close()
        return None

# --- FUNÇÃO 4: PAINEL REAL VS PREVISTO (FACETADO) ---
def generate_actual_vs_predicted_plots(forecasts_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or forecasts_df.empty: return None

    order_precedence = {'benchmark_statistical': 0, 'benchmark_standalone_dl': 1, 'hybrid_direct': 2, 'hybrid_mimo': 3, 'hybrid_recursive': 4}
    unique_groups = forecasts_df['comparison_group'].unique()
    sorted_groups = sorted(unique_groups, key=lambda x: order_precedence.get(x, 99))
    n_groups = len(sorted_groups)
    if n_groups == 0: return None

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 5 * n_groups), sharex=True)
    if n_groups == 1: axes = [axes]
    
    fig.suptitle(f"Painel Geral por Grupo - {dataset_name}", fontsize=16, y=0.99)
    forecasts_df = forecasts_df.sort_values('date_index')
    first_model = forecasts_df.iloc[0]['model_type']
    real_data = forecasts_df[forecasts_df['model_type'] == first_model][['date_index', 'real']].drop_duplicates().set_index('date_index')

    filename = f"forecast_panel_{dataset_name}.png"
    
    for ax, group in zip(axes, sorted_groups):
        group_data = forecasts_df[forecasts_df['comparison_group'] == group]
        ax.plot(real_data.index, real_data['real'], color='black', linestyle='--', linewidth=2, label='REAL', alpha=0.6)
        
        for model in sorted(group_data['model_type'].unique()):
            model_subset = group_data[group_data['model_type'] == model].set_index('date_index')
            if not model_subset.empty:
                ax.plot(model_subset.index, model_subset['previsao'], marker='.', markersize=4, label=model, alpha=0.8)
        
        ax.set_title(f"Grupo: {group}", fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')

    axes[-1].set_xlabel("Data")
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close()
        return filename
    except:
        plt.close()
        return None

# --- FUNÇÃO PRINCIPAL ---
def generate_report(main_config: dict, successful_executions: list):
    print("Iniciando a geração do relatório completo...")
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    comparison_results_path = os.path.join("results", "comparison_tests")
    report_dir = "reports"
    report_plot_dir = os.path.join(report_dir, "plots")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(report_plot_dir, exist_ok=True)

    all_metrics = []
    all_forecasts = []
    
    # Carrega estatísticas par-a-par (Diebold-Mariano)
    dm_df_all = pd.DataFrame()
    dm_file = os.path.join(comparison_results_path, "diebold_mariano_summary.csv")
    if os.path.exists(dm_file):
        try: dm_df_all = pd.read_csv(dm_file)
        except: pass

    for dataset_conf, model_conf in successful_executions:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        metric_file = os.path.join(metrics_path, f"metrics_{execution_name}.csv")
        forecast_file = os.path.join(plots_path, f"forecasts_{execution_name}.csv")
        comparison_group = model_conf.get('comparison_group', 'other')

        if os.path.exists(metric_file):
            try:
                df = pd.read_csv(metric_file)
                if not df.empty:
                    df['comparison_group'] = comparison_group
                    all_metrics.append(df)
            except: pass
        
        if os.path.exists(forecast_file):
            try:
                df_f = pd.read_csv(forecast_file)
                if not df_f.empty:
                    date_col = df_f.columns[0]
                    df_f.rename(columns={date_col: 'date_index'}, inplace=True)
                    try: df_f['date_index'] = pd.to_datetime(df_f['date_index'])
                    except: pass
                    df_f['model_type'] = model_conf['model_type']
                    df_f['dataset'] = dataset_conf['name']
                    df_f['comparison_group'] = comparison_group
                    cols = ['date_index', 'real', 'previsao', 'model_type', 'dataset', 'comparison_group']
                    valid_cols = [c for c in cols if c in df_f.columns]
                    all_forecasts.append(df_f[valid_cols])
            except: pass

    if not all_metrics:
        print("Nenhuma métrica encontrada.")
        return

    summary_df = pd.concat(all_metrics, ignore_index=True)
    forecasts_df_all = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    # --- FILTRO DE MODELOS ---
    MODELS_TO_EXCLUDE = ['ETS', 'NAIVE', 'SEASONAL_NAIVE']
    summary_df = summary_df[~summary_df['model_type'].isin(MODELS_TO_EXCLUDE)]
    if not forecasts_df_all.empty:
        forecasts_df_all = forecasts_df_all[~forecasts_df_all['model_type'].isin(MODELS_TO_EXCLUDE)]
    if not dm_df_all.empty:
        dm_df_all = dm_df_all[~dm_df_all['modelo_base'].isin(MODELS_TO_EXCLUDE)]
        dm_df_all = dm_df_all[~dm_df_all['modelo_comparado'].isin(MODELS_TO_EXCLUDE)]

    if summary_df.empty:
        print("Todos os modelos filtrados.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_content = f"# Relatório Final de Experimentos\n\nGerado em: {now}\n\n"

    # --- 1. TABELA DE CAMPEÕES ---
    md_content += "## 1. Resumo dos Campeões (Menor MAPE)\n\n"
    try:
        winners_idx = summary_df.groupby('dataset')['MAPE'].idxmin()
        winners = summary_df.loc[winners_idx].sort_values('dataset')[['comparison_group', 'model_type', 'dataset', 'MAPE', 'MASE', 'RMSSE']]
        winners.columns = ['Grupo', 'Modelo', 'Dataset', 'MAPE', 'MASE', 'RMSSE']
        for col in ['MAPE', 'MASE', 'RMSSE']:
            winners[col] = winners[col].apply(lambda x: f"**{x:.4f}**")
        md_content += winners.to_markdown(index=False)
    except: pass
    md_content += "\n\n---\n\n"

    # --- 2. ANÁLISE GLOBAL (BOXPLOT) ---
    md_content += "## 2. Análise Global\n\n"
    
    # Boxplot
    boxplot_file = generate_rank_boxplot(summary_df, report_plot_dir)
    if boxplot_file:
        md_content += "### Consistência dos Rankings\n"
        md_content += f"![Boxplot](plots/{boxplot_file})\n\n"
    md_content += "---\n\n"

    # --- 3. DETALHES POR DATASET ---
    md_content += "## 3. Detalhamento por Dataset\n\n"
    datasets = sorted(summary_df['dataset'].unique())
    metrics = ['MAPE', 'MASE', 'RMSSE']
    sort_mapping = {'benchmark_statistical': 0, 'benchmark_standalone_dl': 1, 'hybrid_direct': 2, 'hybrid_mimo': 3, 'hybrid_recursive': 4}

    for ds in datasets:
        md_content += f"### Dataset: {ds}\n\n"
        ds_df = summary_df[summary_df['dataset'] == ds].copy()
        ds_forc = forecasts_df_all[forecasts_df_all['dataset'] == ds] if not forecasts_df_all.empty else pd.DataFrame()

        # A. GRÁFICO PD
        pd_plot = generate_pd_plot(ds_df, report_plot_dir, ds)
        if pd_plot:
            md_content += f"**Performance Relativa (PD):**\n\n![PD](plots/{pd_plot})\n\n"

        # B. TABELA ESTATÍSTICA (DM)
        if not dm_df_all.empty:
            ds_dm = dm_df_all[dm_df_all['dataset'] == ds]
            if not ds_dm.empty:
                md_content += "**Teste Diebold-Mariano:**\n\n"
                tbl = ds_dm[['modelo_comparado', 'dm_statistic', 'p_value']].copy()
                tbl.columns = ['Modelo', 'DM Stat', 'p-Value']
                tbl['p-Value'] = tbl['p-Value'].apply(lambda x: f"{x:.2E}")
                md_content += tbl.to_markdown(index=False) + "\n\n"

        # C. PROPOSTA VS REFERÊNCIA
        if not ds_forc.empty:
            prop_plot = generate_proposal_vs_reference_plot(ds_forc, report_plot_dir, ds)
            if prop_plot:
                md_content += f"**Destaque: Proposta vs Referência**\n\n![PropVsRef](plots/{prop_plot})\n\n"
            
            # D. PAINEL GERAL
            panel_plot = generate_actual_vs_predicted_plots(ds_forc, report_plot_dir, ds)
            if panel_plot:
                md_content += f"**Visão Geral: Real vs Previsto**\n\n![Panel](plots/{panel_plot})\n\n"

        # E. TABELAS DE RANKING
        for m in metrics:
            if m not in ds_df.columns: continue
            md_content += f"**Ranking {m}:**\n\n"
            best_val = ds_df[m].min()
            ds_df['sort'] = ds_df['comparison_group'].map(sort_mapping).fillna(99)
            sorted_df = ds_df.sort_values(by=['sort', 'model_type'])
            
            view = sorted_df[['comparison_group', 'model_type', 'dataset', m]].copy()
            view.columns = ['Grupo', 'Modelo', 'Dataset', m]
            view[m] = view[m].apply(lambda x: f"{x:.4f}")
            
            def bold(row):
                if abs(sorted_df.loc[row.name, m] - best_val) < 1e-9:
                    return [f"**{x}**" for x in row]
                return row.tolist()
            
            final_tbl = pd.DataFrame(view.apply(bold, axis=1).tolist(), columns=view.columns)
            md_content += final_tbl.to_markdown(index=False) + "\n\n"
        
        md_content += "---\n"

    report_file = os.path.join(report_dir, f"relatorio_final_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(report_file, 'w', encoding='utf-8') as f: f.write(md_content)
    print(f"Relatório gerado: {report_file}")