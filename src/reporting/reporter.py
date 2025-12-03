# File: src/reporting/reporter.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import math
from scipy import stats

# Garante backend não interativo para plotagem
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    PLOTTING_ENABLED = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        PLOTTING_ENABLED = True
    except:
        PLOTTING_ENABLED = False
        logging.warning("Matplotlib não encontrado. Gráficos não serão gerados.")

def calculate_pd(mape_base, mape_comp):
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(mape_comp) or mape_base == 0:
        return np.nan
    return 100 * (mape_comp - mape_base) / mape_base

# --- ESTATÍSTICA: DIEBOLD-MARIANO ---
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

# --- VISUALIZAÇÃO 1: MATRIZ CONSOLIDADA ---
def generate_consolidated_matrix(summary_df, metric='MAPE'):
    if summary_df.empty: return "Sem dados."

    pivot_df = summary_df.pivot_table(
        index=['comparison_group', 'model_type'], 
        columns='dataset', 
        values=metric,
        aggfunc='first'
    )

    sort_mapping = {
        'benchmark_statistical': 0,
        'benchmark_standalone_dl': 1,
        'hybrid_direct': 2,
        'hybrid_mimo': 3,
        'hybrid_recursive': 4
    }
    
    pivot_df = pivot_df.sort_index(key=lambda x: x.map(sort_mapping) if x.name == 'comparison_group' else x)

    def highlight_min(s):
        is_min = s == s.min()
        return [f"**{v:.4f}**" if is_min_val and pd.notna(v) else f"{v:.4f}" if pd.notna(v) else "-" 
                for v, is_min_val in zip(s, is_min)]

    formatted_df = pivot_df.apply(highlight_min, axis=0).reset_index()
    formatted_df.columns.name = None 
    formatted_df.rename(columns={'comparison_group': 'Grupo', 'model_type': 'Modelo'}, inplace=True)

    return formatted_df.to_markdown(index=False)

# --- VISUALIZAÇÃO 2: BOXPLOT DE RANKINGS ---
def generate_rank_boxplot(summary_df, output_dir):
    if not PLOTTING_ENABLED or summary_df.empty: return None

    df_ranked = summary_df.copy()
    df_ranked['rank'] = df_ranked.groupby('dataset')['MAPE'].rank(ascending=True, method='min')
    
    model_stats = df_ranked.groupby('model_type')['rank'].median().sort_values()
    sorted_models = model_stats.index.tolist()

    data_to_plot = []
    for model in sorted_models:
        ranks = df_ranked[df_ranked['model_type'] == model]['rank'].values
        data_to_plot.append(ranks)

    plt.figure(figsize=(14, 8))
    box = plt.boxplot(data_to_plot, patch_artist=True, showfliers=True)
    
    for patch in box['boxes']:
        patch.set_facecolor('#ADD8E6')
        patch.set_alpha(0.7)

    plt.xticks(range(1, len(sorted_models) + 1), sorted_models, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Ranking (1 = Melhor)", fontsize=12)
    plt.title("Distribuição dos Rankings (Consistência Global)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    max_rank = df_ranked['rank'].max()
    plt.yticks(np.arange(1, max_rank + 2, 1))
    plt.tight_layout()

    filename = "boxplot_rankings_global.png"
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
        return filename
    except:
        plt.close()
        return None

# --- VISUALIZAÇÃO 3: CD DIAGRAM (ESTILO DEMŠAR - LAYOUT LIMPO) ---
def generate_cd_diagram(summary_df, output_dir):
    """
    Gera um Diagrama de Diferença Crítica (CD Diagram) estilo Demšar.
    Melhoria: Algoritmo de layout 'L-Shape' para evitar sobreposição de textos e linhas.
    """
    if not PLOTTING_ENABLED or summary_df.empty: return None

    # 1. Calcular Rankings
    df_ranked = summary_df.copy()
    df_ranked['rank'] = df_ranked.groupby('dataset')['MAPE'].rank(ascending=True, method='min')
    avg_ranks = df_ranked.groupby('model_type')['rank'].mean().sort_values()
    
    k = len(avg_ranks)
    N = df_ranked['dataset'].nunique()
    
    if k < 2 or N < 2: return None

    # 2. Calcular CD (Nemenyi alpha=0.05)
    # Valores críticos aproximados para infinito graus de liberdade
    q_alpha = {2:1.96, 3:2.34, 4:2.57, 5:2.73, 6:2.85, 7:2.95, 8:3.03, 9:3.10, 10:3.16, 15:3.39, 20:3.54}
    qa = q_alpha.get(k, 3.544 + (k-20)*0.02)
    cd = qa * np.sqrt((k * (k + 1)) / (6 * N))

    # 3. Configuração do Canvas
    # Altura dinâmica baseada no número de modelos para garantir espaço
    fig_height = 4 + (k * 0.3) 
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Configurar Eixo X (Rankings) - Invertido: 1 na direita, K na esquerda
    min_rank = 1
    max_rank = k
    ax.set_xlim(max_rank + 0.5, min_rank - 0.5) 
    ax.set_ylim(-2.5 - (k * 0.5), 1.5) # Espaço generoso para baixo
    
    # Limpar bordas
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_position(('data', 0))
    ax.spines['top'].set_linewidth(1.5) # Linha do eixo mais grossa
    
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks_position('top')
    
    # Ticks
    major_ticks = np.arange(min_rank, max_rank + 1, 1)
    ax.set_xticks(major_ticks)
    ax.tick_params(axis='x', which='major', length=8, width=1.5, labelsize=11, pad=5)

    # A. Desenhar Barra de Escala CD (Topo)
    cd_x_end = max_rank
    cd_x_start = cd_x_end - cd
    cd_y = 0.8 # Posição acima do eixo
    
    ax.hlines(cd_y, cd_x_start, cd_x_end, color='black', linestyle='--', linewidth=1.5)
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.1, cd_y + 0.1], color='black', linewidth=1.5)
    ax.plot([cd_x_end, cd_x_end], [cd_y - 0.1, cd_y + 0.1], color='black', linewidth=1.5)
    ax.text(cd_x_start + cd/2, cd_y + 0.15, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title("Diagrama de Diferença Crítica (Nemenyi, α=0.05)\nRanking Médio (Menor é Melhor)", fontsize=13, y=1.15)

    # B. Desenhar Cliques (Barras horizontais de significância) - LOGO ABAIXO DO EIXO
    # Isso evita que as barras grossas fiquem em cima dos textos
    sorted_ranks = avg_ranks.values
    sorted_names = avg_ranks.index
    
    bar_y_start = -0.2
    bar_y_step = 0.15
    current_level = 0
    
    i = 0
    while i < k:
        j = i + 1
        while j < k and (sorted_ranks[j] - sorted_ranks[i] < cd):
            j += 1
        
        if j > i + 1:
            start_rank = sorted_ranks[i]
            end_rank = sorted_ranks[j-1]
            
            # Desenha barra grossa
            bar_y = bar_y_start - (current_level * bar_y_step)
            ax.hlines(bar_y, start_rank, end_rank, color='black', linewidth=4, alpha=0.8)
            
            current_level = (current_level + 1) % 4 # Usa 4 níveis para evitar colisão de barras
        i += 1
        
    # Definir onde começam os textos (abaixo das barras de clique)
    text_y_start = bar_y_start - (4 * bar_y_step) - 0.5 

    # C. Desenhar Nomes e Linhas Conectoras (L-Shape)
    # Divide em Esquerda (Piores Ranks) e Direita (Melhores Ranks)
    mid_idx = math.ceil(k / 2)
    right_models = list(zip(sorted_ranks[:mid_idx], sorted_names[:mid_idx])) # Melhores (Rank baixo)
    left_models = list(zip(sorted_ranks[mid_idx:], sorted_names[mid_idx:]))  # Piores (Rank alto)
    
    # --- LADO DIREITO (Melhores Modelos - Ranks 1, 2, 3...) ---
    # Texto alinhado à direita do gráfico (perto do Rank 1)
    text_x_right = min_rank - 0.5 
    
    for idx, (rank, name) in enumerate(right_models):
        y_pos = text_y_start - (idx * 0.6) # Espaçamento fixo vertical
        
        # 1. Linha Vertical descendo do eixo
        ax.plot([rank, rank], [0, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 2. Linha Horizontal conectando ao texto
        # Vai do rank até a posição do texto na direita
        ax.plot([rank, text_x_right], [y_pos, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 3. Texto
        ax.text(text_x_right, y_pos, f" {name} ({rank:.2f})", ha='left', va='center', fontsize=9)

    # --- LADO ESQUERDO (Piores Modelos - Ranks N, N-1...) ---
    # Texto alinhado à esquerda do gráfico (perto do Rank Máximo)
    # Processamos de "pior" para "melhor" dentro desse grupo para alinhar visualmente de baixo para cima ou topo para baixo
    
    text_x_left = max_rank + 0.5
    
    for idx, (rank, name) in enumerate(left_models):
        # Continua a descida vertical de onde o lado direito parou, ou começa paralelo?
        # Para ficar bonito (como na imagem de exemplo), começamos do topo também
        y_pos = text_y_start - (idx * 0.6)
        
        # 1. Linha Vertical
        ax.plot([rank, rank], [0, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 2. Linha Horizontal
        ax.plot([rank, text_x_left], [y_pos, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 3. Texto
        ax.text(text_x_left, y_pos, f"({rank:.2f}) {name} ", ha='right', va='center', fontsize=9)

    plt.tight_layout()
    
    filename = "cd_diagram.png"
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico CD Diagram salvo: {filename}")
        return filename
    except Exception as e:
        logging.error(f"Erro ao salvar CD Diagram: {e}")
        plt.close()
        return None

# --- VISUALIZAÇÃO 4: PD PLOT ---
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

# --- VISUALIZAÇÃO 5: PROPOSTA VS REF ---
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
    
    ref_data = subset_df[subset_df['model_type'] == REFERENCE_MODEL].set_index('date_index')
    if not ref_data.empty:
        plt.plot(ref_data.index, ref_data['previsao'], color='#ff7f0e', marker='x', markersize=6, linestyle='-', label=f'Ref: {REFERENCE_MODEL}', alpha=0.9)

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

# --- VISUALIZAÇÃO 6: PAINEL REAL VS PREVISTO ---
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
    print("Iniciando a geração do relatório estruturado...")
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    report_dir = "reports"
    report_plot_dir = os.path.join(report_dir, "plots")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(report_plot_dir, exist_ok=True)

    all_metrics = []
    all_forecasts = []

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
        print("Nenhuma métrica encontrada. Relatório abortado.")
        return

    summary_df = pd.concat(all_metrics, ignore_index=True)
    forecasts_df_all = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    # --- FILTRO DE MODELOS ---
    MODELS_TO_EXCLUDE = ['ETS', 'NAIVE', 'SEASONAL_NAIVE']
    summary_df = summary_df[~summary_df['model_type'].isin(MODELS_TO_EXCLUDE)]
    if not forecasts_df_all.empty:
        forecasts_df_all = forecasts_df_all[~forecasts_df_all['model_type'].isin(MODELS_TO_EXCLUDE)]

    if summary_df.empty:
        print("Todos os modelos foram filtrados. Relatório abortado.")
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

    # --- 2. MATRIZES CONSOLIDADAS ---
    md_content += "## 2. Matrizes Consolidadas de Performance\n\n"
    md_content += "### Matriz MAPE\n" + generate_consolidated_matrix(summary_df, metric='MAPE') + "\n\n"
    md_content += "### Matriz MASE\n" + generate_consolidated_matrix(summary_df, metric='MASE') + "\n\n"
    md_content += "### Matriz RMSSE\n" + generate_consolidated_matrix(summary_df, metric='RMSSE') + "\n\n"
    md_content += "---\n\n"

    # --- 3. ANÁLISE GLOBAL ---
    md_content += "## 3. Análise Global\n\n"
    
    # 3.1 Boxplot
    boxplot_file = generate_rank_boxplot(summary_df, report_plot_dir)
    if boxplot_file:
        md_content += "### Distribuição dos Rankings (Boxplot)\n"
        md_content += f"![Boxplot](plots/{boxplot_file})\n\n"
    
    # 3.2 CD Diagram
    cd_file = generate_cd_diagram(summary_df, report_plot_dir)
    if cd_file:
        md_content += "### Diagrama de Diferença Crítica (CD Diagram)\n"
        md_content += "Compara o ranking médio de todos os modelos. Linhas conectam modelos estatisticamente equivalentes (p > 0.05).\n\n"
        md_content += f"![CD Diagram](plots/{cd_file})\n\n"
        
    md_content += "---\n\n"

    # --- 4. DETALHES POR DATASET ---
    md_content += "## 4. Detalhamento por Dataset\n\n"
    datasets = sorted(summary_df['dataset'].unique())
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
        if not ds_forc.empty and not ds_df.empty:
            winner_model_type = ds_df.loc[ds_df['MAPE'].idxmin()]['model_type']
            md_content += f"**Teste Diebold-Mariano (Referência Global: {winner_model_type})**\n\n"
            
            try:
                winner_preds_series = ds_forc[ds_forc['model_type'] == winner_model_type].sort_values('date_index')
                winner_preds = winner_preds_series['previsao'].values
                real_vals = winner_preds_series['real'].values
                
                if len(real_vals) > 0:
                    dm_dataset_list = []
                    groups_in_ds = ds_df['comparison_group'].unique()
                    groups_sorted = sorted(groups_in_ds, key=lambda x: sort_mapping.get(x, 99))
                    
                    for group in groups_sorted:
                        group_df = ds_df[ds_df['comparison_group'] == group]
                        models_in_group = group_df['model_type'].unique()
                        
                        for model in models_in_group:
                            if model == winner_model_type: continue
                            
                            other_preds = ds_forc[ds_forc['model_type'] == model].sort_values('date_index')['previsao'].values
                            
                            if len(other_preds) == len(real_vals):
                                dm_stat, p_val = calculate_dm_statistic(real_vals, winner_preds, other_preds)
                                dm_dataset_list.append({
                                    'Grupo': group,
                                    'Modelo Comparado': model,
                                    'DM Stat': f"{dm_stat:.3f}",
                                    'p-Value': f"{p_val:.2E}"
                                })
                    
                    if dm_dataset_list:
                        md_content += pd.DataFrame(dm_dataset_list).to_markdown(index=False) + "\n\n"
            except: pass

        # C. PROPOSTA VS REFERÊNCIA
        if not ds_forc.empty:
            prop_plot = generate_proposal_vs_reference_plot(ds_forc, report_plot_dir, ds)
            if prop_plot:
                md_content += f"**Destaque: Proposta vs Referência**\n\n![PropVsRef](plots/{prop_plot})\n\n"
            
            # D. PAINEL GERAL
            panel_plot = generate_actual_vs_predicted_plots(ds_forc, report_plot_dir, ds)
            if panel_plot:
                md_content += f"**Visão Geral: Real vs Previsto**\n\n![Panel](plots/{panel_plot})\n\n"
        
        md_content += "---\n"

    report_file = os.path.join(report_dir, f"relatorio_final_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(report_file, 'w', encoding='utf-8') as f: f.write(md_content)
    print(f"Relatório gerado: {report_file}")