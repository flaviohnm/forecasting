# File: src/reporting/visualizer.py
import os
import numpy as np
import math
import logging
from . import statistics # Importa o módulo de estatística

# Configuração do Matplotlib
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

def _save_plot(output_dir, filename):
    try:
        plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico salvo: {filename}")
    except Exception as e:
        logging.error(f"Erro ao salvar {filename}: {e}")
        plt.close()

# --- VISUALIZAÇÃO 3: CD DIAGRAM (SEU CÓDIGO ADAPTADO) ---
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

    # 2. Calcular CD (Chama a função do módulo statistics)
    cd = statistics.get_nemenyi_cd(k, N)

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
    
    ax.set_title("Diagrama de Diferença Crítica (Nemenyi, α=0.05)", fontsize=13, y=1.15)

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
    
    text_x_left = max_rank + 0.5
    
    for idx, (rank, name) in enumerate(left_models):
        y_pos = text_y_start - (idx * 0.6)
        
        # 1. Linha Vertical
        ax.plot([rank, rank], [0, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 2. Linha Horizontal
        ax.plot([rank, text_x_left], [y_pos, y_pos], color='black', linewidth=0.8, alpha=0.6)
        
        # 3. Texto
        ax.text(text_x_left, y_pos, f"({rank:.2f}) {name} ", ha='right', va='center', fontsize=9)

    plt.tight_layout()
    
    filename = "cd_diagram.png"
    _save_plot(output_dir, filename)
    return filename

# --- OUTRAS VISUALIZAÇÕES (MANTIDAS) ---

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
    plt.title("Distribuição dos Rankings de Cada Modelo", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    max_rank = df_ranked['rank'].max()
    plt.yticks(np.arange(1, max_rank + 2, 1))
    plt.tight_layout()

    filename = "boxplot_rankings_global.png"
    _save_plot(output_dir, filename)
    return filename

def generate_pd_plot(metrics_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or metrics_df.empty: return None
    
    df = metrics_df.copy()
    best = df['MAPE'].min()
    df['PD'] = 100 * (df['MAPE'] - best) / best
    df = df.sort_values('PD')
    
    group_colors = {
        'benchmark_statistical': '#1f77b4', 'benchmark_standalone_dl': '#ff7f0e', 
        'hybrid_recursive': '#2ca02c', 'hybrid_direct': '#d62728', 'hybrid_mimo': '#9467bd', 'other': '#7f7f7f'
    }
    bar_c = [group_colors.get(g, 'gray') for g in df['comparison_group']]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(df)), df['PD'], color=bar_c, alpha=0.8)
    plt.title(f"PD (%) do MAPE - {dataset_name}", fontsize=14)
    plt.xticks(range(len(df)), df['model_type'], rotation=45, ha='right')
    plt.ylabel("PD (%)")
    
    handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items() if g in df['comparison_group'].unique()]
    plt.legend(handles=handles, title="Grupo")
    
    for bar in bars:
        h = bar.get_height()
        if h > 0: plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    filename = f"pd_plot_{dataset_name}.png"
    _save_plot(output_dir, filename)
    return filename

def generate_proposal_vs_reference_plot(forecasts_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or forecasts_df.empty: return None
    
    PROPOSED, REF = "Hybrid_MIMO_NBEATS_NF", "Hybrid_Direct_NBEATS_NF"
    subset = forecasts_df[forecasts_df['model_type'].isin([PROPOSED, REF])].sort_values('date_index')
    if subset.empty: return None
    
    real = subset[['date_index', 'real']].drop_duplicates().set_index('date_index')
    
    plt.figure(figsize=(12, 6))
    plt.plot(real.index, real['real'], 'k--', lw=2, label='Real')
    for m, c, mk in [(REF, 'orange', 'x'), (PROPOSED, 'blue', 'o')]:
        d = subset[subset['model_type'] == m].set_index('date_index')
        if not d.empty: plt.plot(d.index, d['previsao'], color=c, marker=mk, label=m)

    plt.title(f"Comparativo: Proposta vs Referência - {dataset_name}")
    plt.legend(); plt.grid(True, ls=':')
    plt.tight_layout()
    
    filename = f"proposal_vs_ref_{dataset_name}.png"
    _save_plot(output_dir, filename)
    return filename

def generate_actual_vs_predicted_plots(forecasts_df, output_dir, dataset_name):
    if not PLOTTING_ENABLED or forecasts_df.empty: return None
    
    groups = sorted(forecasts_df['comparison_group'].unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4*len(groups)), sharex=True)
    if len(groups) == 1: axes = [axes]
    
    real = forecasts_df.sort_values('date_index')[['date_index', 'real']].drop_duplicates().set_index('date_index')
    
    for ax, grp in zip(axes, groups):
        dat = forecasts_df[forecasts_df['comparison_group'] == grp].sort_values('date_index')
        ax.plot(real.index, real['real'], 'k--', label='Real')
        for m in dat['model_type'].unique():
            d = dat[dat['model_type'] == m].set_index('date_index')
            ax.plot(d.index, d['previsao'], marker='.', label=m, alpha=0.7)
        ax.set_title(grp); ax.legend(bbox_to_anchor=(1, 1)); ax.grid(True)
        
    plt.tight_layout()
    filename = f"forecast_panel_{dataset_name}.png"
    _save_plot(output_dir, filename)
    return filename