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
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    logging.warning("Matplotlib não encontrado. Gráficos não serão gerados.")


def calculate_pd(mape_base, mape_comp):
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(mape_comp) or mape_base == 0:
        return np.nan
    return 100 * (mape_comp - mape_base) / mape_base

# --- NOVA FUNÇÃO: PLOT PROPOSTA VS REFERÊNCIA ---
def generate_proposal_vs_reference_plot(forecasts_df, output_dir, dataset_name):
    """
    Gera um gráfico focado comparando apenas a Proposta (MIMO) e a Referência (Direct)
    contra o Real.
    """
    if not PLOTTING_ENABLED or forecasts_df.empty:
        return None

    # Definição dos modelos de interesse
    PROPOSED_MODEL = "Hybrid_MIMO_NBEATS_NF"
    REFERENCE_MODEL = "Hybrid_Direct_NBEATS_NF"
    
    # Filtra apenas os dados desses modelos
    subset_df = forecasts_df[forecasts_df['model_type'].isin([PROPOSED_MODEL, REFERENCE_MODEL])].copy()
    
    # Se não tivermos dados de nenhum dos dois, não gera o gráfico
    if subset_df.empty:
        return None
        
    # Ordena por data para garantir linhas contínuas
    subset_df = subset_df.sort_values('date_index')
    
    # Pega dados reais de um dos modelos (assumindo consistência no dataset)
    # Se o modelo proposto não existir, tenta pegar do referência para extrair o real
    if not subset_df[subset_df['model_type'] == PROPOSED_MODEL].empty:
        base_model_for_real = PROPOSED_MODEL
    else:
        base_model_for_real = REFERENCE_MODEL
        
    real_data = subset_df[subset_df['model_type'] == base_model_for_real][['date_index', 'real']].drop_duplicates().set_index('date_index')
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plota Real
    plt.plot(real_data.index, real_data['real'], color='black', linestyle='--', linewidth=2, label='Valor Real', alpha=0.6)
    
    # 2. Plota Referência (Direct)
    ref_data = subset_df[subset_df['model_type'] == REFERENCE_MODEL].set_index('date_index')
    if not ref_data.empty:
        plt.plot(ref_data.index, ref_data['previsao'], color='#ff7f0e', marker='x', markersize=6, linestyle='-', linewidth=1.5, label=f'Ref: {REFERENCE_MODEL}', alpha=0.9)

    # 3. Plota Proposta (MIMO)
    prop_data = subset_df[subset_df['model_type'] == PROPOSED_MODEL].set_index('date_index')
    if not prop_data.empty:
        plt.plot(prop_data.index, prop_data['previsao'], color='#1f77b4', marker='o', markersize=5, linestyle='-', linewidth=2, label=f'Prop: {PROPOSED_MODEL}', alpha=0.9)

    plt.title(f"Comparativo: Proposta vs Referência - {dataset_name}", fontsize=14, pad=15)
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    filename = f"proposal_vs_ref_{dataset_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        plt.savefig(filepath, dpi=120)
        plt.close()
        print(f"  Gráfico Proposta vs Referência salvo: {filename}")
        return filename
    except Exception as e:
        logging.error(f"Erro ao salvar gráfico destaque {dataset_name}: {e}")
        plt.close()
        return None

def generate_actual_vs_predicted_plots(forecasts_df, output_dir, dataset_name):
    """
    Gera um painel de gráficos (subplots) comparando REAL x PREVISTO por grupo.
    """
    if not PLOTTING_ENABLED or forecasts_df.empty: return None

    order_precedence = {
        'benchmark_statistical': 0,
        'benchmark_standalone_dl': 1,
        'hybrid_direct': 2,
        'hybrid_mimo': 3,
        'hybrid_recursive': 4
    }
    
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

    plot_filename = f"forecast_panel_{dataset_name}.png"
    
    for ax, group in zip(axes, sorted_groups):
        group_data = forecasts_df[forecasts_df['comparison_group'] == group]
        
        ax.plot(real_data.index, real_data['real'], color='black', linestyle='--', linewidth=2, label='REAL', alpha=0.6)
        
        models_in_group = sorted(group_data['model_type'].unique())
        
        for model in models_in_group:
            model_subset = group_data[group_data['model_type'] == model].set_index('date_index')
            if not model_subset.empty:
                ax.plot(model_subset.index, model_subset['previsao'], marker='.', markersize=4, linewidth=1.5, label=model, alpha=0.8)
        
        ax.set_title(f"Grupo: {group}", fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylabel("Valor")
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')

    axes[-1].set_xlabel("Data")
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    
    plot_filepath = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_filepath, dpi=100)
        plt.close()
        return plot_filename
    except Exception as e:
        plt.close()
        return None

def generate_report(main_config: dict, successful_executions: list):
    """
    Gera um relatório consolidado.
    """
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
                    cols_to_keep = ['date_index', 'real', 'previsao', 'model_type', 'dataset', 'comparison_group']
                    cols_exist = [c for c in cols_to_keep if c in df_f.columns]
                    all_forecasts.append(df_f[cols_exist])
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
    md_content = f"# Relatório de Experimentos de Forecasting\n\nGerado em: {now}\n\n"
    
    # --- SEÇÃO 1: VENCEDORES (ORDEM CORRIGIDA: GRUPO | MODELO | DATASET | MÉTRICAS) ---
    md_content += "## Resumo dos Campeões (Menor MAPE)\n\n"
    try:
        winners_idx = summary_df.groupby('dataset')['MAPE'].idxmin()
        # Seleciona e ordena as colunas na ordem desejada
        winners = summary_df.loc[winners_idx].sort_values('dataset')[['comparison_group', 'model_type', 'dataset', 'MAPE', 'MASE', 'RMSSE']]
        
        winners_formatted = winners.copy()
        # Renomeia
        winners_formatted.columns = ['Grupo', 'Modelo', 'Dataset', 'MAPE', 'MASE', 'RMSSE']
        
        for col in ['MAPE', 'MASE', 'RMSSE']:
            winners_formatted[col] = winners_formatted[col].apply(lambda x: f"**{x:.4f}**")
        
        md_content += winners_formatted.to_markdown(index=False)
    except Exception as e: 
        logging.error(f"Erro tabela vencedores: {e}")
        pass
    md_content += "\n\n---\n\n"

    # --- SEÇÃO 2: DETALHES POR DATASET ---
    datasets = sorted(summary_df['dataset'].unique())
    metrics_to_show = ['MAPE', 'MASE', 'RMSSE']
    sort_mapping = {'benchmark_statistical': 0, 'benchmark_standalone_dl': 1, 'hybrid_direct': 2, 'hybrid_mimo': 3, 'hybrid_recursive': 4}

    for ds in datasets:
        md_content += f"## Dataset: {ds}\n\n"
        
        ds_forecasts = pd.DataFrame()
        if not forecasts_df_all.empty:
            ds_forecasts = forecasts_df_all[forecasts_df_all['dataset'] == ds]

        # A. Gráfico Destaque: Proposta vs Referência (NOVO)
        if not ds_forecasts.empty:
            prop_ref_plot = generate_proposal_vs_reference_plot(ds_forecasts, report_plot_dir, ds)
            if prop_ref_plot:
                md_content += "### Comparativo: Proposta vs Referência\n"
                md_content += f"Comparação direta entre o modelo proposto (**Hybrid_MIMO_NBEATS_NF**) e a referência (**Hybrid_Direct_NBEATS_NF**).\n\n"
                md_content += f"![Proposta vs Referência](plots/{prop_ref_plot})\n\n"

        # B. Painel Geral: Real vs Previsto por Grupo
        if not ds_forecasts.empty:
            plot_file = generate_actual_vs_predicted_plots(ds_forecasts, report_plot_dir, ds)
            if plot_file:
                md_content += "### Visão Geral: Real vs Previsto (Todos os Grupos)\n\n"
                md_content += f"![Painel Real vs Previsto](plots/{plot_file})\n\n"
        
        # C. Tabelas de Ranking
        ds_df = summary_df[summary_df['dataset'] == ds].copy()
        
        for metric in metrics_to_show:
            if metric not in ds_df.columns: continue
            
            md_content += f"### Tabela de Resultados: {metric}\n"
            
            best_val_global = ds_df[metric].min()
            
            ds_df['sort_key'] = ds_df['comparison_group'].map(sort_mapping).fillna(99)
            sorted_df = ds_df.sort_values(by=['sort_key', 'model_type']).reset_index(drop=True)
            
            # Seleciona na ordem correta
            view_df = sorted_df[['comparison_group', 'model_type', 'dataset', metric]].copy()
            view_df.columns = ['Grupo', 'Modelo', 'Dataset', metric]
            
            view_df[metric] = view_df[metric].apply(lambda x: f"{x:.4f}")
            
            def highlight_winner(row):
                idx = row.name
                val_float = sorted_df.iloc[idx][metric]
                if abs(val_float - best_val_global) < 1e-9:
                    return [f"**{cell}**" for cell in row]
                return row.tolist()

            formatted_data = view_df.apply(highlight_winner, axis=1)
            final_table_df = pd.DataFrame(formatted_data.tolist(), columns=view_df.columns)
            md_content += final_table_df.to_markdown(index=False)
            md_content += "\n\n"
        
        md_content += "---\n\n"

    report_file = os.path.join(report_dir, f"relatorio_final_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Relatório gerado em: {report_file}")