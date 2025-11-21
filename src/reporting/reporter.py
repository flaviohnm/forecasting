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
    logging.warning(
        "Matplotlib não encontrado. Gráficos de horizonte não serão gerados no relatório."
    )


def calculate_pd(mape_base, mape_comp):
    """Calcula a Diferença Percentual (PD) entre dois valores MAPE escalares."""
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(
            mape_comp) or mape_base == 0:
        return np.nan
    # PD = 100 * (MAPE_comp - MAPE_base) / MAPE_base
    return 100 * (mape_comp - mape_base) / mape_base


def generate_horizon_plots(horizon_metrics_df,
                           output_dir,
                           dataset_name,
                           plot_suffix=""):
    """
    Gera gráficos de linha para métricas por horizonte para UM dataset e UM grupo de modelos.
    """
    if not PLOTTING_ENABLED or horizon_metrics_df.empty:
        return {}

    plot_files = {}
    metrics_to_plot = ['MAPE', 'MASE', 'RMSSE']

    horizon_metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    metrics_grouped = horizon_metrics_df.groupby(
        ['model_type', 'horizon'])[['MAPE', 'MASE',
                                    'RMSSE']].mean(numeric_only=True)

    max_horizon = horizon_metrics_df['horizon'].max()
    horizon_ticks = np.arange(1, int(max_horizon) + 1)
    model_types = horizon_metrics_df['model_type'].unique()

    for metric in metrics_to_plot:
        if metric not in metrics_grouped.columns:
            continue

        plt.figure(figsize=(10, 6))

        has_data = False
        for model_type in model_types:
            try:
                data_to_plot = metrics_grouped.loc[model_type, metric].reindex(
                    horizon_ticks)
                if not data_to_plot.isnull().all():
                    plt.plot(data_to_plot.index,
                             data_to_plot.values,
                             marker='o',
                             markersize=4,
                             linestyle='-',
                             label=model_type)
                    has_data = True
            except KeyError:
                continue

        if not has_data:
            plt.close()
            continue

        plt.xlabel("Horizonte de Previsão (h)")
        plt.ylabel(f"{metric}")
        plt.title(f"Evolução do {metric} - {dataset_name}")
        plt.legend(loc='center left',
                   bbox_to_anchor=(1.02, 0.5),
                   borderaxespad=0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(horizon_ticks)
        plt.tight_layout(rect=[0, 0, 0.80, 1])

        clean_suffix = plot_suffix.replace(" ", "_").replace(".", "").replace(
            "(", "").replace(")", "")
        plot_filename = f"plot_horizon_{dataset_name}_{metric}_{clean_suffix}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)

        try:
            plt.savefig(plot_filepath, dpi=150)
            plt.close()
            plot_files[metric] = plot_filename
            print(f"  Gráfico salvo: {plot_filename}")
        except Exception as e:
            logging.error(f"Erro ao salvar gráfico {metric}: {e}")
            plt.close()

    return plot_files


def get_specific_horizons(max_horizon):
    """Retorna uma lista de horizontes específicos para destacar com base no horizonte máximo."""
    max_horizon = int(max_horizon)
    if max_horizon >= 10: return [1, 5, 10]
    elif max_horizon >= 7: return [1, 3, 7]
    elif max_horizon >= 4: return [1, 2, 4]
    else: return [1]


def generate_report(main_config: dict, successful_executions: list):
    """
    Gera um relatório consolidado em Markdown com painéis de comparação, destaques e tabela de vencedores.
    """
    print("Iniciando a geração do relatório...")
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    comparison_results_path = os.path.join("results", "comparison_tests")
    report_dir = "reports"
    report_plot_dir = os.path.join(report_dir, "plots")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(report_plot_dir, exist_ok=True)

    all_metrics = []
    all_horizon_metrics = []

    # 1. Coleta métricas
    for dataset_conf, model_conf in successful_executions:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        metric_file = os.path.join(metrics_path,
                                   f"metrics_{execution_name}.csv")
        horizon_metric_file = os.path.join(
            metrics_path, f"metrics_horizon_{execution_name}.csv")

        comparison_group = model_conf.get('comparison_group', 'other')

        if os.path.exists(metric_file):
            try:
                df = pd.read_csv(metric_file)
                if not df.empty and all(c in df.columns
                                        for c in ['MAPE', 'MASE', 'RMSSE']):
                    df['comparison_group'] = comparison_group
                    all_metrics.append(df)
            except Exception as e:
                logging.warning(
                    f"Erro ao ler métrica geral {metric_file}: {e}")

        if os.path.exists(horizon_metric_file):
            try:
                df_h = pd.read_csv(horizon_metric_file)
                if not df_h.empty and 'horizon' in df_h.columns:
                    df_h['model_type'] = model_conf['model_type']
                    df_h['dataset'] = dataset_conf['name']
                    df_h['comparison_group'] = comparison_group
                    all_horizon_metrics.append(df_h)
            except Exception as e:
                logging.warning(
                    f"Erro lendo métrica por horizonte {horizon_metric_file}: {e}"
                )

    if not all_metrics:
        print("Nenhuma métrica geral válida encontrada.")
        return

    summary_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = summary_df.sort_values(by=['dataset', 'MAPE'], ascending=True)

    # --- CORREÇÃO AQUI: Adicionada a definição de sorted_df_for_plots ---
    sorted_df_for_plots = summary_df.copy()
    # -------------------------------------------------------------------

    horizon_df_all = pd.DataFrame()
    if all_horizon_metrics:
        horizon_df_all = pd.concat(all_horizon_metrics, ignore_index=True)
        horizon_df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# Relatório de Experimentos de Forecasting

Gerado em: {now}

**Reprodutibilidade:** Todos os modelos estocásticos foram executados com `seed=42`.

## Painéis de Comparação por Abordagem

Resultados agrupados por dataset e estratégia de modelo. 
**Destaque:** O modelo com o menor MAPE em cada tabela está em **negrito**.
"""

    datasets_processed = sorted(summary_df['dataset'].unique())

    benchmark_groups = ['benchmark_statistical', 'benchmark_standalone_dl']

    comparison_panels = {
        "Híbridos (MIMO) vs. Benchmarks":
        benchmark_groups + ['hybrid_mimo'],
        "Híbridos (Recursive) vs. Benchmarks":
        benchmark_groups + ['hybrid_recursive'],
        "Híbridos (Direct) vs. Benchmarks":
        benchmark_groups + ['hybrid_direct']
    }

    # --- LOOP DE PAINÉIS ---
    for panel_title, groups_to_include in comparison_panels.items():
        report_content += f"\n### Painel: {panel_title}\n"

        panel_df = summary_df[summary_df['comparison_group'].isin(
            groups_to_include)]
        panel_horizon_df_full = pd.DataFrame()
        if not horizon_df_all.empty:
            panel_horizon_df_full = horizon_df_all[
                horizon_df_all['comparison_group'].isin(groups_to_include)]

        if panel_df.empty:
            report_content += "\n*Nenhum resultado encontrado para este painel.*\n"
            continue

        for ds_name in datasets_processed:
            if ds_name not in panel_df['dataset'].unique():
                continue

            report_content += f"\n#### Dataset: {ds_name}\n\n"

            ds_df = panel_df[panel_df['dataset'] == ds_name].copy()

            base_mape = ds_df['MAPE'].min()
            best_idx = ds_df['MAPE'].idxmin()

            ds_df['PD_vs_Best (%)'] = ds_df['MAPE'].apply(lambda x: 100 * (
                x - base_mape) / base_mape if base_mape > 0 else 0)

            ds_df_display = ds_df.copy()
            ds_df_display['MAPE'] = ds_df_display['MAPE'].map('{:.4f}'.format)
            ds_df_display['MASE'] = ds_df_display['MASE'].map('{:.4f}'.format)
            ds_df_display['RMSSE'] = ds_df_display['RMSSE'].map(
                '{:.4f}'.format)
            ds_df_display['PD_vs_Best (%)'] = ds_df_display[
                'PD_vs_Best (%)'].apply(lambda x: f"{x:+.2f}"
                                        if pd.notna(x) and x != 0 else '-')

            ds_df_display.loc[best_idx, 'PD_vs_Best (%)'] = '-'

            cols_to_bold = [
                'execution_name', 'model_type', 'MAPE', 'MASE', 'RMSSE'
            ]
            for col in cols_to_bold:
                if col in ds_df_display.columns:
                    ds_df_display.loc[
                        best_idx,
                        col] = f"**{ds_df_display.loc[best_idx, col]}**"

            cols_order = [
                'execution_name', 'model_type', 'comparison_group', 'MAPE',
                'MASE', 'RMSSE', 'PD_vs_Best (%)'
            ]
            cols_order = [
                col for col in cols_order if col in ds_df_display.columns
            ]
            report_content += ds_df_display[cols_order].to_markdown(
                index=False) + "\n\n"

            # 2. Gráficos de Horizonte
            if not panel_horizon_df_full.empty:
                ds_panel_horizon = panel_horizon_df_full[
                    panel_horizon_df_full['dataset'] == ds_name].copy()

                if not ds_panel_horizon.empty:
                    plot_suffix = f"{panel_title}_{ds_name}"
                    plot_files = generate_horizon_plots(
                        ds_panel_horizon, report_plot_dir, ds_name,
                        plot_suffix)

                    if plot_files:
                        report_content += "**Evolução do Erro por Horizonte:**\n\n"
                        for metric, filename in plot_files.items():
                            relative_path = os.path.join("plots",
                                                         filename).replace(
                                                             "\\", "/")
                            report_content += f"![{metric} - {panel_title}]({relative_path})\n"

            report_content += "\n---\n"

    # --- NOVA SEÇÃO: RESUMO DOS VENCEDORES ---
    report_content += "\n## Resumo dos Vencedores por Dataset\n"
    report_content += "Modelo com o menor MAPE encontrado entre **todas** as estratégias testadas.\n\n"

    try:
        winners_idx = summary_df.groupby('dataset')['MAPE'].idxmin()
        winners_df = summary_df.loc[winners_idx].sort_values('dataset').copy()

        winners_df['MAPE'] = winners_df['MAPE'].map('{:.4f}'.format)
        winners_df['MASE'] = winners_df['MASE'].map('{:.4f}'.format)
        winners_df['RMSSE'] = winners_df['RMSSE'].map('{:.4f}'.format)

        cols_winner = [
            'dataset', 'execution_name', 'model_type', 'comparison_group',
            'MAPE', 'MASE', 'RMSSE'
        ]
        report_content += winners_df[cols_winner].to_markdown(
            index=False) + "\n\n"

    except Exception as e:
        logging.error(f"Erro ao gerar tabela de vencedores: {e}")
        report_content += "*Erro ao gerar tabela de vencedores.*\n"

    # --- TESTES ESTATÍSTICOS ---
    report_content += "\n## Testes de Comparação Estatística\n"

    dm_summary_file = os.path.join(comparison_results_path,
                                   "diebold_mariano_summary.csv")
    dm_content = "\n*Resultados do teste Diebold-Mariano não encontrados.*\n"
    if os.path.exists(dm_summary_file):
        try:
            dm_df = pd.read_csv(dm_summary_file)
            dm_df['p_value'] = dm_df['p_value'].map('{:.4f}'.format)
            dm_df['dm_statistic'] = dm_df['dm_statistic'].map('{:.3f}'.format)
            dm_content = "\n### Teste Diebold-Mariano (Comparação Par-a-Par)\n*Compara o melhor modelo contra os outros por dataset.*\n\n"
            for dataset_name in sorted(dm_df['dataset'].unique()):
                dm_content += f"**Dataset: {dataset_name}**\n" + dm_df[
                    dm_df['dataset'] == dataset_name].to_markdown(
                        index=False) + "\n\n"
        except Exception as e:
            dm_content = f"\n*Erro DM: {e}*\n"
    report_content += dm_content

    friedman_file = os.path.join(comparison_results_path,
                                 "friedman_test_summary.csv")
    nemenyi_file = os.path.join(comparison_results_path,
                                "nemenyi_posthoc_pvalues.csv")
    friedman_content = "\n*Resultados Friedman não encontrados.*\n"
    if os.path.exists(friedman_file):
        try:
            friedman_df = pd.read_csv(friedman_file)
            friedman_content = "\n### Teste de Friedman e Nemenyi (Global)\n"
            friedman_content += "**Friedman:**\n" + friedman_df.to_markdown(
                index=False) + "\n\n"
            if os.path.exists(
                    nemenyi_file) and not friedman_df.empty and friedman_df[
                        'significativo (p<0.05)'].iloc[0]:
                nemenyi_df = pd.read_csv(nemenyi_file, index_col=0)
                friedman_content += "**Post-Hoc Nemenyi (p-valores):**\n" + nemenyi_df.round(
                    4).to_markdown() + "\n\n"
        except Exception as e:
            friedman_content = f"\n*Erro Friedman: {e}*\n"
    report_content += friedman_content

    # --- GRÁFICOS INDIVIDUAIS (Apêndice) ---
    report_content += "\n## Gráficos Detalhados por Execução (Completo - Apêndice)\n"
    for index, row in sorted_df_for_plots.iterrows():
        exec_name = row['execution_name']
        plot_filename = f"plot_{exec_name}.png"
        relative_plot_path = os.path.join('..', 'results', 'plots',
                                          plot_filename).replace("\\", "/")
        plot_abs_path = os.path.join(main_config['results_paths']['plots'],
                                     plot_filename)
        if os.path.exists(plot_abs_path):
            plot_markdown = f"![Gráfico para {exec_name}]({relative_plot_path})"
            report_content += f"\n### Execução: {exec_name}\n\n{plot_markdown}\n"

    # Salva o relatório
    report_filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_filepath = os.path.join(report_dir, report_filename)
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Relatório gerado com sucesso em: {report_filepath}")
