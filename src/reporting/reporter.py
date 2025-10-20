# File: src/reporting/reporter.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Optional: Configure logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_pd(mape_base, mape_comp):
    """Calcula a Diferença Percentual (PD) entre dois valores MAPE."""
    if mape_base is None or mape_comp is None or pd.isna(mape_base) or pd.isna(mape_comp) or mape_base == 0:
        return np.nan
    return 100 * (mape_base - mape_comp) / mape_base

def generate_report(main_config: dict, successful_executions: list):
    """
    Gera um relatório consolidado em Markdown com base nos resultados
    das execuções bem-sucedidas e nos testes de comparação.
    """
    print("Iniciando a geração do relatório...")
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    comparison_results_path = os.path.join("results", "comparison_tests")
    
    all_metrics = []
    # Usa successful_executions que vem do main.py (lista de tuplas)
    for dataset_conf, model_conf in successful_executions:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        metric_file = os.path.join(metrics_path, f"metrics_{execution_name}.csv")
        
        if os.path.exists(metric_file):
            try:
                df = pd.read_csv(metric_file)
                # Verifica se tem MAPE, MASE e RMSSE e não são NaN
                if not df.empty and all(col in df.columns for col in ['MAPE', 'MASE', 'RMSSE']) and not df[['MAPE', 'MASE', 'RMSSE']].isnull().any().any():
                    all_metrics.append(df)
                else:
                     logging.warning(f"Relatório: Arquivo de métrica '{execution_name}' vazio, com NaNs ou faltando colunas MAPE/MASE/RMSSE.")
            except Exception as e:
                logging.warning(f"Relatório: Erro ao ler arquivo de métrica {metric_file}: {e}")

    if not all_metrics:
        print("Nenhuma métrica válida encontrada. Relatório não pode ser gerado.")
        return

    # 2. Consolida e formata o DataFrame de resumo
    summary_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = summary_df.sort_values(by=['dataset', 'MAPE'], ascending=True)
    
    sorted_df_for_plots = summary_df.copy() # Guarda cópia ANTES de formatar

    def calculate_pd_for_group(x):
        base_mape = x.min()
        return x.apply(lambda mape_comp: calculate_pd(base_mape, mape_comp))

    summary_df['PD_vs_Best (%)'] = summary_df.groupby('dataset')['MAPE'].transform(calculate_pd_for_group)
    
    best_indices = summary_df.groupby('dataset')['MAPE'].idxmin()
    summary_df['PD_vs_Best (%)'] = summary_df['PD_vs_Best (%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
    summary_df.loc[best_indices, 'PD_vs_Best (%)'] = '-'
    
    # Formatação das métricas principais
    summary_df['MAPE'] = summary_df['MAPE'].map('{:.4f}'.format)
    summary_df['MASE'] = summary_df['MASE'].map('{:.4f}'.format)
    summary_df['RMSSE'] = summary_df['RMSSE'].map('{:.4f}'.format)

    # Reordena colunas para melhor visualização
    cols_order = ['execution_name', 'model_type', 'dataset', 'MAPE', 'MASE', 'RMSSE', 'PD_vs_Best (%)']
    # Garante que só reordena se todas as colunas existirem
    cols_order = [col for col in cols_order if col in summary_df.columns]
    summary_df = summary_df[cols_order]


    # 3. Constrói o conteúdo do arquivo Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# Relatório de Experimentos de Forecasting

Gerado em: {now}

**Reprodutibilidade:** Todos os modelos estocásticos foram executados com `seed=42`.

## Resumo Geral das Métricas

A tabela abaixo consolida o desempenho (MAPE, MASE, RMSSE) de todas as execuções bem-sucedidas, agrupadas por dataset e ordenadas pelo menor MAPE. A coluna 'PD_vs_Best (%)' mostra a melhoria percentual em relação ao melhor modelo (menor MAPE) para aquele dataset.

{summary_df.to_markdown(index=False)}

## Análise dos Resultados

> (Espaço para sua análise...)

## Testes de Comparação Estatística

### Teste Diebold-Mariano (Comparação Par-a-Par por Dataset)
- **H0:** Mesma acurácia.
- **H1:** Acurácia diferente.
- **p < 0.05:** Diferença estatisticamente significativa.

"""

    # 4. Carrega e adiciona os resultados do Diebold-Mariano
    dm_summary_file = os.path.join(comparison_results_path, "diebold_mariano_summary.csv")
    if os.path.exists(dm_summary_file):
        try:
            dm_df = pd.read_csv(dm_summary_file)
            dm_df['p_value'] = dm_df['p_value'].map('{:.4f}'.format)
            dm_df['dm_statistic'] = dm_df['dm_statistic'].map('{:.3f}'.format)
            for dataset_name in dm_df['dataset'].unique():
                report_content += f"#### Dataset: {dataset_name}\n\n"
                report_content += dm_df[dm_df['dataset'] == dataset_name].to_markdown(index=False) + "\n\n"
        except Exception as e: report_content += f"\n*AVISO: Erro ao carregar/formatar resultados DM: {e}*\n"
    else: report_content += "\n*Resultados do teste Diebold-Mariano não encontrados.*\n"

    # Carrega resultados Friedman/Nemenyi
    report_content += "\n### Teste de Friedman e Nemenyi (Comparação Global)\n"
    report_content += "Compara os rankings médios de desempenho (MAPE) de todos os modelos em todos os datasets onde todos completaram com sucesso.\n"
    report_content += "- **Friedman H0:** Não há diferença entre os rankings médios.\n"
    report_content += "- **Nemenyi H0 (post-hoc):** Não há diferença entre um par específico.\n\n"

    friedman_file = os.path.join(comparison_results_path, "friedman_test_summary.csv")
    nemenyi_file = os.path.join(comparison_results_path, "nemenyi_posthoc_pvalues.csv")
    if os.path.exists(friedman_file):
        try:
            friedman_df = pd.read_csv(friedman_file)
            report_content += "**Resultado do Teste de Friedman:**\n\n" + friedman_df.to_markdown(index=False) + "\n\n"
            if os.path.exists(nemenyi_file) and friedman_df['significativo (p<0.05)'].iloc[0]:
                 report_content += "**Teste Post-Hoc de Nemenyi (Matriz de p-valores):**\n" + "*Valores < 0.05 indicam diferença significativa.*\n\n"
                 nemenyi_df = pd.read_csv(nemenyi_file, index_col=0)
                 report_content += nemenyi_df.round(4).to_markdown() + "\n\n"
            elif not friedman_df['significativo (p<0.05)'].iloc[0]:
                 report_content += "*Teste de Friedman não foi significativo, Nemenyi não aplicável.*\n\n"
            else: report_content += "*Arquivo Nemenyi não encontrado.*\n\n"
        except Exception as e: report_content += f"\n*AVISO: Erro ao carregar/formatar resultados Friedman/Nemenyi: {e}*\n"
    else:
        report_content += "\n*Resultados do teste de Friedman não encontrados.*\n"

    # 5. Adiciona gráficos
    report_content += "\n## Gráficos Detalhados por Execução\n"
    # Usa sorted_df_for_plots que tem os dados originais ordenados
    for index, row in sorted_df_for_plots.iterrows():
        exec_name = row['execution_name']
        
        # Pega a linha formatada correspondente do summary_df para a tabela
        individual_table_row = summary_df[summary_df['execution_name'] == exec_name]
        # Seleciona e ordena colunas para a tabela individual
        cols_to_show = ['execution_name', 'model_type', 'dataset', 'MAPE', 'MASE', 'RMSSE']
        # Garante que só seleciona colunas que existem
        cols_to_show = [col for col in cols_to_show if col in individual_table_row.columns]
        individual_table = individual_table_row[cols_to_show].to_markdown(index=False)
        
        plot_filename = f"plot_{exec_name}.png"
        relative_plot_path = os.path.join('..', 'results', 'plots', plot_filename).replace("\\", "/")
        plot_abs_path = os.path.join(main_config['results_paths']['plots'], plot_filename)
        plot_markdown = f"![Gráfico para {exec_name}]({relative_plot_path})" if os.path.exists(plot_abs_path) else f"*Gráfico para {exec_name} não encontrado.*"
        report_content += f"\n### Execução: {exec_name}\n\n{individual_table}\n\n{plot_markdown}\n"

    # 6. Salva o relatório final
    report_filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_filepath = os.path.join("reports", report_filename)
    with open(report_filepath, 'w', encoding='utf-8') as f: f.write(report_content)
    print(f"Relatório gerado com sucesso em: {report_filepath}")