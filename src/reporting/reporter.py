# File: src/reporting/reporter.py

import os
import pandas as pd
from datetime import datetime

def generate_report(main_config: dict, executions: list):
    """
    Gera um relatório consolidado em Markdown com base nos resultados
    das execuções bem-sucedidas.
    """
    print("Iniciando a geração do relatório...")
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    
    all_metrics = []

    # 1. Coleta todas as métricas dos arquivos CSV gerados com sucesso
    for dataset_conf, model_conf in executions:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        metric_file = os.path.join(metrics_path, f"metrics_{execution_name}.csv")
        
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file)
            all_metrics.append(df)

    if not all_metrics:
        print("Nenhum arquivo de métrica encontrado. Execute a etapa 'evaluate' primeiro.")
        return

    # 2. Consolida e formata o DataFrame de resumo
    summary_df = pd.concat(all_metrics, ignore_index=True)
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Ordena primeiro por 'dataset' e depois por 'MAPE'
    summary_df = summary_df.sort_values(by=['dataset', 'MAPE'], ascending=True)
    
    # Salva uma cópia ordenada antes de formatar as strings para o Markdown
    sorted_df_for_plots = summary_df.copy()

    # Formatação para melhor visualização no Markdown
    summary_df['MAPE'] = summary_df['MAPE'].map('{:.4f}'.format)
    summary_df['MASE'] = summary_df['MASE'].map('{:.4f}'.format)

    # 3. Constrói o conteúdo do arquivo Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# Relatório de Experimentos de Forecasting

Gerado em: {now}

## Resumo Geral das Métricas

A tabela abaixo consolida o desempenho de todas as execuções bem-sucedidas, agrupadas por dataset e ordenadas pelo menor MAPE.

{summary_df.to_markdown(index=False)}

## Análise dos Resultados

> (Espaço para sua análise. Ex: O modelo 'standalone_ets' apresentou o melhor desempenho geral, enquanto os modelos Keras não performaram bem no dataset 'sunspot'...)



## Gráficos Detalhados por Execução
"""

    # 4. Adiciona uma seção para cada execução bem-sucedida, usando a ordem do DF ordenado
    for index, row in sorted_df_for_plots.iterrows():
        exec_name = row['execution_name']
        
        # Cria a tabela de métrica individual (usando os valores já formatados do summary_df)
        individual_table_row = summary_df[summary_df['execution_name'] == exec_name]
        individual_table = individual_table_row.to_markdown(index=False)
        
        # Monta o caminho relativo para a imagem do gráfico
        plot_filename = f"plot_{exec_name}.png"
        relative_plot_path = os.path.join('..', 'results', 'plots', plot_filename).replace("\\", "/")
        
        report_content += f"""
### Execução: {exec_name}

{individual_table}

![Gráfico para {exec_name}]({relative_plot_path})
"""

    # 5. Salva o relatório final
    report_filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_filepath = os.path.join("reports", report_filename)
    
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Relatório gerado com sucesso em: {report_filepath}")