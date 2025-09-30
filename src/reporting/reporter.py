# File: src/reporting/reporter.py

import os
import pandas as pd
from datetime import datetime
import glob
import argparse

def generate_report(main_config: dict, model_params: dict, args: argparse.Namespace):
    """
    Coleta todos os resultados das métricas e gráficos e gera um relatório
    em Markdown.
    """
    print("Iniciando a geração do relatório...")
    
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    reports_path = "reports" # Salvaremos os relatórios na pasta raiz do projeto

    os.makedirs(reports_path, exist_ok=True)

    # --- 1. Coletar e Consolidar as Métricas ---
    metric_files = glob.glob(os.path.join(metrics_path, "metrics_*.csv"))
    
    if not metric_files:
        print("Nenhum arquivo de métrica encontrado. Execute a etapa 'evaluate' primeiro.")
        return

    all_metrics_df = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
    # Ordena os resultados pelo MAPE para destacar os melhores modelos
    all_metrics_df = all_metrics_df.sort_values(by="MAPE", ascending=True).round(4)

    # --- 2. Construir o Conteúdo do Markdown ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = [
        f"# Relatório de Experimentos de Forecasting\n",
        f"Gerado em: {now}\n",
        "## Resumo Geral das Métricas\n",
        "A tabela abaixo consolida o desempenho de todas as execuções, ordenadas pelo menor MAPE.\n",
        all_metrics_df.to_markdown(index=False),
        "\n",
        "## Análise dos Resultados\n",
        "> (Espaço para sua análise. Ex: O modelo 'hybrid_arima_nbeats' apresentou o melhor desempenho para o dataset 'airline', "
        "sugerindo que a componente não-linear dos resíduos foi capturada com sucesso...)\n",
        "\n",
        "## Gráficos Detalhados por Execução\n",
    ]

    # --- 3. Adicionar os Gráficos ao Relatório ---
    for index, row in all_metrics_df.iterrows():
        execution_name = row['execution_name']
        plot_filename = f"plot_{execution_name}.png"
        # O caminho no markdown deve ser relativo à localização do arquivo .md
        relative_plot_path = os.path.join("..", plots_path, plot_filename) 

        report_content.append(f"### Execução: {execution_name}\n")
        
        # Adiciona a tabela de métrica individual
        report_content.append(row.to_frame().T.to_markdown(index=False) + "\n")
        
        # Adiciona a imagem
        if os.path.exists(os.path.join(plots_path, plot_filename)):
            report_content.append(f"![Gráfico para {execution_name}]({relative_plot_path})\n")
        else:
            report_content.append("*Gráfico não encontrado.*\n")

    # --- 4. Salvar o Arquivo de Relatório ---
    report_filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_filepath = os.path.join(reports_path, report_filename)

    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    print(f"SUCESSO: Relatório gerado e salvo em: {report_filepath}")