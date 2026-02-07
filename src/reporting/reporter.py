import os
import glob
import pandas as pd
import logging

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# -----------------------------------

def generate_report(main_config, successful_runs):
    """
    Consolida métricas e gera o relatorio_final.md.
    """
    logging.info("--- [REPORT] Gerando Relatório Consolidado ---")
    
    metrics_path = main_config['results_paths']['metrics']
    reports_path = main_config['results_paths']['reports']
    os.makedirs(reports_path, exist_ok=True)
    
    # 1. Carregar todos os CSVs de métricas
    all_files = glob.glob(os.path.join(metrics_path, "*.csv"))
    if not all_files:
        logging.warning("Nenhum arquivo de métrica encontrado.")
        return

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            logging.error(f"Erro ao ler {filename}: {e}")
            
    if not df_list:
        return

    df_full = pd.concat(df_list, ignore_index=True)
    
    # 2. Ordenar Ranking (Melhor MASE primeiro)
    df_full = df_full.sort_values(by=['dataset', 'horizon', 'mase'], ascending=[True, True, True])
    
    # 3. Salvar CSV Consolidado
    csv_output = os.path.join(reports_path, "consolidated_metrics.csv")
    df_full.to_csv(csv_output, index=False)
    
    # 4. Gerar Markdown (Tabela Bonita)
    md_output = os.path.join(reports_path, "relatorio_final.md")
    
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write("# Relatório Final de Performance: Time Series\n\n")
        f.write("## 1. Ranking Global de Modelos (MASE)\n")
        f.write("A tabela abaixo mostra a performance dos modelos ordenados pelo MASE (menor é melhor).\n\n")
        
        # Gera tabela Markdown via Pandas (requer tabulate instalado)
        try:
            markdown_table = df_full.to_markdown(index=False)
            f.write(markdown_table)
        except ImportError:
            f.write("ERRO: Biblioteca 'tabulate' não instalada. Tabela omitida.")
            f.write("\n\nCSV bruto disponível em: consolidated_metrics.csv")
        
        f.write("\n\n## 2. Análise por Horizonte\n")
        for h in sorted(df_full['horizon'].unique()):
            f.write(f"\n### Horizonte h={h}\n")
            subset = df_full[df_full['horizon'] == h]
            if not subset.empty:
                best_model = subset.iloc[0]
                f.write(f"* **Melhor Modelo:** `{best_model['model']}`\n")
                f.write(f"* **MASE:** {best_model['mase']:.4f}\n")
                f.write(f"* **MAPE:** {best_model['mape']:.4f}%\n")
            
    logging.info(f"Relatório gerado em: {md_output}")

def generate_plots(main_config, successful_runs):
    """
    Gera gráficos PNG comparando Real vs Previsto.
    """
    logging.info("--- [PLOTS] Gerando Gráficos de Previsão ---")
    
    forecasts_path = main_config['results_paths']['forecasts']
    plots_path = main_config['results_paths']['plots']
    os.makedirs(plots_path, exist_ok=True)
    
    count = 0
    for exec_name in successful_runs:
        file_path = os.path.join(forecasts_path, f"forecast_{exec_name}.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['ds'] = pd.to_datetime(df['ds'])
                
                # Cria a figura (Agora usando o backend 'Agg' seguro)
                plt.figure(figsize=(12, 6))
                plt.plot(df['ds'], df['y'], label='Real', color='black', alpha=0.6, linewidth=1)
                plt.plot(df['ds'], df['y_hat'], label='Previsto', color='blue', alpha=0.8, linewidth=1.5)
                
                plt.title(f"Forecast: {exec_name}")
                plt.xlabel("Data")
                plt.ylabel("Valor")
                plt.legend()
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                output_plot = os.path.join(plots_path, f"plot_{exec_name}.png")
                plt.savefig(output_plot, bbox_inches='tight')
                
                # Limpa a memória explicitamente após salvar
                plt.close()
                count += 1
            except Exception as e:
                logging.error(f"Erro ao gerar plot para {exec_name}: {e}")

    logging.info(f"{count} gráficos salvos em {plots_path}")