import os
import glob
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# Importa o teste estatístico (Assumindo que você criou src/analysis/statistical_tests.py)
try:
    from src.analysis.statistical_tests import run_significance_analysis
    HAS_STATS = True
except ImportError:
    logging.warning("Módulo 'src.analysis.statistical_tests' não encontrado. Testes DM serão pulados.")
    HAS_STATS = False

# --- CONFIGURAÇÃO DE CORES (Incluindo Informer) ---
MODEL_COLORS = {
    'Real': 'black',
    'y': 'black',
    'ARIMA': '#1f77b4',         # Azul
    'ETS': '#ff7f0e',           # Laranja
    'Naive': '#2ca02c',         # Verde
    'SeasonalNaive': '#d62728', # Vermelho
    'NBEATS': '#9467bd',        # Roxo
    'NHITS': '#8c564b',         # Marrom
    'Informer': '#e377c2',      # ROSA (Novo!)
    'TFT': '#7f7f7f',           # Cinza
    'AutoARIMA': '#bcbd22'      # Oliva
}

def get_model_color(model_name):
    # Tenta encontrar a cor pelo nome ou substring
    for key, color in MODEL_COLORS.items():
        if key in model_name: 
            return color
    return '#333333' # Cinza padrão se não achar

# -----------------------------------

def generate_report(main_config, successful_runs):
    logging.info("--- [REPORT] Gerando Relatório ---")
    
    metrics_path = main_config['results_paths']['metrics']
    reports_path = main_config['results_paths']['reports']
    statistical_path = main_config['results_paths']['statistical']
    os.makedirs(reports_path, exist_ok=True)
    
    # 1. Carregar Métricas (Sem alterações aqui)
    all_files = glob.glob(os.path.join(metrics_path, "*.csv"))
    if not all_files: return
    df_list = [pd.read_csv(f) for f in all_files]
    df_full = pd.concat(df_list, ignore_index=True)
    df_full.to_csv(os.path.join(reports_path, "consolidated_metrics.csv"), index=False)
    
    # 2. Executar Testes Estatísticos (Na nova pasta)
    if HAS_STATS:
        try:
            # Isso agora gera os Heatmaps em results/statistical/
            run_significance_analysis(main_config, metrics_df=df_full)
        except Exception as e:
            logging.error(f"Erro nos testes estatísticos: {e}")

    # 3. Gerar Markdown
    md_output = os.path.join(reports_path, "relatorio_final.md")
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write("# Relatório Final\n\n")
        f.write("## Ranking de Modelos\n")
        try: f.write(df_full.sort_values('mase').to_markdown(index=False))
        except: f.write(df_full.sort_values('mase').to_string())
        
        f.write("\n\n## Testes Estatísticos\n")
        f.write("Os testes de significância (Diebold-Mariano) foram gerados.\n")
        f.write("Verifique a pasta `results/statistical` para visualizar os **Heatmaps de Significância**.\n")
        
        # Opcional: Tenta incorporar a imagem do heatmap no markdown se existir
        heatmaps = glob.glob(os.path.join(statistical_path, "*.png"))
        if heatmaps:
            f.write("\n### Visualização da Significância Estatística\n")
            for hm in heatmaps:
                rel_path = os.path.relpath(hm, start=reports_path)
                f.write(f"\n![Heatmap DM]({rel_path})\n")

    logging.info(f"Relatório Markdown salvo em: {md_output}")

def generate_plots(main_config, successful_runs):
    """
    Gera gráficos:
    1. Individuais (Real vs Previsto)
    2. Comparativos (Todos os modelos juntos por dataset) - NOVO!
    """
    logging.info("--- [PLOTS] Gerando Gráficos de Previsão ---")
    
    comparison_path = main_config['results_paths']['comparison']
    forecasts_path = main_config['results_paths']['forecasts']
    plots_path = main_config['results_paths']['plots']
    os.makedirs(plots_path, exist_ok=True)
    
    # --- Parte A: Gráficos Individuais (Mantido da sua versão) ---
    count = 0
    # Dicionário para agrupar arquivos por dataset para a Parte B
    dataset_files_map = {} 

    for exec_name in successful_runs:
        file_path = os.path.join(forecasts_path, f"forecast_{exec_name}.csv")
        
        if os.path.exists(file_path):
            try:
                # Ler CSV
                df = pd.read_csv(file_path)
                df['ds'] = pd.to_datetime(df['ds'])
                
                # Identifica Dataset (assumindo nome: forecast_DATASET_MODELO...)
                # Ex: forecast_ETTm1_ARIMA.csv
                parts = os.path.basename(file_path).split('_')
                if len(parts) >= 2:
                    ds_name = parts[1] # Pega 'ETTm1'
                    if ds_name not in dataset_files_map:
                        dataset_files_map[ds_name] = []
                    dataset_files_map[ds_name].append((exec_name, df))

                # Plot Individual
                plt.figure(figsize=(12, 6))
                plt.plot(df['ds'], df['y'], label='Real', color='black', alpha=0.6, linewidth=1)
                
                # Cor dinâmica
                color = get_model_color(exec_name)
                plt.plot(df['ds'], df['y_hat'], label=f'Previsto ({exec_name})', color=color, alpha=0.8, linewidth=1.5)
                
                plt.title(f"Forecast: {exec_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                output_plot = os.path.join(plots_path, f"plot_{exec_name}.png")
                plt.savefig(output_plot, bbox_inches='tight')
                plt.close()
                count += 1
                
            except Exception as e:
                logging.error(f"Erro ao gerar plot individual para {exec_name}: {e}")

    # --- Parte B: Gráficos Comparativos (Todos juntos) ---
    logging.info(f"Gerando gráficos comparativos para {len(dataset_files_map)} datasets...")
    
    for ds_name, runs_list in dataset_files_map.items():
        try:
            plt.figure(figsize=(15, 8))
            
            # 1. Plotar Ground Truth (apenas uma vez)
            # Pegamos o dataframe do primeiro modelo da lista
            df_ref = runs_list[0][1]
            plt.plot(df_ref['ds'], df_ref['y'], label='Real (Ground Truth)', color='black', linewidth=2.5, alpha=0.7, zorder=10)
            
            # 2. Plotar cada modelo
            for exec_name, df in runs_list:
                # Pega nome limpo do modelo (ex: tira dataset e horizonte se der)
                label_name = exec_name.replace(f"{ds_name}_", "")
                color = get_model_color(label_name)
                
                plt.plot(df['ds'], df['y_hat'], label=label_name, color=color, linewidth=1.5, alpha=0.8)

            plt.title(f"Comparativo de Modelos - Dataset: {ds_name}")
            plt.xlabel("Data")
            plt.ylabel("Valor")
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left') # Legenda fora do gráfico
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            comp_output = os.path.join(comparison_path, f"comparativo_{ds_name}.png")
            plt.savefig(comp_output)
            plt.close()
            
        except Exception as e:
            logging.error(f"Erro ao gerar comparativo para {ds_name}: {e}")

    logging.info(f"Processo de report finalizado. {count} gráficos individuais gerados.")