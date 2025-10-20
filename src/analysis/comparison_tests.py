# File: src/analysis/comparison_tests.py

import os
import pandas as pd
import numpy as np
import pingouin as pg
import scipy.stats as ss # Para Friedman
import scikit_posthocs as sp # Para Nemenyi post-hoc
from itertools import combinations
import logging

# Configura o logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger('pingouin').setLevel(logging.ERROR)
logging.getLogger('numexpr').setLevel(logging.WARNING) # Suprime aviso do numexpr no pandas

# --- LÓGICA DO DIEBOLD-MARIANO ---
def run_diebold_mariano_tests(main_config: dict, successful_executions: list, results_path: str):
    """Realiza o teste Diebold-Mariano (aproximado por t-test pareado nos erros)
       comparando o melhor modelo com os outros, por dataset."""
    print("\n--- Iniciando Testes de Comparação (Diebold-Mariano) ---")
    metrics_path = main_config['results_paths']['metrics']
    forecasts_path = main_config['results_paths']['plots']

    datasets = {}
    for ds_conf, md_conf in successful_executions:
        ds_name = ds_conf['name']
        if ds_name not in datasets: datasets[ds_name] = []
        datasets[ds_name].append((ds_conf, md_conf))

    dm_results_all = []
    for dataset_name, executions_list in datasets.items():
        print(f"  Analise DM para Dataset: {dataset_name}")
        metrics, forecast_data, valid_executions = [], {}, []
        for ds_conf, md_conf in executions_list:
            execution_name = f"{dataset_name}_{md_conf['model_name']}"
            metric_file = os.path.join(metrics_path, f"metrics_{execution_name}.csv")
            forecast_file = os.path.join(forecasts_path, f"forecasts_{execution_name}.csv")
            if os.path.exists(metric_file) and os.path.exists(forecast_file):
                try:
                    metric_df = pd.read_csv(metric_file)
                    forecast_df = pd.read_csv(forecast_file)
                    if 'MAPE' in metric_df.columns and 'real' in forecast_df.columns and 'previsao' in forecast_df.columns:
                        if not forecast_df[['real', 'previsao']].isnull().any().any():
                            metrics.append(metric_df.iloc[0])
                            forecast_data[execution_name] = forecast_df
                            valid_executions.append((ds_conf, md_conf, execution_name))
                        else: logging.warning(f"DM [{dataset_name}]: Previsões NaN em '{execution_name}'.")
                    else: logging.warning(f"DM [{dataset_name}]: Colunas ausentes em '{execution_name}'.")
                except Exception as e: logging.warning(f"DM [{dataset_name}]: Erro lendo '{execution_name}': {e}.")
            else: logging.warning(f"DM [{dataset_name}]: Arquivos não encontrados para '{execution_name}'.")

        if not metrics or len(valid_executions) < 2:
            print("    Modelos insuficientes para teste DM.")
            continue

        metrics_df = pd.DataFrame(metrics).sort_values(by='MAPE')
        best_model_exec_name = metrics_df.iloc[0]['execution_name']
        print(f"    Melhor modelo (MAPE): {best_model_exec_name} ({metrics_df.iloc[0]['MAPE']:.4f})")
        y_true = forecast_data[best_model_exec_name]['real'].values
        y_pred_best = forecast_data[best_model_exec_name]['previsao'].values

        for ds_conf, md_conf, exec_name_other in valid_executions:
            if exec_name_other == best_model_exec_name: continue
            y_pred_other = forecast_data[exec_name_other]['previsao'].values
            min_len = min(len(y_true), len(y_pred_best), len(y_pred_other))
            if len(y_true) > min_len or len(y_pred_best) > min_len or len(y_pred_other) > min_len:
                logging.warning(f"DM [{dataset_name}]: Ajustando tamanho das séries entre {best_model_exec_name} e {exec_name_other}.")

            error_best = np.abs(y_true[:min_len] - y_pred_best[:min_len])
            error_other = np.abs(y_true[:min_len] - y_pred_other[:min_len])

            # Verifica se os erros têm variância (necessário para t-test)
            if np.var(error_best) < 1e-10 or np.var(error_other) < 1e-10:
                 logging.warning(f"DM [{dataset_name}]: Erros constantes detectados entre {best_model_exec_name} e {exec_name_other}. Pulando teste.")
                 continue # Pula este par se os erros forem constantes

            try:
                # --- CORREÇÃO APLICADA AQUI ---
                # Usar pingouin.ttest para comparação pareada direta
                dm_result = pg.ttest(error_best, error_other, paired=True, alternative='two-sided')

                dm_stat = dm_result['T'].iloc[0]
                p_value = dm_result['p-val'].iloc[0]
                significance = "Sim" if p_value < 0.05 else "Não"
                print(f"    vs {exec_name_other}: DM Stat={dm_stat:.3f}, p={p_value:.4f} (Sig? {significance})")
                dm_results_all.append({'dataset': dataset_name, 'modelo_base': best_model_exec_name,
                                       'modelo_comparado': exec_name_other, 'dm_statistic': dm_stat,
                                       'p_value': p_value, 'diferenca_significativa (p<0.05)': significance})
            except Exception as e: logging.error(f"DM [{dataset_name}]: Falha no teste ttest entre {best_model_exec_name} e {exec_name_other}: {e}")

    if dm_results_all:
        dm_df = pd.DataFrame(dm_results_all)
        output_file = os.path.join(results_path, "diebold_mariano_summary.csv")
        dm_df.to_csv(output_file, index=False)
        print(f"  Resultados DM salvos em: {output_file}")
    else: print("  Nenhum teste DM pôde ser realizado.")


# --- LÓGICA DO FRIEDMAN E NEMENYI (sem alterações) ---
def run_friedman_nemenyi_test(main_config: dict, successful_executions: list, results_path: str):
    """Realiza o teste de Friedman e Nemenyi post-hoc."""
    print("\n--- Iniciando Testes de Comparação Global (Friedman + Nemenyi) ---")
    metrics_path = main_config['results_paths']['metrics']

    all_metrics = []
    for ds_conf, md_conf in successful_executions:
        execution_name = f"{ds_conf['name']}_{md_conf['model_name']}"
        metric_file = os.path.join(metrics_path, f"metrics_{execution_name}.csv")
        if os.path.exists(metric_file):
            try:
                df = pd.read_csv(metric_file)
                if 'MAPE' in df.columns and not df.empty and not df['MAPE'].isnull().any():
                    df['model_name'] = md_conf['model_name']
                    all_metrics.append(df[['dataset', 'model_name', 'MAPE']].iloc[0])
                else: logging.warning(f"Friedman: Coluna MAPE ausente ou NaN em '{execution_name}'.")
            except Exception as e: logging.warning(f"Friedman: Erro lendo métrica '{execution_name}': {e}.")
        else: logging.warning(f"Friedman: Arquivo de métrica não encontrado para '{execution_name}'.")

    if not all_metrics:
        print("  Nenhuma métrica válida encontrada para o teste de Friedman.")
        return

    metrics_df = pd.DataFrame(all_metrics)

    try:
        pivot_df = metrics_df.pivot(index='dataset', columns='model_name', values='MAPE')
    except ValueError as e:
         logging.error(f"Friedman: Erro ao pivotar: {e}. Tentando agregar duplicatas.")
         metrics_df_agg = metrics_df.groupby(['dataset', 'model_name'])['MAPE'].mean().reset_index()
         try:
             pivot_df = metrics_df_agg.pivot(index='dataset', columns='model_name', values='MAPE')
             logging.warning("Friedman: Médias de MAPE usadas devido a execuções duplicadas.")
         except ValueError as e2:
              logging.error(f"Friedman: Falha ao pivotar mesmo após agregação: {e2}. Abortando Friedman.")
              return

    pivot_df.dropna(axis=0, how='any', inplace=True)
    pivot_df.dropna(axis=1, how='any', inplace=True)

    if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
        print("  Número insuficiente de datasets ou modelos completos para o teste de Friedman.")
        print(f"  Shape final: {pivot_df.shape}")
        return

    print(f"  Datasets completos usados: {pivot_df.index.tolist()}")
    print(f"  Modelos completos usados: {pivot_df.columns.tolist()}")

    ranks_df = pivot_df.apply(ss.rankdata, axis=1, result_type='broadcast')

    try:
        stat, p_friedman = ss.friedmanchisquare(*[ranks_df[col] for col in ranks_df.columns])
        print(f"  Resultado Friedman: Estatística={stat:.3f}, p-valor={p_friedman:.4g}")
        friedman_result_df = pd.DataFrame({'statistic': [stat], 'p_value': [p_friedman], 'significativo (p<0.05)': [p_friedman < 0.05]})
        friedman_output_file = os.path.join(results_path, "friedman_test_summary.csv")
        friedman_result_df.to_csv(friedman_output_file, index=False)
        print(f"  Resultado Friedman salvo em: {friedman_output_file}")

        if p_friedman < 0.05:
            print("  Teste de Friedman significativo. Executando post-hoc de Nemenyi...")
            try:
                nemenyi_result = sp.posthoc_nemenyi_friedman(ranks_df)
                print("  Matriz de p-valores (Teste de Nemenyi):")
                print(nemenyi_result.round(4))
                nemenyi_output_file = os.path.join(results_path, "nemenyi_posthoc_pvalues.csv")
                nemenyi_result.to_csv(nemenyi_output_file)
                print(f"  Resultados Nemenyi salvos em: {nemenyi_output_file}")
            except Exception as e_nemenyi: logging.error(f"Friedman: Erro no teste Nemenyi: {e_nemenyi}")
        else: print("  Teste de Friedman não significativo. Teste post-hoc não necessário.")
    except Exception as e_friedman: logging.error(f"Friedman: Erro no teste de Friedman: {e_friedman}")


# --- FUNÇÃO PRINCIPAL DO MÓDULO ---
def run_tests(main_config: dict, successful_executions: list):
    """Executa todos os testes de comparação definidos."""
    results_path = os.path.join("results", "comparison_tests")
    os.makedirs(results_path, exist_ok=True)

    if not successful_executions:
        print("Nenhuma execução bem-sucedida encontrada. Pulando testes de comparação.")
        return

    run_diebold_mariano_tests(main_config, successful_executions, results_path)
    run_friedman_nemenyi_test(main_config, successful_executions, results_path)

    print("\nExecução de todos os testes de comparação concluída.")