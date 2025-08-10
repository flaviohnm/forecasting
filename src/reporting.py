# /forecasting/src/reporting.py (VERSÃO COM CORREÇÃO DE NOMES DE ARQUIVO)
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mase(train_series, test_series, forecast_series):
    # (Esta função não muda)
    train_series = np.array(train_series); test_series = np.array(test_series); forecast_series = np.array(forecast_series)
    n = len(train_series)
    d = np.abs(np.diff(train_series)).sum() / (n - 1)
    if d == 0: return np.inf
    errors = np.abs(test_series - forecast_series)
    return errors.mean() / d

# --- MUDANÇA AQUI: Adicionando o parâmetro 'model_name' ---
def evaluate_forecasts(forecast_path: str, train_path: str, results_dir: str, target_column: str, model_name: str):
    """Calcula e salva um conjunto de métricas de erro."""
    print(f"Avaliando previsões de: {forecast_path}")
    forecast_df, train_df = pd.read_csv(forecast_path), pd.read_csv(train_path)
    
    clean_df = forecast_df.dropna(subset=['actual', 'forecast'])
    if len(clean_df) == 0: return None

    actuals, forecasts, train_series = clean_df['actual'], clean_df['forecast'], train_df[target_column]
    metrics = {
        "MAE": mean_absolute_error(actuals, forecasts),
        "RMSE": np.sqrt(mean_squared_error(actuals, forecasts)),
        "MASE": calculate_mase(train_series.values, actuals.values, forecasts.values)
    }
    print(f"Métricas para o modelo {model_name}: {metrics}")

    # --- MUDANÇA AQUI: Usando o 'model_name' para salvar o arquivo ---
    results_path = Path(results_dir) / f"{model_name}_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Métricas salvas em: {results_path}")
    return str(results_path)

def generate_final_report(results_dir: str, dataset_name: str, forecast_horizon: int, model_names: list):
    # (O restante do arquivo não muda)
    print("\n[ETAPA FINAL] Gerando relatório consolidado...")
    metrics_dir = Path(results_dir) / "metrics" / dataset_name
    all_metrics = []
    for model_name in model_names:
        metric_file = metrics_dir / f"{model_name}_metrics.json"
        if metric_file.exists():
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
                metrics['model'] = model_name
                all_metrics.append(metrics)
        else:
            print(f"AVISO: Arquivo de métrica não encontrado para o modelo '{model_name}' em {metric_file}")
    if not all_metrics:
        print("Nenhum arquivo de métrica encontrado para gerar o relatório.")
        return
    metrics_df = pd.DataFrame(all_metrics).set_index('model')
    metrics_table_md = metrics_df.round(4).to_markdown()
    forecast_dir = Path(results_dir) / "forecasts" / dataset_name / "csv"
    plt.figure(figsize=(12, 8))
    first_forecast_file = next(forecast_dir.glob('*.csv'), None)
    if first_forecast_file:
        df_actual = pd.read_csv(first_forecast_file)
        plt.plot(pd.to_datetime(df_actual['ds']), df_actual['actual'], label='Valores Reais', color='black', linewidth=2.5)
    for model_name in model_names:
        forecast_file = forecast_dir / f"{dataset_name}_{model_name}_forecasts.csv"
        if forecast_file.exists():
            df_forecast = pd.read_csv(forecast_file)
            plt.plot(pd.to_datetime(df_forecast['ds']), df_forecast['forecast'], label=model_name, linestyle='--')
    plt.title(f'Comparação de Previsões - Dataset {dataset_name.capitalize()}', fontsize=16)
    plt.xlabel('Data'); plt.ylabel('Valor'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    reports_dir = Path(results_dir) / "reports"; images_dir = reports_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_path = images_dir / f"forecast_comparison_{dataset_name}.png"
    plt.savefig(plot_path)
    plt.close()
    report_content = f"""
# Relatório de Benchmark de Modelos de Forecasting
**Dataset:** `{dataset_name}`
**Horizonte de Previsão:** `{forecast_horizon}` passos
## Resumo das Métricas de Performance
A tabela abaixo compara o desempenho de todos os modelos executados.
{metrics_table_md}
## Gráfico Comparativo das Previsões
O gráfico abaixo mostra as previsões de cada modelo em comparação com os valores reais.
![Comparativo de Previsões](images/forecast_comparison_{dataset_name}.png)
"""
    report_path = reports_dir / "final_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"--- Relatório Final gerado com sucesso em: {report_path} ---")