# /forecasting/src/reporting.py (VERSÃO COM RELATÓRIO CONSOLIDADO)
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mase(train_series, test_series, forecast_series):
    """Calcula o Mean Absolute Scaled Error (MASE)."""
    train_series = np.array(train_series)
    test_series = np.array(test_series)
    forecast_series = np.array(forecast_series)
    n = len(train_series)
    if n < 2: return np.inf # Não é possível calcular a diferença se houver menos de 2 pontos
    d = np.abs(np.diff(train_series)).sum() / (n - 1)
    if d == 0: return np.inf # Evita divisão por zero se a série de treino for constante
    errors = np.abs(test_series - forecast_series)
    return errors.mean() / d

def create_and_save_forecast_df(test_df: pd.DataFrame, forecast_values: np.ndarray, forecast_dir: str, model_name: str, dataset_name: str, target_column: str, ds_column='ds'):
    """Cria um DataFrame padronizado com as previsões e salva em CSV."""
    df_out = pd.DataFrame({
        'unique_id': dataset_name,
        'ds': test_df[ds_column].values,
        'actual': test_df[target_column].values,
        'forecast': forecast_values
    })
    
    output_path = Path(forecast_dir)
    csv_path_dir = output_path / "csv"
    csv_path_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{dataset_name}_{model_name}_forecasts"
    csv_path = csv_path_dir / f"{base_filename}.csv"
    df_out.to_csv(csv_path, index=False)
    
    pkl_path_dir = output_path / "pkl"
    pkl_path_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_path_dir / f"{base_filename}.pkl"
    df_out.to_pickle(pkl_path)
    
    print(f"Previsões do modelo '{model_name}' salvas em: {csv_path}")
    return str(csv_path)

def evaluate_forecasts(forecast_path: str, train_df: pd.DataFrame, results_dir: str, target_column: str, model_name: str):
    """Calcula e salva um conjunto de métricas de erro. Recebe train_df diretamente."""
    forecast_df = pd.read_csv(forecast_path)
    
    clean_df = forecast_df.dropna(subset=['actual', 'forecast'])
    if len(clean_df) == 0: return None

    actuals, forecasts, train_series = clean_df['actual'], clean_df['forecast'], train_df[target_column]
    
    metrics = {
        "MAE": mean_absolute_error(actuals, forecasts),
        "RMSE": np.sqrt(mean_squared_error(actuals, forecasts)),
        "MASE": calculate_mase(train_series.values, actuals.values, forecasts.values)
    }
    print(f"Métricas para o modelo {model_name}: {metrics}")

    results_path = Path(results_dir) / f"{model_name}_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Métricas salvas em: {results_path}")
    return str(results_path)

def generate_final_report(results_dir: str, all_dataset_results: list):
    """Agrega todas as métricas e previsões para gerar um relatório final consolidado em Markdown."""
    print("\n[ETAPA FINAL] Gerando relatório consolidado para todos os datasets...")
    
    final_report_content = "# Relatório de Benchmark de Modelos de Forecasting\n\n"
    
    reports_dir = Path(results_dir) / "reports"
    images_dir = reports_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for dataset_result in all_dataset_results:
        dataset_name = dataset_result["dataset_name"]
        forecast_horizon = dataset_result["forecast_horizon"]
        model_names = dataset_result["model_names"]
        
        print(f"\n--- Processando relatório para o dataset: {dataset_name} ---")

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
            print(f"Nenhum arquivo de métrica encontrado para o dataset '{dataset_name}'. Pulando.")
            continue

        metrics_df = pd.DataFrame(all_metrics).set_index('model')
        
        styled_df = metrics_df.round(3).astype(str)
        for col in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[col]):
                min_idx = metrics_df[col].idxmin()
                original_value = metrics_df.loc[min_idx, col]
                bold_value = f"**{original_value:.3f}**"
                styled_df.loc[min_idx, col] = bold_value

        metrics_table_md = styled_df.to_markdown()

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
        
        plot_path = images_dir / f"forecast_comparison_{dataset_name}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico comparativo para '{dataset_name}' salvo em: {plot_path}")

        report_content_for_dataset = f"""
## Dataset: `{dataset_name}`

**Horizonte de Previsão:** `{forecast_horizon}` passos

### Resumo das Métricas de Performance

A tabela abaixo compara o desempenho de todos os modelos executados. A métrica principal para comparação, conforme as boas práticas, é o **MASE (Mean Absolute Scaled Error)**. Um valor de MASE < 1 indica que o modelo é melhor que uma previsão ingênua (naive) no conjunto de treino.

{metrics_table_md}

### Gráfico Comparativo das Previsões

O gráfico abaixo mostra as previsões de cada modelo em comparação com os valores reais do conjunto de teste.

![Comparativo de Previsões - {dataset_name}](images/forecast_comparison_{dataset_name}.png)
<br>

---
"""
        final_report_content += report_content_for_dataset

    report_path = reports_dir / "final_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report_content)
        
    print(f"\n--- Relatório Final consolidado gerado com sucesso em: {report_path} ---")