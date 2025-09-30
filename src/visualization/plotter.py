import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_plots(main_config: dict, executions_to_run: list):
    """
    Gera e salva os gráficos de previsão para as execuções especificadas.

    - main_config: Configuração geral do projeto.
    - executions_to_run: Lista de tuplas (dataset_conf, model_conf) para plotar.
    """
    print("Iniciando a geração de gráficos...")

    # Path onde as previsões foram salvas pela etapa de 'evaluate'
    forecasts_path = main_config['results_paths']['plots']

    if not executions_to_run:
        print("Nenhuma execução válida para plotar.")
        return

    for dataset_conf, model_conf in executions_to_run:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        csv_filename = f"forecasts_{execution_name}.csv"
        csv_filepath = os.path.join(forecasts_path, csv_filename)

        if not os.path.exists(csv_filepath):
            print(
                f"AVISO: Arquivo de previsão '{csv_filepath}' não encontrado. Pulando gráfico para '{execution_name}'."
            )
            continue

        print(f"Gerando gráfico para '{execution_name}'...")

        # Carrega os dados da previsão
        df = pd.read_csv(csv_filepath, index_col=0, parse_dates=True)

        # Cria o gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(df.index,
                 df['real'],
                 label='Valores Reais',
                 color='black',
                 marker='.')
        plt.plot(df.index,
                 df['previsao'],
                 label='Previsão do Modelo',
                 color='red',
                 linestyle='--')

        # Se houver outras colunas de previsão (como no modelo híbrido), plota também
        if 'previsao_arima' in df.columns:
            plt.plot(df.index,
                     df['previsao_arima'],
                     label='Previsão ARIMA (Linear)',
                     color='blue',
                     linestyle=':')

        # Configurações do gráfico
        plt.title(f"Previsão vs. Real para '{execution_name}'")
        plt.xlabel("Data")
        plt.ylabel("Valor da Série")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Salva o gráfico
        plot_filename = f"plot_{execution_name}.png"
        plot_filepath = os.path.join(forecasts_path, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Gráfico salvo em: {plot_filepath}")
        plt.close()  # Fecha a figura para economizar memória

    print("Geração de gráficos concluída.")
