# File: src/visualization/plotter.py

import matplotlib
# --- CORRECTION ADDED HERE ---
# Force matplotlib to use a non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd


def generate_plots(main_config: dict, successful_executions: list):
    """
    Gera gráficos de previsão para as execuções bem-sucedidas.
    """
    print("Iniciando a geração de gráficos...")
    plots_path = main_config['results_paths']['plots']
    os.makedirs(plots_path, exist_ok=True)

    for dataset_conf, model_conf in successful_executions:
        execution_name = f"{dataset_conf['name']}_{model_conf['model_name']}"
        forecast_file = os.path.join(plots_path,
                                     f"forecasts_{execution_name}.csv")
        plot_file = os.path.join(plots_path, f"plot_{execution_name}.png")

        if os.path.exists(forecast_file):
            try:
                print(f"Gerando gráfico para '{execution_name}'...")
                forecast_df = pd.read_csv(forecast_file)

                # Tenta converter a coluna de índice para datetime se existir
                index_col_name = forecast_df.columns[
                    0]  # Assume a primeira coluna como índice
                if pd.api.types.is_string_dtype(forecast_df[index_col_name]):
                    try:
                        forecast_df[index_col_name] = pd.to_datetime(
                            forecast_df[index_col_name])
                        forecast_df = forecast_df.set_index(index_col_name)
                    except Exception:
                        print(
                            f"  Aviso: Não foi possível converter a coluna '{index_col_name}' para data/hora. Usando índice numérico."
                        )
                        forecast_df = forecast_df.set_index(
                            pd.RangeIndex(len(forecast_df))
                        )  # Usa índice numérico como fallback

                plt.figure(figsize=(12, 6))

                # Plota os dados reais
                if 'real' in forecast_df.columns:
                    plt.plot(forecast_df.index,
                             forecast_df['real'],
                             label='Real',
                             color='blue',
                             marker='o',
                             linestyle='-')

                # Plota a previsão principal
                if 'previsao' in forecast_df.columns:
                    plt.plot(forecast_df.index,
                             forecast_df['previsao'],
                             label='Previsão',
                             color='red',
                             marker='x',
                             linestyle='--')

                # Opcional: Plota componentes do híbrido se existirem
                if 'previsao_arima' in forecast_df.columns:
                    plt.plot(forecast_df.index,
                             forecast_df['previsao_arima'],
                             label='Previsão ARIMA (Híbrido)',
                             color='green',
                             linestyle=':',
                             alpha=0.7)
                if 'previsao_residuos' in forecast_df.columns:
                    # Pode ser útil plotar os resíduos em um eixo secundário ou em outro gráfico
                    # Por simplicidade, vamos plotá-los no mesmo eixo por enquanto
                    plt.plot(forecast_df.index,
                             forecast_df['previsao_residuos'],
                             label='Previsão Resíduos (Híbrido)',
                             color='orange',
                             linestyle='-.',
                             alpha=0.7)

                plt.title(f"Previsão vs Real - {execution_name}")
                plt.xlabel("Tempo")
                plt.ylabel("Valor")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close()  # Fecha a figura para liberar memória
                print(f"Gráfico salvo em: {plot_file}")

            except Exception as e:
                import traceback
                print(
                    f"ERRO ao gerar gráfico para '{execution_name}': {e}\n{traceback.format_exc()}"
                )
        else:
            print(
                f"AVISO: Arquivo de previsão '{forecast_file}' não encontrado. Pulando gráfico para '{execution_name}'."
            )

    print("Geração de gráficos concluída.")
