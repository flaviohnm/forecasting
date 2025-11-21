# File: src/analysis/statistical_tests.py

import os
import matplotlib
# Garante backend não interativo para plotagem
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss
import logging
import numpy as np  # Importado para o cálculo de lags

from src.data_management import preprocessing


def run_tests(main_config: dict, datasets_to_run: list):
    """
    Executa os testes estatísticos (KPSS, ACF, PACF) para os datasets especificados.
    """
    print("Iniciando a execução dos testes estatísticos...")

    # Cria a pasta para salvar os resultados dos testes
    results_path = os.path.join("results", "statistical_tests")
    os.makedirs(results_path, exist_ok=True)

    if not datasets_to_run:
        print("Nenhum dataset válido para processar.")
        return

    # Garante que datasets_to_run seja uma lista de dicionários
    if not isinstance(datasets_to_run, list) or (
            datasets_to_run and not isinstance(datasets_to_run[0], dict)):
        logging.error(
            "Formato inesperado para 'datasets_to_run'. Esperava uma lista de dicionários."
        )
        # Tenta extrair os valores do dicionário se foi passado errado
        if isinstance(datasets_to_run, dict):
            datasets_to_run = list(datasets_to_run.values())
        else:
            return  # Não pode prosseguir

    for dataset_conf in datasets_to_run:
        dataset_name = dataset_conf['name']
        print("\n" + "=" * 50)
        print(f"Analisando o dataset: {dataset_name}")
        print("=" * 50)

        try:
            # Carrega apenas os dados de treino para a análise
            train_series, _ = preprocessing.load_and_prepare_data(
                main_config, dataset_conf)

            if train_series.empty or len(
                    train_series) < 10:  # Pula se a série for muito curta
                logging.warning(
                    f"Série de treino para '{dataset_name}' está vazia ou é muito curta (<10 pontos). Pulando testes."
                )
                continue

            # --- 1. Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin) ---
            print("\n--- Teste de Estacionariedade (KPSS) ---")
            # Adiciona try-except para o KPSS que pode falhar com 'auto' lags em séries curtas
            try:
                kpss_stat, p_value, lags, crit = kpss(train_series,
                                                      regression='c',
                                                      nlags="auto")

                print(f'  Estatística do teste KPSS: {kpss_stat:.4f}')
                print(f'  P-valor: {p_value:.4f}')
                print('  Valores Críticos:')
                for key, value in crit.items():
                    print(f'    {key}: {value}')

                if p_value < 0.05:
                    print(
                        "  Resultado: A hipótese nula é rejeitada (p < 0.05). A série provavelmente NÃO é estacionária."
                    )
                else:
                    print(
                        "  Resultado: Falha ao rejeitar a hipótese nula (p >= 0.05). A série provavelmente é estacionária."
                    )
            except Exception as e_kpss:
                logging.warning(
                    f"Falha ao executar o teste KPSS para '{dataset_name}': {e}"
                )

            # --- 2. Gráficos ACF e PACF ---
            print("\n--- Gerando gráficos ACF e PACF ---")

            # --- CORREÇÃO APLICADA AQUI ---
            n_obs = len(train_series)
            # Calcula o máximo de lags permitido (n/2 - 1)
            # O -2 é para segurança, garantindo que seja < n/2
            max_dynamic_lags = (n_obs // 2) - 2

            # Usa 40 lags OU o máximo permitido, o que for MENOR
            # Garante que lags seja pelo menos 1, caso n_obs seja muito pequeno
            lags_to_use = max(1, min(40, max_dynamic_lags))

            print(
                f"  (Info: Usando {lags_to_use} lags para ACF/PACF baseado no tamanho da amostra de {n_obs})"
            )

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Gráfico ACF (usando lags dinâmico)
            plot_acf(train_series, ax=axes[0], lags=lags_to_use)
            axes[0].set_title(
                f'Função de Autocorrelação (ACF) - {dataset_name}')

            # Gráfico PACF (usando lags dinâmico)
            plot_pacf(train_series, ax=axes[1], lags=lags_to_use)
            axes[1].set_title(
                f'Função de Autocorrelação Parcial (PACF) - {dataset_name}')

            plt.tight_layout()

            # Salva o gráfico
            plot_filename = f"acf_pacf_{dataset_name}.png"
            plot_filepath = os.path.join(results_path, plot_filename)
            plt.savefig(plot_filepath)
            print(f"Gráficos salvos em: {plot_filepath}")
            plt.close(fig)  # Fecha a figura específica

        except FileNotFoundError:
            logging.error(
                f"Arquivo de dados não encontrado para '{dataset_name}'. Pulando testes."
            )
        except Exception as e:
            import traceback
            logging.error(
                f"Erro ao processar testes para '{dataset_name}': {e}\n{traceback.format_exc()}"
            )
            if 'fig' in locals():
                plt.close(fig)  # Garante fechar a figura em caso de erro

    print("\nAnálise estatística concluída.")
