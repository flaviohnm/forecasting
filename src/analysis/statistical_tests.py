# File: src/analysis/statistical_tests.py

import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss

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

    for dataset_conf in datasets_to_run:
        dataset_name = dataset_conf['name']
        print("\n" + "="*50)
        print(f"Analisando o dataset: {dataset_name}")
        print("="*50)

        # Carrega apenas os dados de treino para a análise
        train_series, _ = preprocessing.load_and_prepare_data(main_config, dataset_conf)

        # --- 1. Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin) ---
        # Hipótese Nula (H0): A série é estacionária em torno de uma tendência.
        print("\n--- Teste de Estacionariedade (KPSS) ---")
        kpss_stat, p_value, lags, crit = kpss(train_series, regression='c', nlags="auto")
        
        print(f'  Estatística do teste KPSS: {kpss_stat:.4f}')
        print(f'  P-valor: {p_value:.4f}')
        print('  Valores Críticos:')
        for key, value in crit.items():
            print(f'    {key}: {value}')

        if p_value < 0.05:
            print("  Resultado: A hipótese nula é rejeitada (p < 0.05). A série provavelmente NÃO é estacionária.")
        else:
            print("  Resultado: Falha ao rejeitar a hipótese nula (p >= 0.05). A série provavelmente é estacionária.")

        # --- 2. Gráficos ACF e PACF ---
        print("\n--- Gerando gráficos ACF e PACF ---")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Gráfico ACF
        plot_acf(train_series, ax=axes[0], lags=40)
        axes[0].set_title('Função de Autocorrelação (ACF)')

        # Gráfico PACF
        plot_pacf(train_series, ax=axes[1], lags=40)
        axes[1].set_title('Função de Autocorrelação Parcial (PACF)')
        
        plt.tight_layout()

        # Salva o gráfico
        plot_filename = f"acf_pacf_{dataset_name}.png"
        plot_filepath = os.path.join(results_path, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Gráficos salvos em: {plot_filepath}")
        plt.close()

    print("\nAnálise estatística concluída.")