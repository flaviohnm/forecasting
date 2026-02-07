# Pipeline de Forecasting Híbrido: Estratégia HyS-MF

Este projeto implementa uma infraestrutura experimental para séries temporais de longo prazo, utilizando a metodologia **HyS-MF** (Hybrid Strategy - Model Fusion). A arquitetura foca na decomposição de tendência linear (via modelos estatísticos) combinada com a modelagem de resíduos não-lineares via modelos de Deep Learning SOTA.

## 1. Objetivo do Experimento

O objetivo central é realizar um benchmarking rigoroso entre modelos puramente estatísticos (ARIMA, ETS), modelos globais de Deep Learning (NBEATS, NHITS, iTransformer, Informer) e a abordagem híbrida proposta. O foco principal é a performance em horizontes de previsão que variam entre **24 e 720 passos** utilizando o dataset **ETTh1**.

## 2. Arquitetura do Projeto

A organização do projeto garante a rastreabilidade total dos artefatos gerados e o isolamento de responsabilidades:

```text
forecasting/
├── data/                       # Datasets crus e processados (Ex: ETTh1.csv)
├── config/                     # Definições de parâmetros
│   ├── main_config.yaml        # Caminhos globais e datasets ativos
│   └── model_params.yaml       # Hiperparâmetros e estratégias de modelos
├── src/                        # Código-fonte (Módulos)
│   ├── data_management/        # Carga e pré-processamento (preprocessing.py)
│   ├── models/                 # Definições de treinamento (arima, neuralforecast)
│   ├── pipelines/              # Orquestradores de Treino e Avaliação
│   ├── analysis/               # Testes estatísticos (Diebold-Mariano)
│   ├── reporting/              # Relatórios Markdown e Visualização
│   └── utils/                  # Auxiliares de configuração e gestão
├── results/                    # Artefatos Gerados (Saídas)
│   ├── saved_models/           # Binários de modelos (.joblib, .pkl, pastas .nf)
│   ├── metrics/                # CSVs com MAPE e MASE por execução
│   ├── forecasts/              # CSVs com séries temporais (Real vs Previsão)
│   ├── plots/                  # Gráficos de performance (PNG)
│   ├── comparison/             # Resultados do teste Diebold-Mariano (CSV)
│   └── reports/                # Relatório final consolidado (Markdown)
├── main.py                     # Ponto de entrada (Orquestrador Dinâmico)
└── README.md                   # Documentação técnica (Este arquivo)
```

## 3. Padrão de Nomenclatura e Rastreabilidade

Para evitar colisões de dados e garantir a integridade dos resultados, todos os artefatos em `results/` seguem este padrão:

* **Modelos Standalone:** `{Dataset}_standalone_{ModelName}_h{Horizon}`
    * *Exemplo:* `ETTh1_standalone_nhits_h24`

* **Modelos Híbridos:** `{Dataset}_{ResidualModel}_on_{BaseModel}_h{Horizon}`
    * *Exemplo:* `ETTh1_nhits_on_arima_h24`

## 4. Fluxo Experimental (Fases)

O experimento é processado em cinco fases sequenciais e interdependentes:

1.  **Fase 1: Modelos Base:** Treinamento dos componentes lineares/estatísticos (ARIMA, ETS) para estabelecer a base de tendência.
2.  **Fase 2: Modelos Híbridos:** Treinamento dos modelos de Deep Learning para prever exclusivamente os resíduos (erros) gerados pela Fase 1.
3.  **Fase 3: Avaliação:** Geração de previsões fora da amostra e cálculo das métricas **MASE** e **MAPE**.
4.  **Fase 4: Análise Estatística:** Execução do teste de **Diebold-Mariano** para validar a significância da melhoria do modelo híbrido.
5.  **Fase 5: Reporting:** Geração automática de gráficos de performance e consolidação do `relatorio_final.md`.

## 5. Configuração do Ambiente e Execução

* **Hardware:** Configurado para execução exclusiva em **CPU** para evitar erros de vínculo dinâmico (DLL WinError 1114) no Windows.
* **Check-point:** O orquestrador verifica automaticamente arquivos existentes em `results/metrics/` para pular modelos já treinados.

### Comandos de Execução

```bash
# Desativa otimizações oneDNN para evitar instabilidade de DLL entre TF e Torch
set TF_ENABLE_ONEDNN_OPTS=0

# Inicia o experimento orquestrado
python main.py
```

## 6. Dicionário de Métricas

Utilizamos métricas que garantem a comparabilidade entre diferentes horizontes e escalas:

* **MASE (Mean Absolute Scaled Error):** Escala o erro pelo desvio absoluto médio do benchmark Naive no treino. Valores $MASE < 1.0$ indicam performance superior ao benchmark simples.
* **MAPE (Mean Absolute Percentage Error):** Fornece a magnitude do erro em termos percentuais relativos.

## 7. Metodologia de Comparação Estatística

A validação da hipótese de pesquisa é realizada através do **Teste de Diebold-Mariano (DM)**:

* **Hipótese Nula ($H_0$):** Não há diferença de acurácia entre o modelo híbrido e o benchmark.
* **Significância:** Resultados com $p < 0.05$ são considerados estatisticamente superiores.