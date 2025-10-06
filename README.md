# Framework de Experimentação para Previsão de Séries Temporais

Este repositório contém o código-fonte de um framework de experimentação desenvolvido como parte de um projeto de mestrado. O objetivo principal é avaliar e comparar o desempenho de diferentes modelos de previsão de séries temporais, incluindo abordagens estatísticas clássicas, modelos de Deep Learning e sistemas híbridos.

A metodologia de hibridização é inspirada no trabalho de Duarte, Firmino, & de Mattos Neto (2024), que propõe um sistema híbrido combinando um modelo linear recursivo com um modelo não-linear de previsão direta para os resíduos.

**Artigo de Referência:**
> Duarte, F. C. L., Firmino, P. R. A., & de Mattos Neto, P. S. G. (2024). *A hybrid recursive direct system for multi-step mortality rate forecasting*. The Journal of Supercomputing, 80, 18430-18463.

---

## Metodologias Implementadas

O framework foi projetado para ser flexível, permitindo a fácil comparação entre diferentes estratégias de previsão multi-step:

### 1. Modelos Puros (Standalone)
Modelos estatísticos e de Deep Learning aplicados diretamente na série temporal.
- **`ARIMA`**: Modelo Auto-Regressivo Integrado de Médias Móveis, com seleção automática de ordens via `auto_arima`.
- **`ETS`**: Modelo de Suavização Exponencial (Error, Trend, Seasonality).
- **`LSTM`**: Redes de Memória de Curto e Longo Prazo, usando uma abordagem *Direta* (um modelo treinado para cada passo do horizonte).
- **`N-HiTS` / `iTransformer`**: Modelos de Deep Learning baseados em Transformers, que preveem todo o horizonte de uma vez (abordagem *MIMO*).

### 2. Híbrido ARIMA + Deep Learning (MIMO)
Uma abordagem de hibridização onde:
1.  Um modelo `ARIMA` é treinado na série original.
2.  Um único modelo de Deep Learning (`NBEATS`, `NHiTS`) é treinado para prever **todo o horizonte dos resíduos** de uma só vez (Multi-Input Multi-Output).
3.  A previsão final é a soma das previsões do ARIMA e do modelo de resíduos.

### 3. Híbrido ARIMA + Deep Learning (Direto - HyS-MF)
Implementação da metodologia do artigo de referência, **Hybrid System for Mortality Forecasting (HyS-MF)**:
1.  Um modelo `ARIMA` é treinado e faz previsões de forma **recursiva**.
2.  **H modelos** de Deep Learning (`NBEATS`, `NHiTS`) são treinados, onde cada modelo é um especialista em prever o resíduo de um passo específico do horizonte `h` (abordagem **Direta**).
3.  A previsão final para cada passo `h` é a soma da previsão recursiva do ARIMA e da previsão direta do `h`-ésimo modelo de resíduos.

---

## Estrutura do Projeto

O repositório está organizado da seguinte forma para garantir modularidade e reprodutibilidade:

```
.
├── configs/
│   ├── main_config.yaml      # Configurações principais (caminhos, datasets)
│   └── model_params.yaml     # Configurações dos modelos e estratégias
│
├── data/
│   ├── raw/                  # Datasets originais baixados
│   └── processed/            # Dados processados (se necessário)
│
├── reports/
│   └── relatorio_....md        # Relatório final em Markdown gerado pelo pipeline
│
├── results/
│   ├── metrics/              # Arquivos .csv com as métricas de cada execução
│   └── plots/                # Gráficos e dados de previsão em .csv
│
├── saved_models/               # Modelos treinados (.joblib, .pkl, .keras)
│
├── scripts/
│   ├── setup_venv.sh         # Script para configurar o ambiente do zero
│   ├── run_experiment.sh     # Script para executar os experimentos
│   └── clean_venv.sh         # Script para limpar o ambiente virtual
│
├── src/
│   ├── __main__.py           # Ponto de entrada para execução com 'python -m src'
│   ├── data_management/
│   ├── models/
│   ├── pipelines/
│   └── analysis/
│
├── requirements.txt            # Dependências do projeto
└── ...
```

---

## Configuração do Ambiente (Setup)

### Método Simplificado com Script (Recomendado)

Os scripts na pasta `scripts/` automatizam todo o processo.

**Nota para usuários Windows:** Recomenda-se executar os scripts `.sh` através de um terminal como o **Git Bash** (que já vem com a instalação do Git for Windows).

1.  **Clone o repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DA_PASTA]
    ```

2.  **Dê permissão de execução aos scripts (apenas para Linux/macOS):**
    ```bash
    chmod +x scripts/*.sh
    ```

3.  **Execute o script de configuração:**
    ```bash
    ./scripts/setup_venv.sh
    ```
    Este script irá criar o ambiente virtual, ativá-lo e instalar todas as dependências do `requirements.txt`.

<details>
<summary><strong>Método Manual (Alternativo)</strong></summary>

1.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Linux/macOS
    # ou
    .\.venv\Scripts\activate   # No Windows
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
</details>

---

## Como Utilizar o Framework

A execução dos experimentos é controlada pelos arquivos de configuração e pode ser iniciada com um único script.

### 1. Configure os Datasets e Modelos
Antes de rodar, você pode ajustar os arquivos `configs/main_config.yaml` (para datasets) e `configs/model_params.yaml` (para modelos e estratégias) conforme a necessidade do seu experimento.

### 2. Execute os Experimentos

#### Método Simplificado com Script (Recomendado)
O script `run_experiment.sh` ativa o ambiente virtual e inicia a execução do pipeline principal definido em `src/main.py`.

```bash
./scripts/run_experiment.sh
```

#### Execução Manual (para Controle Avançado)
Você também pode executar o pipeline manualmente, o que é útil para rodar configurações específicas sem alterar os arquivos. O ponto de entrada é o módulo `src.main`.

```bash
# Ative o ambiente virtual primeiro
source .venv/Scripts/activate

# Exemplo: Executa a estratégia 'full_comparison' para os datasets 'airline' e 'sunspot'
python -m src.main --datasets airline sunspot --strategy full_comparison
```

### 3. Analise os Resultados
Após a execução, os resultados estarão disponíveis nas seguintes pastas:
-   `results/metrics/`: Arquivos `.csv` com o MAPE e MASE de cada execução.
-   `results/plots/`: Gráficos de previsão e arquivos `.csv` com os dados (`real` vs. `previsao`).
-   `reports/`: Um relatório completo em formato Markdown, com tabelas e gráficos consolidados, pronto para análise.

---

### Scripts Utilitários

-   **`scripts/clean_venv.sh`**: Use este script para remover completamente a pasta `.venv` e desativar o ambiente virtual. É útil para começar uma instalação limpa do zero.