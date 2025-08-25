# Projeto de Benchmark de Modelos de Forecasting para Séries Temporais

Este projeto é parte de uma pesquisa de mestrado com o objetivo de implementar e avaliar o desempenho de diversos modelos de previsão de séries temporais, com foco em arquiteturas híbridas e estratégias de previsão multi-step ahead, conforme as melhores práticas da literatura científica.

## O Desafio do Projeto

A previsão de múltiplos passos no futuro (*multi-step ahead forecasting*) é um desafio em aberto na área de séries temporais. Modelos simples frequentemente falham em capturar as complexidades presentes em dados do mundo real, que podem incluir múltiplas sazonalidades, tendências e estruturas não-lineares.

Inspirado por pesquisas recentes, este projeto busca enfrentar os seguintes desafios:

1.  **Implementação de Modelos Híbridos:** A literatura sugere que a combinação de modelos lineares (como o ARIMA, que captura bem tendências) e modelos não-lineares de deep learning (como N-BEATS e N-HiTS, que modelam padrões complexos) pode gerar previsões mais acuradas. O pipeline implementa e valida essa abordagem, modelando a série original com um modelo linear e os seus resíduos com um modelo de deep learning.

2.  **Comparação de Estratégias e Modelos:** O pipeline foi projetado para comparar de forma justa o desempenho de diferentes classes de modelos e estratégias de previsão multi-step, incluindo:
    * **Modelos Estatísticos Clássicos:** ARIMA e ETS.
    * **Modelos de Deep Learning:** N-BEATS, N-HiTS, LSTM, MLP, entre outros, usando estratégias *Direct* e *MIMO*.
    * **Modelos Híbridos:** Combinações de ARIMA com os principais modelos de Deep Learning.

3.  **Benchmark e Avaliação Robusta:** Para garantir a validade científica dos resultados, este projeto adota as melhores práticas recomendadas:
    * Uso de métricas escaladas como **MASE** (Mean Absolute Scaled Error), que permitem uma comparação justa entre modelos em diferentes séries temporais.
    * Uso de particionamento de dados via **origem fixa** para os experimentos, com um framework que pode ser estendido para validação cruzada no futuro.

## Estrutura do Projeto

/forecasting/
|
|-- config/               # Contém os arquivos de configuração dos experimentos
|   |-- datasets_config.json
|   -- models_config.json | |-- data/                 # Armazena os datasets brutos e processados |-- results/              # Contém todas as saídas: previsões, métricas e relatórios |   |-- forecasts/ |   |-- metrics/ |   -- reports/
|-- scripts
    | -- setup_venv.sh    # Script para criar o ambiente virtual -- 
|-- src/                  # Contém o código fonte modularizado
|   |-- init.py
|   |-- data_processing.py
|   |-- models.py
|   |-- reporting.py |
|-- run_experiments.py    # Orquestrador principal da pipeline de experimentos
|-- generate_report.py    # Script para gerar o relatório final a partir de resultados existentes
|-- requirements.txt      # Lista de dependências do Python

## Como Executar os Experimentos

Este projeto foi configurado para ser executado localmente em um ambiente Python isolado, com controle total sobre os experimentos através de arquivos de configuração.

### 1. Configuração dos Experimentos (O Principal Painel de Controle)
Toda a execução é controlada por dois arquivos na pasta `config/`:

* **`config/datasets_config.json`**: Use este arquivo para habilitar (`"enabled": true`) ou desabilitar (`"enabled": false`) os datasets que você deseja incluir na rodada de testes.
* **`config/models_config.json`**: Use este arquivo para habilitar ou desabilitar modelos específicos e para ajustar seus hiperparâmetros (`max_steps`, `learning_rate`, etc.).

### 2. Pré-requisitos
* Python 3.11 instalado na sua máquina.
* Um terminal que suporte scripts shell (como Git Bash no Windows, ou o terminal padrão do macOS/Linux).

### 3. Configuração do Ambiente (Executar apenas uma vez)
Este script irá criar o ambiente virtual e instalar todas as bibliotecas necessárias listadas no `requirements.txt`.

```bash
./setup_venv.sh
```
