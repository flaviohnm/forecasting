# Projeto de Benchmark de Modelos de Forecasting para Séries Temporais

Este projeto é parte de uma pesquisa de mestrado com o objetivo de implementar e avaliar o desempenho de modelos de previsão de séries temporais, com foco em arquiteturas híbridas e estratégias de previsão multi-step ahead, conforme as melhores práticas da literatura científica.

## O Desafio do Projeto

A previsão de múltiplos passos no futuro (*multi-step ahead forecasting*) é um desafio em aberto na área de séries temporais. Modelos simples frequentemente falham em capturar as complexidades presentes em dados do mundo real, que podem incluir múltiplas sazonalidades, tendências e estruturas não-lineares.

Inspirado por pesquisas recentes, este projeto busca enfrentar os seguintes desafios:

1.  **Implementação de Modelos Híbridos:** A literatura sugere que a combinação de modelos lineares (como o ARIMA, que captura bem tendências) e modelos não-lineares de deep learning (como o N-BEATS, que modela padrões complexos) pode gerar previsões mais acuradas. Nosso principal desafio é implementar e validar um modelo híbrido que utiliza ARIMA para a componente linear e N-BEATS para modelar os resíduos.

2.  **Comparação de Estratégias de Previsão:** A forma como a previsão multi-step é gerada (a "estratégia") impacta significativamente o resultado. Com base nos artigos de referência, este projeto irá comparar:
    * **Modelos Puros:** ARIMA (estratégia recursiva) e N-BEATS (estratégia direta).
    * **Modelos Híbridos:** Uma implementação fiel do sistema **Híbrido Recursivo-Direto** (proposto por Duarte et al., 2024) e uma adaptação para a estratégia **Híbrido MIMO**, que segundo Taieb et al. (2011), tende a ter um desempenho superior.

3.  **Benchmark e Avaliação Robusta:** A avaliação de modelos de forecasting é repleta de armadilhas. Para garantir la validade científica dos resultados, este projeto adota as melhores práticas recomendadas:
    * Uso de métricas escaladas como **MASE** (Mean Absolute Scaled Error), que permitem uma comparação justa entre modelos.
    * Uso de particionamento de dados via **origem fixa** para os experimentos iniciais, com planos de evoluir para **validação cruzada de origem rolante** (`tsCV`) para uma avaliação mais robusta.

## Estrutura do Projeto

```
/forecasting/
|
|-- data/                 # Armazena os datasets brutos e processados
|-- results/              # Contém todas as saídas: modelos, previsões, métricas e relatórios
|   |-- forecasts/
|   |-- metrics/
|   |-- models/
|   `-- reports/
|
|-- src/                  # Contém o código fonte modularizado
|   |-- __init__.py
|   |-- data_processing.py # Funções para baixar e processar dados
|   |-- models.py          # Funções de treinamento e previsão dos modelos
|   `-- reporting.py       # Funções para avaliação e geração de relatórios
|
|-- main.py               # Orquestrador principal da pipeline de experimentos
|-- requirements.txt      # Lista de dependências do Python
|-- clean_venv.sh         # Script para limpar o ambiente virtual
|-- setup_venv.sh         # Script para criar o ambiente virtual e instalar dependências
`-- run_experiment.sh     # Script para limpar o terminal e executar a pipeline
```

## Como Executar os Experimentos

Este projeto foi configurado para ser executado localmente em um ambiente Python isolado.

### 1. Pré-requisitos
* Python 3.11 instalado na sua máquina.
* Um terminal que suporte scripts shell (como Git Bash no Windows, ou o terminal padrão do macOS/Linux).

### 2. Configuração do Ambiente (Executar apenas uma vez)
Este script irá criar o ambiente virtual e instalar todas as bibliotecas necessárias listadas no `requirements.txt`.

```bash
./setup_venv.sh
```

### 3. Execução da Pipeline
Para rodar a pipeline completa (download de dados, pré-processamento, treinamento de todos os modelos e avaliação), execute o seguinte script:

```bash
./run_experiment.sh
```
O script irá limpar o terminal, ativar o ambiente virtual e iniciar a execução do `main.py`.

### 4. Configuração dos Experimentos
Todos os parâmetros dos experimentos (dataset a ser usado, horizonte de previsão, número de épocas, etc.) podem ser facilmente ajustados no "Painel de Controle" no topo do arquivo `main.py`.

## Próximos Passos
- [ ] Implementar a etapa final de `generate_final_report` para consolidar as métricas de todos os modelos.
- [ ] Adicionar testes de significância estatística (ex: Teste de Diebold-Mariano) ao relatório final.
- [ ] Expandir a matriz de experimentos para incluir mais datasets e modelos.
- [ ] (Opcional) Migrar a pipeline local para um orquestrador como o Airflow para otimizar a execução paralela em larga escala.