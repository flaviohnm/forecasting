#!/bin/bash
# Script corrigido para executar a pipeline

# 1. Limpa a tela do terminal
clear

# 2. Ativa o ambiente virtual Python (IMPORTANTE)
echo "Ativando o ambiente virtual..."
source .venv/Scripts/activate

# 3. Executa o script principal da pipeline com o nome do MÓDULO
echo "Iniciando a execucao dos experimentos..."
python -m src.main

echo "Execucao dos experimentos concluida."