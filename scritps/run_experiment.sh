#!/bin/bash
# Script para limpar o terminal Git Bash e executar a pipeline de experimentos

# 1. Limpa a tela do terminal
clear

# 2. Ativa o ambiente virtual Python
# (Recomendado para garantir que está usando as dependências corretas)
# source .venv/Scripts/activate

# 3. Executa o script principal da pipeline
echo "Iniciando a execucao do experimento..."
python main.py

echo "Execucao concluida."