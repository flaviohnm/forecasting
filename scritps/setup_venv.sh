#!/bin/bash
echo "--- Configurando Novo Ambiente Virtual ---"

# 1. Cria o ambiente virtual usando python3.11
echo "Criando nova venv com Python 3.11..."
python3.11 -m venv .venv

# 2. Ativa o novo ambiente
echo "Ativando a venv..."
source .venv/Scripts/activate

# 3. Garante que o pip está atualizado
echo "Atualizando o pip..."
python3.11.exe -m pip install --upgrade pip

# 4. Instala as dependências do projeto
echo "Instalando as dependências do requirements.txt..."
pip install -r requirements.txt

echo ""
echo "--- Ambiente Virtual configurado com sucesso! ---"
echo "Para ativá-lo no futuro, use o comando: source .venv/Scripts/activate"