#!/bin/bash

set -e

echo "======================================================="
echo "   🛠️  Setup Inteligente de Ambiente (Poetry)"
echo "======================================================="

# Verifica se o Poetry está instalado
if ! command -v poetry &> /dev/null; then
    echo "❌ Erro: Poetry não encontrado no sistema."
    echo "Instale executando: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# --- 1. Configurar o Poetry ---
echo "⚙️  Configurando o Poetry para criar a .venv localmente..."
poetry config virtualenvs.in-project true

echo "🐍 Apontando o Poetry para o Python 3.11..."
poetry env use 3.11

# --- 2. Instalar Dependências Base ---
echo "📦 Instalando dependências do pyproject.toml..."
poetry install

# --- 3. Instalação Dinâmica do PyTorch (Auto-GPU) ---
echo "🔍 Verificando hardware para PyTorch..."

# Tenta detectar nvidia-smi para instalar o CUDA correto
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU NVIDIA detectada! Instalando versão com CUDA dentro do Poetry..."
    # Usa o poetry run pip para garantir que vai pro lugar certo
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "🐢 Nenhuma GPU detectada. Instalando versão CPU dentro do Poetry..."
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "======================================================="
echo "✅ Setup Finalizado com Sucesso!"