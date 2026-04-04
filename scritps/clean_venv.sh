#!/bin/bash

# Garante que o script pare se houver erro crítico
set -e

echo "--- Limpando Ambiente Virtual (Poetry) ---"

# Remove o ambiente gerenciado pelo Poetry
echo "🧹 Desvinculando ambiente do Poetry..."
poetry env remove --all || true

# Verifica se a pasta .venv existe e a remove
if [ -d ".venv" ]; then
    echo "🗑️  Removendo a pasta .venv..."
    rm -rf .venv
    echo "✅ Pasta .venv removida com sucesso."
else
    echo "ℹ️  Nenhuma pasta .venv encontrada para remover."
fi

# Opcional: Remove o arquivo de lock se quiser refazer o projeto do zero
if [ -f "poetry.lock" ]; then
    echo "🗑️  Removendo poetry.lock..."
    rm -f poetry.lock
fi

echo "--- Limpeza concluída ---"