#!/bin/bash

# Garante que o script pare se houver erro crítico
set -e

echo "--- Limpando Ambiente Virtual Antigo ---"

# Tenta desativar apenas se a função existir no shell atual
if type deactivate >/dev/null 2>&1; then
    echo "Desativando ambiente virtual..."
    deactivate || true
fi

# Verifica se a pasta .venv existe e a remove
if [ -d ".venv" ]; then
    echo "🗑️  Removendo a pasta .venv antiga..."
    rm -rf .venv
    echo "✅ Pasta .venv removida com sucesso."
else
    echo "ℹ️  Nenhuma pasta .venv encontrada para remover."
fi

echo "--- Limpeza concluída ---"