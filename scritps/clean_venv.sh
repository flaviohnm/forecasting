#!/bin/bash
echo "--- Limpando Ambiente Virtual Antigo ---"

# Verifica se o ambiente virtual está ativo e tenta desativá-lo
if command -v deactivate &> /dev/null
then
    echo "Desativando ambiente virtual..."
    deactivate
fi

# Verifica se a pasta .venv existe e a remove
if [ -d ".venv" ]; then
    echo "Removendo a pasta .venv antiga..."
    rm -rf .venv
    echo "Pasta .venv removida com sucesso."
else
    echo "Nenhuma pasta .venv encontrada para remover."
fi

echo "--- Limpeza concluída ---"