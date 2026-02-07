#!/bin/bash

# Encerra se der erro
set -e

echo "=== Configurando Ambiente Virtual (Universal) ==="

# 1. Definir qual Python usar
# Tenta achar o 3.11, se não achar, usa o python3 padrão do sistema
if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
    echo "✅ Python 3.11 encontrado."
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
    echo "⚠️ Python 3.11 não encontrado. Usando python3 padrão."
else
    echo "❌ Erro: Nenhum Python 3 encontrado."
    exit 1
fi

# 2. Criar a VENV
echo "🔨 Criando .venv..."
rm -rf .venv  # Limpa anterior se existir para evitar conflitos
$PY_CMD -m venv .venv

# 3. Ativar a VENV (Compatível com Linux e Windows)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "🐧 Venv ativada (Modo Linux/Mac)."
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "🪟 Venv ativada (Modo Windows)."
else
    echo "❌ Erro: Arquivo de ativação não encontrado."
    exit 1
fi

# 4. Instalar Dependências
echo "⬇️ Atualizando pip e instalando libs..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Tudo pronto! Para ativar manualmente no futuro, use:"
echo "   source .venv/bin/activate"

