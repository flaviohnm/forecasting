#!/bin/bash

# Encerra se der erro
set -e

echo "======================================================="
echo "   🛠️  Setup Inteligente de Ambiente (Auto-GPU)"
echo "======================================================="

# 1. Definir qual Python usar
if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    echo "❌ Erro: Nenhum Python 3 encontrado."
    exit 1
fi
echo "🐍 Usando Python: $PY_CMD"

# 2. Criar a VENV (Limpa anterior se existir)
if [ -d ".venv" ]; then
    echo "♻️  Recriando .venv..."
    rm -rf .venv
else
    echo "🔨 Criando .venv..."
fi
$PY_CMD -m venv .venv

# 3. Ativar a VENV
source .venv/bin/activate
echo "✅ Venv ativada."

# 4. Atualizar PIP
pip install --upgrade pip

# --- 5. INSTALAÇÃO DO PYTORCH (A Mágica Acontece Aqui) ---
echo "🔍 Verificando hardware..."

if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU NVIDIA detectada! Instalando versão com CUDA..."
    # Instala PyTorch compatível com CUDA 12.x
    # O --no-cache-dir evita pegar versões cacheadas antigas de CPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "🐢 Nenhuma GPU detectada (ou driver ausente). Instalando versão CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 6. Instalar o Restante das Dependências
echo "📦 Instalando outras bibliotecas do requirements.txt..."
# O pip vai ver que o torch já está instalado e não vai tentar baixar a versão errada
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "✅ Instalação Concluída!"
echo "   Para testar, rode: python -c 'import torch; print(torch.cuda.is_available())'"
echo "======================================================="