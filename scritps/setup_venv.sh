#!/bin/bash

# Encerra se der erro
set -e

echo "======================================================="
echo "   🛠️  Setup Inteligente de Ambiente (Universal)"
echo "======================================================="

# --- 1. Definir qual Python usar (Compatível Win/Linux) ---
# Tenta achar o 3.11, depois python3, depois python (comum no Windows)
if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "❌ Erro: Nenhum Python encontrado no PATH."
    exit 1
fi
echo "🐍 Usando interpretador: $PY_CMD"

# --- 2. Criar a VENV ---
if [ -d ".venv" ]; then
    echo "♻️  Recriando .venv (limpando antiga)..."
    rm -rf .venv
else
    echo "🔨 Criando nova .venv..."
fi
$PY_CMD -m venv .venv

# --- 3. Ativar a VENV (Ajuste Crítico Cross-Platform) ---
# Verifica se estamos no Linux/Mac (bin) ou Windows (Scripts)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "🐧 Venv ativada (Modo Linux/Mac)."
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "🪟 Venv ativada (Modo Windows)."
else
    echo "❌ Erro: Script de ativação não encontrado!"
    exit 1
fi

# --- 4. Atualizar PIP ---
echo "⬇️  Atualizando pip..."
python -m pip install --upgrade pip

# --- 5. INSTALAÇÃO DO PYTORCH (Auto-GPU) ---
echo "🔍 Verificando hardware para PyTorch..."

# Tenta detectar nvidia-smi (funciona no Linux e no Git Bash do Windows se drivers instalados)
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU NVIDIA detectada! Instalando versão com CUDA..."
    
    # Nota: No Windows o pip as vezes precisa de cache limpo para trocar de CPU para GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "🐢 Nenhuma GPU detectada. Instalando versão CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# --- 6. Instalar Dependências ---
echo "📦 Instalando requisitos do requirements.txt..."
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "✅ Setup Finalizado com Sucesso!"
echo "   Para testar a GPU, rode:"
echo "   python -c 'import torch; print(f\"GPU Disponível: {torch.cuda.is_available()}\")'"
echo "======================================================="