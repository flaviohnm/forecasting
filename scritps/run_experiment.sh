#!/bin/bash

# Encerra o script se houver erro (exceto durante o menu)
set -e

# --- 1. Configuração do Ambiente ---
clear
echo "======================================================="
echo "   🚀  Time Series Forecasting Pipeline - Universal"
echo "======================================================="

export TF_ENABLE_ONEDNN_OPTS=0

# --- CORREÇÃO AQUI: Tenta Linux (bin) PRIMEIRO ---
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Ambiente Virtual (Linux/Mac) ativado."
    PYTHON_CMD="python"
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Ambiente Virtual (Windows) ativado."
    PYTHON_CMD="python"
else
    echo "❌ ERRO: Ambiente virtual (.venv) não encontrado ou incompleto!"
    echo "   Execute os comandos de reset abaixo."
    exit 1
fi

echo ""

# --- 2. Modo Automação ---
if [ $# -gt 0 ]; then
    echo "🔄 Modo direto detectado. Executando..."
    $PYTHON_CMD -m src.main "$@"
    exit $?
fi

# --- 3. Menu Interativo ---
echo "Escolha uma opção de execução:"
echo ""
echo "  [1] 🏃 RODAR TUDO (Padrão)"
echo "  [2] 📊 APENAS RELATÓRIOS (Rápido)"
echo "  [3] 🧠 APENAS TREINAMENTO"
echo "  [4] 🔥 FORÇAR RE-TREINO TOTAL"
echo "  [0] Sair"
echo ""
read -p "Opção: " option
echo ""

case $option in
    1) $PYTHON_CMD -m src.main --mode all ;;
    2) $PYTHON_CMD -m src.main --mode report ;;
    3) $PYTHON_CMD -m src.main --mode train ;;
    4) $PYTHON_CMD -m src.main --mode all --force ;;
    0) exit 0 ;;
    *) echo "❌ Opção inválida!"; exit 1 ;;
esac

echo ""
echo "✅ Processo finalizado."