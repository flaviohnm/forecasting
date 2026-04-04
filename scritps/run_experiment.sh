#!/bin/bash

# Encerra o script se houver erro (exceto durante o menu)
set -e

# --- 1. Configuração do Ambiente ---
clear
echo "======================================================="
echo "   🚀  Time Series Forecasting Pipeline - Universal"
echo "======================================================="

export TF_ENABLE_ONEDNN_OPTS=0

# Verifica se o Poetry está acessível
if ! command -v poetry &> /dev/null; then
    echo "❌ ERRO: Poetry não encontrado! Execute o setup_venv.sh primeiro."
    exit 1
fi

echo "✅ Ambiente gerenciado pelo Poetry."
echo ""

# --- 2. Modo Automação ---
if [ $# -gt 0 ]; then
    echo "🔄 Modo direto detectado. Executando..."
    poetry run python -m src.main "$@"
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
read -p "Opção: " OPCAO

echo ""
case $OPCAO in
    1)
        poetry run python -m src.main --mode all
        ;;
    2)
        poetry run python -m src.main --mode report
        ;;
    3)
        poetry run python -m src.main --mode train
        ;;
    4)
        poetry run python -m src.main --mode all --force
        ;;
    0)
        echo "Saindo..."
        exit 0
        ;;
    *)
        echo "❌ Opção inválida!"
        exit 1
        ;;
esac