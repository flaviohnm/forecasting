#!/bin/bash

# Encerra o script se houver erro (exceto durante o menu)
set -e

# --- 1. Configuração do Ambiente ---
clear
echo "======================================================="
echo "   🚀  Time Series Forecasting Pipeline - Launcher"
echo "======================================================="

export TF_ENABLE_ONEDNN_OPTS=0

# Ativação do Virtual Environment (Compatível Linux/Mac/Windows)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Ambiente Virtual (Linux/Mac) ativado."
    PYTHON_CMD="python"
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Ambiente Virtual (Windows) ativado."
    PYTHON_CMD="python"
else
    echo "❌ ERRO: Ambiente virtual (.venv) não encontrado!"
    echo "   Execute: ./scripts/setup_venv.sh"
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
echo "      -> Treina modelos pendentes + Avalia + Gera Relatórios"
echo ""
echo "  [2] 📊 APENAS RELATÓRIOS (Rápido)"
echo "      -> Não treina nada. Apenas regera gráficos e tabelas."
echo ""
echo "  [3] 🧠 APENAS TREINAMENTO"
echo "      -> Apenas processa os modelos, sem gráficos."
echo ""
echo "  [4] 🔥 FORÇAR RE-TREINO TOTAL (Cuidado!)"
echo "      -> Apaga cache e treina TUDO do zero."
echo ""
echo "  [0] Sair"
echo ""
echo "-------------------------------------------------------"
read -p "Digite o número da opção: " option
echo ""

case $option in
    1)
        echo ">>> Iniciando Pipeline Completa..."
        $PYTHON_CMD -m src.main --mode all
        ;;
    2)
        echo ">>> Gerando Apenas Relatórios..."
        $PYTHON_CMD -m src.main --mode report
        ;;
    3)
        echo ">>> Iniciando Apenas Treinamento..."
        $PYTHON_CMD -m src.main --mode train
        ;;
    4)
        echo ">>> ATENÇÃO: Forçando re-treinamento..."
        $PYTHON_CMD -m src.main --mode all --force
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

echo ""
echo "✅ Processo finalizado."
# read -p removido ou ajustado para não travar automação, 
# mas mantido aqui para uso interativo.
read -p "Pressione [Enter] para fechar..."