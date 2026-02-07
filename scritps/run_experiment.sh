#!/bin/bash

# Encerra o script se houver erro (exceto durante o menu)
set -e

# --- 1. Configuração do Ambiente ---
# Limpa a tela
clear

echo "======================================================="
echo "   🚀  Time Series Forecasting Pipeline - Launcher"
echo "======================================================="

# Variável crítica para Windows/Git Bash + TensorFlow/Torch
export TF_ENABLE_ONEDNN_OPTS=0

# Ativação do Virtual Environment
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Ambiente Virtual (Windows) ativado."
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Ambiente Virtual (Linux/Mac) ativado."
else
    echo "❌ ERRO: Ambiente virtual (.venv) não encontrado!"
    echo "   Execute: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

echo ""

# --- 2. Modo Automação (Bypass do Menu) ---
# Se o usuário passou argumentos (ex: ./run.sh --mode report), roda direto
if [ $# -gt 0 ]; then
    echo "🔄 Modo direto detectado. Executando..."
    python -m src.main "$@"
    exit $?
fi

# --- 3. Menu Interativo ---
echo "Escolha uma opção de execução:"
echo ""
echo "  [1] 🏃 RODAR TUDO (Padrão)"
echo "      -> Treina modelos pendentes + Avalia + Gera Relatórios"
echo ""
echo "  [2] 📊 APENAS RELATÓRIOS (Rápido)"
echo "      -> Não treina nada. Apenas regera gráficos e tabelas dos modelos já salvos."
echo "      -> Use isso se alterou cores de gráficos ou quer recalcular métricas."
echo ""
echo "  [3] 🧠 APENAS TREINAMENTO"
echo "      -> Apenas processa os modelos, sem perder tempo gerando gráficos agora."
echo ""
echo "  [4] 🔥 FORÇAR RE-TREINO TOTAL (Cuidado!)"
echo "      -> Apaga o cache lógico e treina TUDO do zero (mesmo se já existir)."
echo ""
echo "  [0] Sair"
echo ""
echo "-------------------------------------------------------"
read -p "Digite o número da opção: " option

echo ""
echo "-------------------------------------------------------"

case $option in
    1)
        echo ">>> Iniciando Pipeline Completa..."
        python -m src.main --mode all
        ;;
    2)
        echo ">>> Gerando Apenas Relatórios..."
        python -m src.main --mode report
        ;;
    3)
        echo ">>> Iniciando Apenas Treinamento..."
        python -m src.main --mode train
        ;;
    4)
        echo ">>> ATENÇÃO: Forçando re-treinamento de todos os modelos..."
        python -m src.main --mode all --force
        ;;
    0)
        echo "Saindo..."
        exit 0
        ;;
    *)
        echo "❌ Opção inválida! Tente novamente."
        exit 1
        ;;
esac

echo ""
echo "✅ Processo finalizado."
# A LINHA ABAIXO É ONDE GERALMENTE OCORRE O ERRO DE COPIA
read -p "Pressione [Enter] para fechar..."