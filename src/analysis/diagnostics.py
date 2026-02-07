import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

def save_residual_diagnostics(y_true, y_pred, model_name, dataset_name, horizon, save_path):
    """
    Gera e salva 3 gráficos de diagnóstico: Resíduos, ACF e PACF.
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        has_statsmodels = True
    except ImportError:
        print("   -> [AVISO] 'statsmodels' não instalado. ACF/PACF serão pulados.")
        has_statsmodels = False

    try:
        # Tratamento de dados (Garante numérico)
        y_true = pd.to_numeric(y_true, errors='coerce').to_numpy()
        y_pred = pd.to_numeric(y_pred, errors='coerce').to_numpy()
        
        # Remove NaNs
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        residuals = y_true[mask] - y_pred[mask]

        if len(residuals) < 2:
            return

        # Configurar Figura
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f'Diagnóstico Residual: {model_name} (H={horizon})', fontsize=16)

        # Plot 1: Resíduos no Tempo
        axes[0].plot(residuals, color='black', lw=1, alpha=0.7)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title('Resíduos no Tempo')
        axes[0].set_ylabel('Erro')

        # Plot 2 & 3: ACF e PACF
        lags = min(40, len(residuals) // 2 - 1)
        if has_statsmodels and lags > 1:
            plot_acf(residuals, ax=axes[1], lags=lags, title='Autocorrelação (ACF)', auto_ylims=True)
            plot_pacf(residuals, ax=axes[2], lags=lags, title='Autocorrelação Parcial (PACF)', method='ywm', auto_ylims=True)
        
        plt.tight_layout()
        
        # Salvar
        os.makedirs(save_path, exist_ok=True)
        filename = f"diag_{model_name}.png"
        full_path = os.path.join(save_path, filename)
        
        plt.savefig(full_path)
        plt.close(fig)
        
        print(f"   -> [DIAG] Gráfico salvo em: {full_path}")

    except Exception as e:
        print(f"   -> [DIAG] ERRO ao gerar gráfico: {e}")
        traceback.print_exc()