import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class TimeSeriesScaler:
    """
    Wrapper para StandardScaler que respeita a estrutura de Séries Temporais (unique_id, ds, y).
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_col = 'y'
        self.is_fitted = False

    def fit(self, df):
        """Aprende a média e desvio padrão apenas do conjunto de treino"""
        if self.target_col in df.columns:
            # Reshape necessário para o sklearn (n_samples, n_features)
            self.scaler.fit(df[[self.target_col]])
            self.is_fitted = True
            logging.info(f"Scaler ajustado. Média: {self.scaler.mean_[0]:.4f}, Var: {self.scaler.var_[0]:.4f}")
        return self

    def transform(self, df):
        """Aplica a normalização (z-score)"""
        if not self.is_fitted:
            raise ValueError("O scaler precisa ser treinado (fit) antes de usar transform.")
        
        df_scaled = df.copy()
        if self.target_col in df_scaled.columns:
            df_scaled[self.target_col] = self.scaler.transform(df_scaled[[self.target_col]])
        return df_scaled

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform(self, df_or_array):
        """
        Reverte a normalização para as previsões voltarem à escala real.
        Aceita DataFrame (com coluna 'y' ou 'y_hat') ou numpy array.
        """
        if not self.is_fitted:
            raise ValueError("O scaler precisa ser treinado antes de inverter.")

        if isinstance(df_or_array, pd.DataFrame):
            df_inv = df_or_array.copy()
            # Reverte 'y' se existir
            if 'y' in df_inv.columns:
                df_inv['y'] = self.scaler.inverse_transform(df_inv[['y']])
            # Reverte 'y_hat' (previsão) se existir
            if 'y_hat' in df_inv.columns:
                df_inv['y_hat'] = self.scaler.inverse_transform(df_inv[['y_hat']])
            return df_inv
        
        elif isinstance(df_or_array, np.ndarray):
            # Assume array 1D ou 2D da coluna alvo
            shape_original = df_or_array.shape
            datos_reshaped = df_or_array.reshape(-1, 1)
            inv = self.scaler.inverse_transform(datos_reshaped)
            return inv.reshape(shape_original)
        
        else:
            return df_or_array