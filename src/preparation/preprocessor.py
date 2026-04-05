import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesScaler:
    """
    Wrapper para StandardScaler que respeita a estrutura de Séries Temporais.
    Flexível para normalizar a série original ('y') ou os resíduos ('residual').
    """

    def __init__(self, target_col="y"):
        self.scaler = StandardScaler()
        self.target_col = target_col
        self.is_fitted = False

    def fit(self, df):
        """Aprende a média e variância APENAS do conjunto de treino (Prevenção de Data Leakage)"""
        if self.target_col not in df.columns:
            raise ValueError(f"Coluna {self.target_col} não encontrada para o fit.")

        self.scaler.fit(df[[self.target_col]])
        self.is_fitted = True
        logging.info(
            f"Scaler ajustado para '{self.target_col}'. Média: {self.scaler.mean_[0]:.4f}, Var: {self.scaler.var_[0]:.4f}"
        )
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

    def inverse_transform(self, df_or_array, invert_col=None):
        """
        Reverte a normalização. Pode reverter colunas específicas (ex: 'y' ou 'y_hat').
        """
        if not self.is_fitted:
            raise ValueError("O scaler precisa ser treinado antes de inverter.")

        col_to_invert = invert_col if invert_col else self.target_col

        if isinstance(df_or_array, pd.DataFrame):
            df_inv = df_or_array.copy()
            if col_to_invert in df_inv.columns:
                df_inv[col_to_invert] = self.scaler.inverse_transform(df_inv[[col_to_invert]])
            return df_inv

        elif isinstance(df_or_array, np.ndarray):
            shape_original = df_or_array.shape
            datos_reshaped = df_or_array.reshape(-1, 1)
            datos_inv = self.scaler.inverse_transform(datos_reshaped)
            return datos_inv.reshape(shape_original)
        else:
            raise TypeError("A entrada deve ser um DataFrame Pandas ou Numpy Array.")


def split_time_series(df, horizon, val_size=None):
    """
    Realiza o split temporal estrito.
    Teste = Últimos 'horizon' passos.
    Validação = 'val_size' passos antes do Teste.
    Treino = Todo o restante inicial.
    """
    df_sorted = df.sort_values("ds").reset_index(drop=True)
    total_len = len(df_sorted)

    if val_size is None:
        # Se não fornecido, a validação tem o mesmo tamanho do horizonte de teste
        val_size = horizon

    if total_len < (horizon + val_size):
        raise ValueError("Dataset muito pequeno para o horizonte e tamanho de validação exigidos.")

    test_start_idx = total_len - horizon
    val_start_idx = test_start_idx - val_size

    train_df = df_sorted.iloc[:val_start_idx].copy()
    val_df = df_sorted.iloc[val_start_idx:test_start_idx].copy()
    test_df = df_sorted.iloc[test_start_idx:].copy()

    logging.info(f"Split concluído -> Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")

    return train_df, val_df, test_df
