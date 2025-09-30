import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

def build_nbeats_model(input_shape: tuple, output_shape: int, n_neurons: list, learning_rate: float):
    """
    Constrói um modelo simples de rede neural (similar a um bloco do N-BEATS).
    """
    model = Sequential()
    model.add(Dense(n_neurons[0], activation='relu', input_shape=input_shape))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_shape))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_save_nbeats(X_train, y_train, model_path: str, model_params: dict):
    """
    Treina e salva um modelo N-BEATS para um horizonte específico.
    """
    print(f"Treinando modelo N-BEATS para {os.path.basename(model_path)}...")
    
    input_shape = (X_train.shape[1],)
    output_shape = 1

    model = build_nbeats_model(
        input_shape,
        output_shape,
        n_neurons=model_params['n_neurons'],
        learning_rate=model_params['learning_rate']
    )

    model.fit(
        X_train,
        y_train,
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        verbose=0 # Mude para 1 se quiser ver o progresso de cada época
    )
    print("Treinamento N-BEATS concluído.")

    model.save(model_path)
    print(f"Modelo N-BEATS salvo em: {model_path}")

def load_and_predict_direct(model_path: str, input_data):
    """
    Carrega um modelo N-BEATS e faz uma previsão direta.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo N-BEATS não encontrado em {model_path}")
        
    model = load_model(model_path)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
        
    prediction = model.predict(input_data, verbose=0)
    return prediction[0, 0]