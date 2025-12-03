# main.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# Caminhos
MODEL_PATH = "models/model_gru.keras"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "notebooks/data/btc_limpo.csv"

# 1. Carregar scaler e modelo
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Carregar dados
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
data = df["Close"].values.reshape(-1, 1)
data_scaled = scaler.transform(data)

# 3. Criar sequência para previsão
SEQ_LEN = 60
last_sequence = data_scaled[-SEQ_LEN:]
X_input = np.expand_dims(last_sequence, axis=0)

# 4. Fazer previsão
prediction_scaled = model.predict(X_input)
prediction = scaler.inverse_transform(prediction_scaled)

print(f"\n Preço previsto do Bitcoin (próximo dia): ${prediction[0][0]:.2f}")
