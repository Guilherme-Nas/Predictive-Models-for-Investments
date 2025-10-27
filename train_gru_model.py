# train_gru_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Caminhos
DATA_PATH = "notebooks/data/btc_limpo.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_gru.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Garantir pasta models
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Carregar dados
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
data = df["Close"].values.reshape(-1, 1)

# 2. Normalizar
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 3. Criar sequências
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i])
    return np.array(X), np.array(y)


SEQ_LEN = 60
X, y = create_sequences(data_scaled, SEQ_LEN)

# 4. Separar treino/teste
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. Criar modelo GRU
model = tf.keras.Sequential(
    [
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1),
    ]
)

# 6. Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError()
)

# 7. Treinar
EPOCHS = 20
BATCH_SIZE = 32

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

# 8. Salvar modelo e scaler
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n✅ Modelo salvo em: {MODEL_PATH}")
print(f"✅ Scaler salvo em: {SCALER_PATH}")
