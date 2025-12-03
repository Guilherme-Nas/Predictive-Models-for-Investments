import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import os

# === Caminho do CSV e saída ===
CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_PATH = 'models/model_gru_ret.keras'

# === Carrega dados ===
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df[['Date', 'Close']]

# === Calcula retornos percentuais ===
df['Return'] = df['Close'].pct_change() * 100
df.dropna(inplace=True)

# === Normaliza ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Return']])

# === Cria janelas ===
lookback = 60
X, y = [], []
for i in range(lookback, len(scaled)):
    X.append(scaled[i - lookback:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

# === Modelo GRU ===
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    GRU(64),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# === Treinamento ===
es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=[es])

# === Salva modelo ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f'\n✅ Modelo salvo em: {MODEL_PATH}')
