import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_PATH = 'models/model_gru_z.keras'

# === Carrega dados ===
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df[['Date', 'Close']]

# === Normalização Z-score ===
mean = df['Close'].mean()
std = df['Close'].std()
df['Close_Z'] = (df['Close'] - mean) / std

# === Cria janelas ===
lookback = 60
X, y = [], []
scaled = df['Close_Z'].values
for i in range(lookback, len(scaled)):
    X.append(scaled[i - lookback:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

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
