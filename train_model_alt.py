import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_OUT = 'models/model_gru_alt.keras'
LOOKBACK = 200
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0005
VAL_SPLIT = 0.1

os.makedirs('models', exist_ok=True)

# ============================================================
# CARREGAR DADOS
# ============================================================
print('[INFO] Carregando CSV...')
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# ============================================================
# CRIAR SEQUÊNCIAS
# ============================================================
def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

print('[INFO] Criando sequências...')
X, y = create_sequences(prices_scaled, LOOKBACK)
print(f'[DEBUG] X: {X.shape}, y: {y.shape}')

# ============================================================
# CONSTRUIR MODELO
# ============================================================
print('[INFO] Construindo modelo GRU alternativo...')
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='mse')

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss')
]

# ============================================================
# TREINAR
# ============================================================
print('[INFO] Iniciando treino...')
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# SALVAR MODELO FINAL
# ============================================================
model.save(MODEL_OUT)
print(f'✅ Modelo alternativo salvo em: {MODEL_OUT}')

# ============================================================
# AVALIAÇÃO FINAL
# ============================================================
val_loss = min(history.history['val_loss'])
print(f'[RESULT] Melhor val_loss: {val_loss:.8f}')
