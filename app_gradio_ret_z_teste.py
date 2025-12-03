import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tempfile
import os
import random

CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_RET = 'models/model_gru_ret.keras'
MODEL_Z = 'models/model_gru_z.keras'

# Carrega modelos
model_ret = load_model(MODEL_RET)
model_z = load_model(MODEL_Z)


def load_csv():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]


def plot_results(df, preds_ret, preds_z, idx, dias):
    datas_hist = df['Date'].iloc[idx - 200:idx]
    hist = df['Close'].iloc[idx - 200:idx]
    datas_future = df['Date'].iloc[idx:idx + dias].reset_index(drop=True)
    reais = df['Close'].iloc[idx:idx + dias].values

    plt.figure(figsize=(10, 5))
    plt.plot(datas_hist, hist, label='Histórico real', color='tab:blue')
    plt.plot(datas_future, reais, label='Real futuro', color='tab:green', linestyle='--')
    plt.plot(datas_future, preds_ret, label='Pred (retornos)', color='tab:orange')
    plt.plot(datas_future, preds_z, label='Pred (z-score)', color='tab:red')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), 'compare_ret_z.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def testar_modelos(dias):
    df = load_csv()

    # ---- Modelo RET ----
    df_ret = df.copy()
    df_ret['Return'] = df_ret['Close'].pct_change() * 100
    df_ret.dropna(inplace=True)
    scaler_ret = MinMaxScaler()
    scaled_ret = scaler_ret.fit_transform(df_ret[['Return']])

    lookback = 60
    min_idx = lookback
    max_idx = len(scaled_ret) - dias - 1
    idx = random.randint(min_idx, max_idx)

    seq = scaled_ret[idx - lookback:idx].reshape(1, lookback, 1)
    seq_copy = np.copy(seq)
    preds_ret_scaled = []
    for _ in range(dias):
        pred = model_ret.predict(seq_copy, verbose=0)
        preds_ret_scaled.append(pred[0, 0])
        seq_copy = np.append(seq_copy[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    preds_ret = scaler_ret.inverse_transform(np.array(preds_ret_scaled).reshape(-1, 1)).flatten()

    # ---- Modelo Z ----
    mean = df['Close'].mean()
    std = df['Close'].std()
    df_z = df.copy()
    df_z['Close_Z'] = (df_z['Close'] - mean) / std
    scaled_z = df_z['Close_Z'].values

    seq = scaled_z[idx - lookback:idx].reshape(1, lookback, 1)
    seq_copy = np.copy(seq)
    preds_z_scaled = []
    for _ in range(dias):
        pred = model_z.predict(seq_copy, verbose=0)
        preds_z_scaled.append(pred[0, 0])
        seq_copy = np.append(seq_copy[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    preds_z = np.array(preds_z_scaled) * std + mean

    # ---- Plot ----
    path = plot_results(df, preds_ret, preds_z, idx, dias)
    return path, f"Teste concluído (índice {idx})"


with gr.Blocks() as demo:
    gr.Markdown("## Comparação: GRU Retornos (%) vs GRU Z-score (modo CSV Teste)")
    dias = gr.Slider(1, 7, value=3, step=1, label='Dias de previsão')
    botao = gr.Button('Executar Teste')
    img = gr.Image(label='Comparativo')
    txt = gr.Textbox(label='Log', lines=2)
    botao.click(testar_modelos, inputs=[dias], outputs=[img, txt])

demo.launch()
