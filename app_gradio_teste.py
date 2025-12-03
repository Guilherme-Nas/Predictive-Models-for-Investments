# app_gradio_teste.py
import os
import random
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

# -----------------------
# Config / caminhos
# -----------------------
CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_PATH = 'models/model_gru_alt.keras'  # troque se quiser usar outro modelo
LOOKBACK = 200  # janela usada pelo modelo
DEFAULT_PLOT_DAYS = 120  # quantos dias do histórico mostrar no zoom

# -----------------------
# Carrega modelo (com fallback legível)
# -----------------------
try:
    model = load_model(MODEL_PATH)
    print(f'[INFO] Modelo carregado: {MODEL_PATH}')
except Exception as e:
    model = None
    print(f'[ERRO] Não foi possível carregar modelo {MODEL_PATH}: {e}')

# -----------------------
# Utilitários
# -----------------------
def load_csv():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]

def fetch_online_data(years=2):
    """Baixa histórico (últos `years` anos) incluindo hoje."""
    end = dt.datetime.now() + dt.timedelta(days=1)  # incluir dia atual
    start = end - dt.timedelta(days=365 * years)
    df = yf.download('BTC-USD', start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.shape[0] == 0:
        raise RuntimeError('Falha ao obter dados do Yahoo Finance.')
    df = df.reset_index()[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def fit_scaler_from_df(df):
    sc = MinMaxScaler()
    vals = df['Close'].astype(float).values.reshape(-1, 1)
    sc.fit(vals)
    return sc

def recursive_predict(scaled_seq, dias, model):
    """Recebe seq com shape (1, LOOKBACK, 1) ou (LOOKBACK,1) e prevê recursivamente dias passos.
       Retorna array 1D de valores escalados (não inverso-transformados)."""
    seq = np.copy(scaled_seq)
    if seq.ndim == 2:
        seq = np.expand_dims(seq, axis=0)  # (1, lookback, 1)
    preds_scaled = []
    for _ in range(dias):
        p = model.predict(seq, verbose=0)
        val = float(np.array(p).reshape(-1)[0])
        preds_scaled.append(val)
        # append mantendo shape (1, lookback, 1)
        seq = np.concatenate([seq[:, 1:, :], np.array(val).reshape(1, 1, 1)], axis=1)
    return np.array(preds_scaled).reshape(-1, 1)  # coluna

def safe_to_list(x):
    """Garante que seja 1D numpy array ou lista"""
    a = np.array(x)
    return a.flatten().tolist()

# -----------------------
# Plot helper
# -----------------------
def plot_forecast(dates_hist, hist_prices, dates_forecast, pred_prices, title='Previsão Bitcoin', real_future=None):
    """Plota histórico (dates_hist, hist_prices) e previsão (dates_forecast, pred_prices).
       real_future (opcional) é a série real dos dias previstos para comparação."""
    # Garantias de tipo
    dates_hist = pd.to_datetime(pd.Series(dates_hist)).reset_index(drop=True)
    hist_prices = np.array(hist_prices).flatten()
    dates_forecast = pd.to_datetime(pd.Series(dates_forecast)).reset_index(drop=True)
    pred_prices = np.array(pred_prices).flatten()

    if real_future is not None:
        real_future = np.array(real_future).flatten()
        if len(real_future) != len(pred_prices):
            real_future = None  # evitar incompatibilidade

    # Monta zoom: últimos DEFAULT_PLOT_DAYS do histórico (ou menos se não houver)
    n_hist = len(dates_hist)
    start_idx = max(0, n_hist - DEFAULT_PLOT_DAYS)
    zoom_dates_hist = dates_hist.iloc[start_idx:]
    zoom_hist_prices = hist_prices[start_idx:]

    # Y-limits considerando hist + preds + real_future (se existir)
    y_candidates = list(zoom_hist_prices) + list(pred_prices)
    if real_future is not None:
        y_candidates += list(real_future)
    y_candidates = np.array(y_candidates)
    y_min, y_max = float(y_candidates.min()), float(y_candidates.max())
    y_margin = max((y_max - y_min) * 0.12, 1.0)

    plt.figure(figsize=(10, 5))
    # histórico (linha contínua)
    plt.plot(dates_hist, hist_prices, label='Histórico (real)', color='tab:blue', linewidth=1.8)
    # previsão (a partir do dia seguinte ao último histórico)
    # desenhamos a previsão conectada: primeiro ponto = último preço real, depois preds
    last_real = hist_prices[-1]
    preds_full_dates = pd.concat([pd.Series([dates_hist.iloc[-1]]), dates_forecast], ignore_index=True)
    preds_full_values = np.concatenate([[last_real], pred_prices])

    plt.plot(preds_full_dates, preds_full_values, label='Previsão (modelo)', color='tab:orange', linewidth=1.8)

    # Se houver real_future, plotar sobre dates_forecast
    if real_future is not None:
        plt.plot(dates_forecast, real_future, label='Real futuro (CSV)', color='tab:green', linestyle='--', linewidth=1.6)

    # Linha vertical indicando começo da previsão (na última data histórica)
    plt.axvline(dates_hist.iloc[-1], linestyle='--', color='gray', alpha=0.7)
    plt.text(dates_hist.iloc[-1], y_max + y_margin*0.05, ' Início previsão', va='bottom', ha='left', fontsize=9, color='gray')

    # Ajustes visuais / zoom
    plt.xlim(zoom_dates_hist.iloc[0], dates_forecast.iloc[-1])
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()

    temp_path = os.path.join(tempfile.gettempdir(), 'forecast_plot_debug.png')
    plt.savefig(temp_path, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    return temp_path

# -----------------------
# Modos
# -----------------------
def previsao_csv_teste(dias):
    """Escolhe ponto aleatório no CSV, prevê dias à frente e compara com real."""
    if model is None:
        return None, 'Modelo não carregado.'

    df = load_csv()
    n = len(df)
    if n < LOOKBACK + dias + 1:
        return None, 'CSV muito curto para este teste.'

    min_idx = LOOKBACK
    max_idx = n - dias - 1
    idx = random.randint(min_idx, max_idx)
    base_date = df['Date'].iloc[idx]

    print(f'[DEBUG] CSV Teste: n={n}, idx={idx}, base_date={base_date.date()}, dias={dias}')

    # escala conforme este trecho do CSV
    scaler = fit_scaler_from_df(df)
    scaled = scaler.transform(df[['Close']].values.reshape(-1, 1))

    seq_scaled = scaled[idx - LOOKBACK: idx]  # shape (LOOKBACK,1)
    preds_scaled = recursive_predict(seq_scaled, dias, model)  # shape (dias,1)
    preds = scaler.inverse_transform(preds_scaled).reshape(-1)

    # verdadeiros
    reais = df['Close'].iloc[idx: idx + dias].values
    # datas
    dates_hist = df['Date'].iloc[idx - LOOKBACK: idx].reset_index(drop=True)
    dates_future = df['Date'].iloc[idx: idx + dias].reset_index(drop=True)

    # métricas
    if len(reais) == len(preds) and np.all(reais != 0):
        mape = np.mean(np.abs((preds - reais) / reais)) * 100
        msg = f'CSV Teste — idx {idx} ({base_date.date()}) | MAPE {mape:.2f}%'
    else:
        msg = f'CSV Teste — idx {idx} ({base_date.date()}) | sem dados reais suficientes'

    # plotando: histórico (até idx-1), previsão (dias) e real_future
    img_path = plot_forecast(
        dates_hist=dates_hist,
        hist_prices=df['Close'].iloc[idx - LOOKBACK: idx].values,
        dates_forecast=dates_future,
        pred_prices=preds,
        title=msg,
        real_future=reais
    )
    return img_path, msg

def previsao_online(dias):
    """Busca dados online e prevê dias à frente (conecta corretamente)."""
    if model is None:
        return None, 'Modelo não carregado.'
    try:
        df = fetch_online_data()
    except Exception as e:
        return None, f'Erro ao obter dados online: {e}'

    if len(df) < LOOKBACK + 1:
        return None, 'Dados online insuficientes para previsão.'

    print(f'[DEBUG] Online: data final do histórico = {df["Date"].iloc[-1].date()} | pontos = {len(df)}')

    scaler = fit_scaler_from_df(df)
    scaled = scaler.transform(df[['Close']].values.reshape(-1, 1))
    seq_scaled = scaled[-LOOKBACK:]  # (LOOKBACK,1)
    preds_scaled = recursive_predict(seq_scaled, dias, model)  # (dias,1)
    preds = scaler.inverse_transform(preds_scaled).reshape(-1)

    # datas futuras (a partir do próximo dia)
    last_day = df['Date'].iloc[-1]
    dates_future = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=dias, freq='D')

    img_path = plot_forecast(
        dates_hist=df['Date'].iloc[-LOOKBACK:].reset_index(drop=True),
        hist_prices=df['Close'].iloc[-LOOKBACK:].values,
        dates_forecast=pd.Series(dates_future),
        pred_prices=preds,
        title=f'Online Atualizado — previsão {dias} dias'
    )
    return img_path, f'Previsão online até {dates_future[-1].date()} pronta.'

# -----------------------
# Interface Gradio
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔎 App Teste — Previsão Bitcoin (GRU)")

    modo = gr.Radio(['csv_test', 'online'], value='csv_test', label='Modo')
    dias = gr.Slider(minimum=1, maximum=7, step=1, value=3, label='Dias de previsão')

    btn = gr.Button('Gerar previsão')
    out_img = gr.Image(label='Gráfico de previsão')
    out_txt = gr.Textbox(label='Resultado', lines=2)

    def runner(modo, dias):
        print(f'[RUN] Modo={modo} | dias={dias}')
        if modo == 'csv_test':
            return previsao_csv_teste(dias)
        else:
            return previsao_online(dias)

    btn.click(runner, inputs=[modo, dias], outputs=[out_img, out_txt])

if __name__ == '__main__':
    demo.launch()
