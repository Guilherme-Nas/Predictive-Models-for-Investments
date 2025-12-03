import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import random
import tempfile
import os
import datetime as dt

# === Caminhos fixos ===
CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_PATH = 'models/model_gru.keras'

# === Carrega modelo ===
model = load_model(MODEL_PATH)


# === Funções básicas ===
def load_csv():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]


def fetch_online_data():
    """Obtém dados reais do BTC até hoje (últimos 2 anos)."""
    end = dt.datetime.now() + dt.timedelta(days=1)  # inclui o dia atual
    start = end - dt.timedelta(days=730)  # 2 anos de histórico
    df = yf.download('BTC-USD', start=start, end=end, progress=False)
    if df is None or df.shape[0] == 0:
        raise RuntimeError('Falha ao obter dados do Yahoo Finance.')
    df = df.reset_index()[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    return scaled, scaler


def plot_forecast(dates_hist, hist_prices, dates_forecast, pred_prices, title='Previsão Bitcoin', real_future=None):
    """Plota histórico, previsão e (opcionalmente) dados reais futuros."""
    plt.figure(figsize=(10, 5))

    # Histórico
    plt.plot(dates_hist, hist_prices, label='Histórico real', color='tab:blue', linewidth=1.8)

    # Previsão (sem marcadores)
    plt.plot(dates_forecast, pred_prices, label='Previsão (modelo)', color='tab:orange', linewidth=1.8)

    # Se existir série real futura (CSV Teste)
    if real_future is not None:
        plt.plot(dates_forecast, real_future, label='Real futuro', color='tab:green', linestyle='--', linewidth=1.8)

    # Linha vertical de separação
    if len(dates_hist) > 0:
        plt.axvline(dates_hist.iloc[-1], linestyle='--', color='gray', alpha=0.7)
        plt.text(dates_hist.iloc[-1], plt.ylim()[1], ' Início previsão', va='top', ha='left', fontsize=9, color='gray')

    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    temp_path = os.path.join(tempfile.gettempdir(), 'forecast_plot.png')
    plt.savefig(temp_path, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    return temp_path


# === Modo CSV Teste ===
def previsao_csv_teste(dias):
    df = load_csv()
    lookback = 200
    scaled, scaler = prepare_data(df)

    # garante espaço suficiente
    if len(scaled) < lookback + dias:
        return None, 'CSV muito curto para esse teste.'

    # escolhe ponto aleatório dentro dos limites
    min_idx = lookback
    max_idx = len(scaled) - dias - 1
    idx = random.randint(min_idx, max_idx)

    seq = scaled[idx - lookback:idx].reshape(1, lookback, 1)
    seq_copy = np.copy(seq)

    preds_scaled = []
    for _ in range(dias):
        pred = model.predict(seq_copy, verbose=0)
        preds_scaled.append(pred[0, 0])
        seq_copy = np.append(seq_copy[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # dados reais para comparação
    reais = df['Close'].iloc[idx:idx + dias].values
    datas_hist = df['Date'].iloc[idx - lookback:idx]
    hist_prices = df['Close'].iloc[idx - lookback:idx]
    datas_future = df['Date'].iloc[idx:idx + dias].reset_index(drop=True)

    # cálculo do erro
    if len(reais) == len(preds):
        mape = np.mean(np.abs((preds - reais) / reais)) * 100
        msg = f'Modo CSV Teste — MAPE: {mape:.2f}% (índice {idx})'
    else:
        msg = f'Modo CSV Teste — sem dados reais suficientes após o ponto escolhido.'

    img_path = plot_forecast(
        dates_hist=datas_hist,
        hist_prices=hist_prices,
        dates_forecast=datas_future,
        pred_prices=preds,
        real_future=reais,
        title=msg
    )
    return img_path, msg


# === Modo Online Atualizado ===
def previsao_online(dias):
    df = fetch_online_data()
    lookback = 200
    scaled, scaler = prepare_data(df)

    if len(scaled) < lookback:
        return None, 'Dados online insuficientes para previsão.'

    seq = scaled[-lookback:].reshape(1, lookback, 1)
    seq_copy = np.copy(seq)

    preds_scaled = []
    for _ in range(dias):
        pred = model.predict(seq_copy, verbose=0)
        preds_scaled.append(pred[0, 0])
        seq_copy = np.append(seq_copy[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    datas_hist = df['Date'].iloc[-lookback:]
    hist_prices = df['Close'].iloc[-lookback:]
    last_day = df['Date'].iloc[-1]
    datas_future = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=dias, freq='D')

    img_path = plot_forecast(
        dates_hist=datas_hist,
        hist_prices=hist_prices,
        dates_forecast=datas_future,
        pred_prices=preds,
        title=f'Previsão Atualizada — {dias} dias à frente (dados reais)'
    )
    return img_path, f'Previsão atualizada até {datas_future[-1].strftime("%d/%m/%Y")} concluída.'


# === Interface Gradio ===
with gr.Blocks() as demo:
    gr.Markdown("## Previsão do Preço do Bitcoin (Simplificada — GRU Model)")

    modo = gr.Radio(['CSV Teste', 'Online Atualizado'], value='Online Atualizado', label='Modo de operação')
    dias = gr.Slider(1, 7, value=3, step=1, label='Dias de previsão')

    botao = gr.Button('Gerar previsão')
    output_img = gr.Image(label='Gráfico de previsão')
    output_txt = gr.Textbox(label='Resultado', lines=2)

    def executar(modo, dias):
        if modo == 'CSV Teste':
            return previsao_csv_teste(dias)
        else:
            return previsao_online(dias)

    botao.click(executar, inputs=[modo, dias], outputs=[output_img, output_txt])

demo.launch()
