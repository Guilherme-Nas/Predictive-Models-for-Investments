import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# Caminhos fixos
MODEL_PATH = "models/model_gru.keras"
DATA_PATH = "notebooks/data/btc_limpo.csv"

# Carregar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar e preparar dados
df = pd.read_csv(DATA_PATH)
prices = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)


# Função de previsão
def predict(horizon):
    try:
        horizon = int(horizon)
        last_seq = scaled_data[-60:].reshape(1, 60, 1)
        preds = []

        for _ in range(horizon):
            pred = model.predict(last_seq, verbose=0)
            preds.append(pred[0, 0])
            last_seq = np.concatenate(
                [last_seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1
            )

        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        # Determinar tendência da previsão
        trend_up = preds[-1][0] > prices[-1][0]
        color_pred = "green" if trend_up else "red"

        # Dados para o gráfico
        history_len = 200
        hist_data = prices[-history_len:]
        hist_x = np.arange(history_len)
        pred_x = np.arange(history_len, history_len + horizon)

        plt.figure(figsize=(8, 4))
        plt.plot(
            hist_x,
            hist_data,
            label="Histórico (últimos 200 dias)",
            color="blue",
            linewidth=1.8,
        )
        plt.plot(
            pred_x,
            preds,
            label="Previsão",
            color=color_pred,
            linewidth=2.3,
            marker="o",
            markersize=6,
        )
        plt.axvline(x=history_len, color="gray", linestyle="--", alpha=0.5)

        direction_text = "⬆️ Alta prevista" if trend_up else "⬇️ Queda prevista"
        plt.title(
            f"Previsão do Bitcoin ({horizon} dias futuros) — {direction_text}",
            fontsize=13,
            pad=10,
        )
        plt.xlabel("Dias (relativos)")
        plt.ylabel("Preço (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        # Converter gráfico em imagem
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        img = plt.imread(buf, format="png")

        resumo = (
            f"📊 Último preço real: ${prices[-1][0]:,.2f}\n"
            f"💡 Previsão final ({horizon} dias): ${preds[-1][0]:,.2f}\n"
            f"{'🟢 Tendência de alta' if trend_up else '🔴 Tendência de queda'}"
        )

        return img, resumo

    except Exception as e:
        return None, f"⚠️ Erro durante a previsão: {str(e)}"


# Interface Gradio
with gr.Blocks(title="Bitcoin Forecast Dashboard") as demo:
    gr.Markdown("## 💹 Previsão de Preço do Bitcoin com GRU")
    gr.Markdown("Ajuste o número de dias futuros que deseja prever:")

    horizon_input = gr.Slider(
        label="Horizonte de previsão (dias)", minimum=1, maximum=7, step=1, value=3
    )

    output_image = gr.Image(label="Gráfico de Previsão")
    output_text = gr.Textbox(label="Resumo")

    btn = gr.Button("Gerar previsão")
    btn.click(predict, inputs=horizon_input, outputs=[output_image, output_text])

if __name__ == "__main__":
    demo.launch()
