# app_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Caminhos
MODEL_PATH = "models/model_gru.keras"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "notebooks/data/btc_limpo.csv"

# Carregar modelo e scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Função de previsão
def predict_next_days(days=1):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    SEQ_LEN = 60
    last_seq = scaled_data[-SEQ_LEN:]

    predictions = []
    current_seq = np.copy(last_seq)

    for _ in range(days):
        pred = model.predict(np.expand_dims(current_seq, axis=0))
        predictions.append(pred[0, 0])
        current_seq = np.vstack((current_seq[1:], pred))

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices, df


# Interface
st.title("📈 Previsão de Preço do Bitcoin")
st.write("Modelo GRU treinado para prever o preço futuro do BTC/USD")

days = st.slider("Escolha o horizonte de previsão (dias):", 1, 30, 7)
if st.button("Gerar Previsão"):
    preds, df = predict_next_days(days)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds.flatten()})

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-100:], df["Close"].values[-100:], label="Histórico")
    plt.plot(pred_df["Date"], pred_df["Predicted"], label="Previsão", color="orange")
    plt.legend()
    st.pyplot(plt)

    st.write("### Previsões:")
    st.dataframe(pred_df)
