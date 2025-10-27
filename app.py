# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import prever_btc, carregar_dados

st.set_page_config(page_title="Previsão Bitcoin GRU", layout="wide")

st.title("📈 Previsão de Preço do Bitcoin")
st.write("Previsões automáticas com modelo de rede neural **GRU**.")

# Entrada do usuário
dias = st.slider("Selecione o horizonte de previsão (em dias):", 1, 30, 7)

# Botão principal
if st.button("🔮 Gerar previsão"):
    with st.spinner("Gerando previsão..."):
        df_prev = prever_btc(dias=dias)
        df_real = carregar_dados()

        # Combinar histórico com previsão
        df_hist = df_real[["Date", "Close"]].rename(columns={"Close": "Preço Real"})
        df_comb = pd.concat(
            [df_hist, df_prev.rename(columns={"Preço Previsto": "Preço Real"})]
        )

        # Plotar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(
            df_hist["Date"], df_hist["Preço Real"], label="Histórico", color="blue"
        )
        plt.plot(
            df_prev["Data"],
            df_prev["Preço Previsto"],
            label="Previsão",
            color="red",
            linestyle="--",
        )
        plt.title(f"Previsão de {dias} dias à frente (Modelo GRU)")
        plt.xlabel("Data")
        plt.ylabel("Preço (USD)")
        plt.legend()
        st.pyplot(plt)

        st.subheader("📅 Tabela de Previsões")
        st.dataframe(df_prev.set_index("Data"))

st.info("💡 Dica: use o controle deslizante para alterar o número de dias previstos.")
