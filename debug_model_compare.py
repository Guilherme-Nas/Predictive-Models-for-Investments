import os
import traceback
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIG
# ============================================================

# Caminho base automático (pasta onde o script está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos relativos
CSV_PATH = os.path.join(BASE_DIR, 'notebooks', 'data', 'btc_limpo.csv')
MODEL_PATH_1 = os.path.join(BASE_DIR, 'models', 'model_gru.keras')
MODEL_PATH_2 = os.path.join(BASE_DIR, 'models', 'model_gru_alt.keras')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'debug_compare')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNÇÕES
# ============================================================

def safe_load_model(path):
    try:
        print(f"[INFO] Carregando modelo: {path}")
        return load_model(path)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo {path}: {e}")
        return None

def load_csv():
    print(f"[INFO] Lendo CSV {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]

def fit_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['Close']].values.reshape(-1, 1))
    return scaler

def recursive_predict(model, seq, dias):
    seq = np.copy(seq)
    preds = []
    for _ in range(dias):
        p = model.predict(seq[np.newaxis, :, :], verbose=0)
        val = float(p.flatten()[0])
        preds.append(val)
        seq = np.concatenate([seq[1:], [[val]]], axis=0)
    return np.array(preds)

def inverse_transform(arr, scaler):
    return scaler.inverse_transform(np.array(arr).reshape(-1, 1)).flatten()

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == '__main__':
    try:
        print("[RUN] Iniciando comparação...")

        df = load_csv()
        scaler = fit_scaler(df)
        model1 = safe_load_model(MODEL_PATH_1)
        model2 = safe_load_model(MODEL_PATH_2)

        if model1 is None:
            raise RuntimeError("Modelo principal não pôde ser carregado!")

        lookback = 200
        dias = 7
        idx = len(df) - dias - 1

        scaled = scaler.transform(df[['Close']].values)
        seq = scaled[idx - lookback: idx]

        preds1_scaled = recursive_predict(model1, seq, dias)
        preds1 = inverse_transform(preds1_scaled, scaler)

        if model2:
            preds2_scaled = recursive_predict(model2, seq, dias)
            preds2 = inverse_transform(preds2_scaled, scaler)
        else:
            preds2 = None

        reais = df['Close'].iloc[idx: idx + dias].values
        datas = df['Date'].iloc[idx: idx + dias].values

        mae1 = mean_absolute_error(reais, preds1)
        rmse1 = np.sqrt(mean_squared_error(reais, preds1))
        print(f"[RESULT] Modelo 1 — MAE={mae1:.2f}, RMSE={rmse1:.2f}")

        if preds2 is not None:
            mae2 = mean_absolute_error(reais, preds2)
            rmse2 = np.sqrt(mean_squared_error(reais, preds2))
            print(f"[RESULT] Modelo 2 — MAE={mae2:.2f}, RMSE={rmse2:.2f}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(datas, reais, label='Real', color='green', linewidth=2)
        plt.plot(datas, preds1, label='Modelo 1', color='orange', linewidth=2)
        if preds2 is not None:
            plt.plot(datas, preds2, label='Modelo 2', color='blue', linewidth=2, linestyle='--')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title('Comparação de modelos — previsões vs reais')
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'compare_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"[OK] Gráfico salvo em {plot_path}")

    except Exception as e:
        print("❌ ERRO CRÍTICO DETECTADO:")
        print(traceback.format_exc())
        input("Pressione Enter para sair...")
