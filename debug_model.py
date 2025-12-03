import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_PATH = 'notebooks/data/btc_limpo.csv'
MODEL_PATH = 'models/model_gru.keras'
OUTPUT_DIR = 'outputs/debug_reports'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_csv():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]

def fit_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['Close']].values.reshape(-1, 1))
    return scaler

def scale_series(df, scaler):
    return scaler.transform(df[['Close']].values.reshape(-1, 1))

def inverse_transform_array(arr, scaler):
    return scaler.inverse_transform(np.array(arr).reshape(-1, 1)).flatten()

def recursive_predict_from_scaled(seq_scaled, dias, model):
    """Prevê recursivamente a partir de uma sequência escalada"""
    if seq_scaled.ndim == 2:
        seq = np.expand_dims(seq_scaled, axis=0)
    else:
        seq = np.copy(seq_scaled)
    seq_copy = np.copy(seq)
    preds_scaled = []
    for _ in range(dias):
        p = model.predict(seq_copy, verbose=0)
        val = float(p.reshape(-1)[0])
        preds_scaled.append(val)
        seq_copy = np.concatenate(
            [seq_copy[:, 1:, :], np.array(val).reshape(1, 1, 1)], axis=1
        )
    return np.array(preds_scaled)

# ============================================================
# FUNÇÃO DE TESTE ÚNICO
# ============================================================

def run_single_test(idx, dias, df, scaler, model, lookback=200, save_plot=True):
    if idx - lookback < 0 or idx + dias > len(df):
        raise ValueError('idx fora do intervalo seguro')

    scaled = scale_series(df, scaler)
    seq_scaled = scaled[idx - lookback: idx]

    preds_scaled = recursive_predict_from_scaled(seq_scaled, dias, model)
    preds = inverse_transform_array(preds_scaled, scaler)
    reais = df['Close'].iloc[idx: idx + dias].values

    mape = np.mean(np.abs((preds - reais) / reais)) * 100
    mae = mean_absolute_error(reais, preds)
    rmse = np.sqrt(mean_squared_error(reais, preds))

    real_return = (reais[-1] - reais[0]) / reais[0]
    pred_return = (preds[-1] - preds[0]) / preds[0]
    dir_real = 1 if real_return > 0 else -1
    dir_pred = 1 if pred_return > 0 else -1
    dir_correct = int(dir_real == dir_pred)
    var_preds = float(np.var(preds))

    plot_path = None
    if save_plot:
        dates_hist = df['Date'].iloc[idx - lookback: idx].reset_index(drop=True)
        hist_prices = df['Close'].iloc[idx - lookback: idx].values
        dates_future = df['Date'].iloc[idx: idx + dias].reset_index(drop=True)

        plt.figure(figsize=(10, 5))
        plt.plot(dates_hist, hist_prices, label='Histórico', color='tab:blue', linewidth=1.5)
        plt.plot(dates_future, preds, label='Previsto', color='tab:orange', linewidth=2)
        plt.plot(dates_future, reais, label='Real futuro', color='tab:green', linestyle='--', linewidth=1.8)
        plt.axvline(dates_hist.iloc[-1], linestyle='--', color='gray', alpha=0.7)
        plt.title(f'Idx {idx} | MAPE {mape:.2f}% | Dir ok:{dir_correct} | VarPred {var_preds:.2f}')
        plt.legend()
        plt.grid(alpha=0.2)
        plot_path = os.path.join(OUTPUT_DIR, f'test_idx_{idx}_dias{dias}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    return {
        'idx': idx, 'mape': mape, 'mae': mae, 'rmse': rmse,
        'dir_real': dir_real, 'dir_pred': dir_pred, 'dir_correct': dir_correct,
        'var_preds': var_preds, 'plot': plot_path
    }

# ============================================================
# TESTES EM LOTE
# ============================================================

def run_batch_tests(n_tests=100, dias=7, seed=42, lookback=200):
    random.seed(seed)
    df = load_csv()
    scaler = fit_scaler(df)
    model = load_model(MODEL_PATH)

    results = []
    min_idx = lookback
    max_idx = len(df) - dias - 1

    for i in range(n_tests):
        idx = random.randint(min_idx, max_idx)
        res = run_single_test(idx, dias, df, scaler, model, lookback, save_plot=True)
        results.append(res)
        if (i + 1) % 10 == 0:
            print(f'[INFO] {i + 1}/{n_tests} testes concluídos')

    df_res = pd.DataFrame(results)
    summary = {
        'n_tests': n_tests,
        'dias': dias,
        'mape_mean': df_res['mape'].mean(),
        'mape_std': df_res['mape'].std(),
        'dir_accuracy': df_res['dir_correct'].mean(),
        'pred_var_mean': df_res['var_preds'].mean(),
        'pred_var_median': df_res['var_preds'].median()
    }

    df_res.to_csv(os.path.join(OUTPUT_DIR, f'batch_results_{n_tests}_d{dias}.csv'), index=False)
    return df_res, summary

# ============================================================
# PLOTS DE DIAGNÓSTICO
# ============================================================

def produce_diagnostic_plots(df_res, dias):
    plt.figure(figsize=(6,4))
    plt.hist(df_res['mape'].values, bins=30)
    plt.title(f'Histograma MAPE ({len(df_res)} testes, {dias} dias)')
    plt.xlabel('MAPE (%)')
    plt.ylabel('Frequência')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, f'hist_mape_d{dias}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(df_res['var_preds'].values, bins=30)
    plt.title('Histograma Var(Preds)')
    plt.xlabel('Var(preds)')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, f'hist_var_preds_d{dias}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(df_res['mape'], df_res['var_preds'], alpha=0.6)
    plt.xlabel('MAPE (%)')
    plt.ylabel('Var(preds)')
    plt.title('MAPE x Var(preds)')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, f'scatter_mape_var_d{dias}.png'), bbox_inches='tight')
    plt.close()

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == '__main__':
    n_tests = 100
    dias = 7
    print('[RUN] iniciando batch...')
    df_res, summary = run_batch_tests(n_tests=n_tests, dias=dias, seed=123)
    print('✅ batch concluído — resumo:', summary)
    produce_diagnostic_plots(df_res, dias=dias)
    top_errors = df_res.sort_values('mape', ascending=False).head(10)
    top_errors[['idx','mape','dir_correct','var_preds','plot']].to_csv(
        os.path.join(OUTPUT_DIR, 'top10_piores.csv'), index=False
    )
    print('Relatórios e plots salvos em', OUTPUT_DIR)
