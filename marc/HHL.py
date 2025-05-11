# ============================
# 📦 INSTALLA PACCHETTI (se serve)
# ============================
# pip install pandas numpy matplotlib scikit-learn tensorflow cvxpy qiskit

# ============================
# 📚 IMPORT
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from qiskit_aer import Aer
from qiskit.algorithms.linear_solvers.hhl import HHL

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ============================
# ⚙️ FUNZIONE UNICA
# ============================
def optimize_portfolio(returns_df, lookback_days=5, epochs=50, risk_aversion=10.0, plot_results=True):
    """
    Input: returns_df → DataFrame con rendimenti storici degli asset (n_days x n_assets)
    Output: Dizionario con pesi ottimali per asset
    """
    n_assets = returns_df.shape[1]
    
    # ============================
    # ⚙️ SCALA I DATI (con sklearn)
    # ============================
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_returns = scaler.fit_transform(returns_df)

    # ============================
    # ⚙️ COSTRUISCI SEQUENZE X/y
    # ============================
    X = []
    y = []
    for i in range(lookback_days, len(scaled_returns) - 1):
        X.append(scaled_returns[i - lookback_days:i])
        y.append(np.mean(scaled_returns[i + 1]))
    X = np.array(X)
    y = np.array(y)

    # ============================
    # ✂️ SPLIT TRAIN/TEST
    # ============================
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ============================
    # ⚙️ COSTRUISCI MODELLO LSTM
    # ============================
    model = Sequential([
        LSTM(units=50, activation='tanh', input_shape=(lookback_days, n_assets)),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # ============================
    # 🏋️‍♂️ ALLENA MODELLO
    # ============================
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, validation_data=(X_test, y_test))

    # ============================
    # 🔮 PREVISIONE LSTM (prossimo giorno)
    # ============================
    last_sequence = scaled_returns[-lookback_days:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    predicted_scaled = model.predict(last_sequence)[0][0]

    # ============================
    # 📊 STATISTICHE STORICHE
    # ============================
    # Rimuove asset con varianza zero (serie costante)
    returns_df = returns_df.loc[:, returns_df.std() > 0]

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    n_assets = returns_df.shape[1]   # Aggiorna n_assets

    # Configura il backend
    backend = Aer.get_backend('aer_simulator')

    # Costruisci A (cov_matrix) e b (predicted_returns)
    A = cov_matrix
    b = predicted_returns

    # Rendi A Hermitiana e invertibile
    reg_lambda = 1e-3
    A_reg = A + reg_lambda * np.identity(A.shape[0])

    # Normalizza b per HHL
    b_norm = b / np.linalg.norm(b)

    # Esegui HHL direttamente con il backend (NUOVO MODO)
    hhl = HHL()
    result = hhl.solve(matrix=A_reg, vector=b_norm, backend=backend)

    # Estrai la soluzione (array numpy)
    optimal_weights = np.real(result.solution)

    # Normalizza i pesi
    optimal_weights = np.clip(optimal_weights, 0.01, 0.40)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    # Controlla validità
    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        print("⚠️ HHL did not find a valid solution. Skipping weights and plot.")
        optimal_weights = np.zeros(n_assets)
    else:
        print("✅ HHL quantum solver returned valid weights")

    optimal_weights = w.value

    if prob.status != "optimal" or optimal_weights is None:
        print("⚠️ CVXPY did not find a valid solution. Skipping weights and plot.")
        optimal_weights = np.zeros(n_assets)   # Mettiamo pesi a zero per sicurezza

    # ============================
    # 📈 PLOT (facoltativo)
    # ============================
    if plot_results:
        plt.figure(figsize=(14,4))

        # Verifica se 'loss' e 'val_loss' sono disponibili nel dizionario
        if 'loss' in history.history:
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('LSTM Training Loss')
            plt.legend()

        plt.subplot(1, 2, 2)
        print("Optimal weights:", optimal_weights)
        #aggiunta per provare a sistemare il problema della matrice
        plt.bar(returns_df.columns, optimal_weights)
        plt.title('Optimal Portfolio Weights')
        plt.ylabel('Weight')

        plt.tight_layout()
        plt.show()

    # ============================
    # 📋 OUTPUT
    # ============================
    output = {}
    for i, weight in enumerate(optimal_weights):
        output[returns_df.columns[i]] = round(weight, 4)

    print(f"\nPredicted average return (next day): {predicted_scaled:.5f}")
    print("Optimal Portfolio Weights:")
    for k, v in output.items():
        print(f"{k}: {v}")

    return output

# ============================
# ✨ ESEMPIO DI USO (REALISTIC DATA — 20 ASSETS)
# ============================
#np.random.seed(42)
#n_days = 150

#assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
#          'NVDA', 'JPM', 'BAC', 'WMT', 'PG',
#          'JNJ', 'PFE', 'UNH', 'XOM', 'CVX',
#          'T', 'VZ', 'NKE', 'KO', 'MCD']

#n_assets = len(assets)

# Simuliamo diversi rendimenti medi annualizzati per settori (in ordine approssimativo)
# Tech: più alto, Financial/Consumer/Healthcare: medio, Energy/Telecom: più basso
# mean_annual_returns = np.array([
 #   0.15, 0.14, 0.14, 0.16, 0.18,  # Tech
 #   0.20,                           # NVDA (semiconduttori ⇒ più volatile)
 #   0.10, 0.09,                    # Financials
 #   0.08, 0.08,                    # Consumer Staples
 #   0.10, 0.09, 0.11,              # Healthcare
 #   0.08, 0.07,                    # Energy
 #   0.06, 0.06,                    # Telecom
 #   0.12, 0.08, 0.11              # Consumer Discretionary / Staples
#])

# Volatilità annualizzate per settori (approssimative)
#std_annual = np.array([
#    0.22, 0.20, 0.21, 0.25, 0.28,
#    0.35,
#    0.18, 0.19,
#    0.15, 0.14,
#    0.16, 0.17, 0.15,
#    0.20, 0.19,
#    0.13, 0.13,
#    0.21, 0.14, 0.18
#])

# Convertiamo a valori giornalieri
#mean_daily_returns = mean_annual_returns / 252
#std_daily = std_annual / np.sqrt(252)

# Simuliamo rendimenti giornalieri realistici
#returns = np.random.normal(loc=mean_daily_returns, scale=std_daily, size=(n_days, n_assets))


def dictWeightedAssets(data):
    #with open(file) as openFile:
    #    data = json.load(openFile)

    # === Crea DataFrame prezzi ===
    prices_df = pd.DataFrame({
    k: pd.Series(v['history']).sort_index()
    for k, v in data.items()
    }).reset_index(drop=True)

   #=== Calcola rendimenti log giornalieri ===
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    print(returns_df)
    optimal_weights = optimize_portfolio(returns_df)
    # ⚡️ CHIAMA LA FUNZIONE con rendimenti (non prezzi!)
    return optimal_weights

with open("marc\selected_assets test1.json") as openFile:
   data = json.load(openFile)

dictWeightedAssets(data)
