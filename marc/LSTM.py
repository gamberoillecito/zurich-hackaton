# ============================
# 📦 INSTALLA PACCHETTI (se serve)
# ============================
# pip install pandas numpy matplotlib scikit-learn tensorflow cvxpy

# ============================
# 📚 IMPORT
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import cvxpy as cp

# ============================
# ⚙️ FUNZIONE UNICA
# ============================
def optimize_portfolio(returns_df, lookback_days=5, epochs=50, risk_aversion=0.1, plot_results=True):
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
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    # ============================
    # ⚙️ COSTRUISCI PREDICTED RETURNS VECTOR
    # ============================
    predicted_returns = mean_returns * (predicted_scaled / np.mean(mean_returns))
    predicted_returns = predicted_returns / np.linalg.norm(predicted_returns)

    # ============================
    # ⚙️ CVXPY PORTFOLIO OPTIMIZER
    # ============================
    w = cp.Variable(n_assets)

    portfolio_return = predicted_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)

    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)

    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    optimal_weights = w.value

    # ============================
    # 📈 PLOT (facoltativo)
    # ============================
    if plot_results:
        plt.figure(figsize=(14,4))

        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('LSTM Training Loss')
        plt.legend()

        plt.subplot(1,2,2)
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
# ✨ ESEMPIO DI USO (FAKE DATA)
# ============================
np.random.seed(42)
n_days = 150
n_assets = 3
returns_df = pd.DataFrame(np.random.normal(0.001, 0.02, size=(n_days, n_assets)),
                           columns=[f"Asset_{i+1}" for i in range(n_assets)])

# ⚡️ CHIAMA LA FUNZIONE
optimal_weights = optimize_portfolio(returns_df)
