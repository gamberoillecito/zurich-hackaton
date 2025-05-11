import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_simulation(prices, n_simulations=10000, n_days=252):
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    S0 = prices.iloc[-1]

    # Simulate future price paths
    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        # Generate random walk
        daily_returns = np.random.normal(mu, sigma, n_days)
        price_path = S0 * np.exp(np.cumsum(daily_returns))
        simulations[:, i] = price_path

    # Plot simulation results
    #plt.figure(figsize=(10, 6))
    #plt.plot(simulations, color='grey', alpha=0.1)
    #plt.title("Monte Carlo Simulated Asset Paths")
    #plt.xlabel("Days")
    #plt.ylabel("Price")
    #plt.show()

    # Expected price distribution at horizon
    final_prices = simulations[-1, :]
    expected_price = np.mean(final_prices)

    plt.figure(figsize=(8, 5))
    plt.hist(final_prices, bins=50, alpha=0.7)
    plt.axvline(expected_price, color='red', linestyle='--', label=f'Expected Price = {expected_price:.2f}')
    plt.title("Expected Final Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return expected_price, final_prices

# Example usage
# Load your own data here (must be a Series of prices)
# Example: prices = pd.read_csv('your_data.csv')['Close']

# Synthetic example:
dates = pd.date_range(start="2023-01-01", periods=252)
synthetic_prices = pd.Series(np.cumprod(1 + np.random.normal(0.0002, 0.01, 252)) * 100, index=dates)

expected_price, simulated_prices = monte_carlo_simulation(synthetic_prices)
print(f"Expected price after simulation: {expected_price:.2f}")
