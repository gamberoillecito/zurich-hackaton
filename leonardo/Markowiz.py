import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def optimize_portfolio(weights, mean_returns, cov_matrix, risk_free_rate):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(weights)))
    result = minimize(negative_sharpe_ratio, weights, args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def generate_efficient_frontier(mean_returns, cov_matrix, n_portfolios=100):
    results = {'returns': [], 'risk': [], 'weights': []}
    num_assets = len(mean_returns)
    for _ in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        port_return, port_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        results['returns'].append(port_return)
        results['risk'].append(port_std)
        results['weights'].append(weights)
    return results

def plot_efficient_frontier(results, optimal_portfolio, risk_free_rate):
    returns = results['returns']
    risk = results['risk']
    sharpe_ratios = [(r - risk_free_rate) / s for r, s in zip(returns, risk)]

    plt.figure(figsize=(10, 6))
    plt.scatter(risk, returns, c=sharpe_ratios, cmap='viridis', label='Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')

    opt_r, opt_s = calculate_portfolio_performance(optimal_portfolio['x'], mean_returns, cov_matrix)
    plt.scatter(opt_s, opt_r, c='red', marker='*', s=200, label='Tangency Portfolio')

    # Capital Market Line
    x = np.linspace(0, max(risk), 100)
    cml = risk_free_rate + ((opt_r - risk_free_rate) / opt_s) * x
    plt.plot(x, cml, 'r--', label='Capital Market Line')

    plt.title("Efficient Frontier with Capital Market Line")
    plt.xlabel("Volatility (Std. Deviation)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()

# === EXAMPLE USAGE ===
# Replace this with your own DataFrame of daily prices
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252)
assets = {
    'Asset A': np.cumprod(1 + np.random.normal(0.0004, 0.01, 252)) * 100,
    'Asset B': np.cumprod(1 + np.random.normal(0.0003, 0.012, 252)) * 100,
    'Asset C': np.cumprod(1 + np.random.normal(0.0002, 0.008, 252)) * 100,
}
df_prices = pd.DataFrame(assets, index=dates)
returns = df_prices.pct_change().dropna()

# Inputs
mean_returns = returns.mean() * 252  # Annualized mean returns
cov_matrix = returns.cov() * 252     # Annualized covariance
risk_free_rate = 0.04                # 4%

# Optimize for Tangency Portfolio
num_assets = len(mean_returns)
initial_weights = np.ones(num_assets) / num_assets
optimal_portfolio = optimize_portfolio(initial_weights, mean_returns, cov_matrix, risk_free_rate)

# Efficient Frontier
results = generate_efficient_frontier(mean_returns, cov_matrix, n_portfolios=1000)
plot_efficient_frontier(results, optimal_portfolio, risk_free_rate)

# Output
print("Optimal weights (Tangency Portfolio):")
for asset, weight in zip(returns.columns, optimal_portfolio['x']):
    print(f"{asset}: {weight:.2%}")


