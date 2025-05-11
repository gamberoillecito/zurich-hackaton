import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import requests
import io


ndays=30
# ---------- 1. Load Data from GitHub ----------
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    data = pd.read_json(io.StringIO(response.text))
    return data

# Replace with your actual GitHub raw CSV URL
github_csv_url = 'https://raw.githubusercontent.com/gamberoillecito/zurich-hackaton/refs/heads/main/giacomo/selected_assets.json?token=GHSAT0AAAAAADDUFJGKUPHQ5E3Q2S6P2U2E2A7YGTQ'
df_prices = load_data_from_github(github_csv_url)

# ---------- 2. Calculate Daily Returns ----------
returns = df_prices.pct_change().dropna()

# ---------- 3. Estimate Expected Return & Volatility via Monte Carlo ----------
def estimate_expected_returns_via_monte_carlo(returns, n_days, n_simulations=10000):
    mc_mean_returns = []
    mc_std_devs = []

    for asset in returns.columns:
        daily_returns = returns[asset]
        mu = daily_returns.mean()
        sigma = daily_returns.std()

        simulated_paths = np.random.normal(mu, sigma, (n_days, n_simulations))
        cumulative_returns = np.exp(np.cumsum(simulated_paths, axis=0))
        final_returns = cumulative_returns[-1, :] - 1

        annual_return = np.mean(final_returns)
        annual_volatility = np.std(final_returns)

        mc_mean_returns.append(annual_return)
        mc_std_devs.append(annual_volatility)

    mean_returns = pd.Series(mc_mean_returns, index=returns.columns)
    std_devs = pd.Series(mc_std_devs, index=returns.columns)

    return mean_returns, std_devs

mean_returns, asset_std_devs = estimate_expected_returns_via_monte_carlo(returns)
cov_matrix = returns.cov() * ndays

# ---------- 4. Portfolio Optimization ----------
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

risk_free_rate = 0.04
optimal_result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
optimal_weights = optimal_result.x
expected_return, std_dev = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)

# ---------- 5. Efficient Frontier Visualization ----------
def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.04):
    num_assets = len(mean_returns)
    results = {'returns': [], 'risk': [], 'sharpe': [], 'weights': []}

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        port_return, port_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (port_return - risk_free_rate) / port_std
        results['returns'].append(port_return)
        results['risk'].append(port_std)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)

    return results

def plot_efficient_frontier(results, optimal_weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.array(results['returns'])
    risk = np.array(results['risk'])
    sharpe = np.array(results['sharpe'])

    plt.figure(figsize=(10, 6))
    plt.scatter(risk, returns, c=sharpe, cmap='viridis', alpha=0.4)
    plt.colorbar(label='Sharpe Ratio')

    opt_ret, opt_std = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    plt.scatter(opt_std, opt_ret, c='red', s=200, marker='*', label='Optimal Portfolio')

    x = np.linspace(0, max(risk), 100)
    cml = risk_free_rate + (opt_ret - risk_free_rate) / opt_std * x
    plt.plot(x, cml, 'r--', label='Capital Market Line')

    plt.title("Efficient Frontier with Optimal Portfolio and CML")
    plt.xlabel("Volatility (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()

frontier_results = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)
plot_efficient_frontier(frontier_results, optimal_weights, mean_returns, cov_matrix, risk_free_rate)

# ---------- 6. Monte Carlo Simulation of Portfolio ----------
def monte_carlo_simulation(expected_return, std_dev, start_value=1000, n_days=252, n_simulations=1000):
    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        daily_returns = np.random.normal(expected_return / n_days, std_dev / np.sqrt(n_days), n_days)
        price_path = start_value * np.exp(np.cumsum(daily_returns))
        simulations[:, i] = price_path

    plt.figure(figsize=(10, 6))
    plt.plot(simulations, alpha=0.05, color='blue')
    plt.title("Monte Carlo Simulated Portfolio Value Paths")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.show()

    final_prices = simulations[-1, :]
    expected_final = np.mean(final_prices)

    plt.figure(figsize=(8, 5))
    plt.hist(final_prices, bins=50, alpha=0.7)
    plt.axvline(expected_final, color='red', linestyle='--', label=f'Expected = {expected_final:.2f}')
    plt.title("Expected Final Portfolio Value Distribution")
    plt.xlabel("Portfolio Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return expected_final, final_prices

final_value, simulation_results = monte_carlo_simulation(expected_return, std_dev)
print(f"\nExpected Final Portfolio Value (Monte Carlo): {final_value:.2f}")

# ---------- 7. Backtesting ----------
def backtest_portfolio(returns, weights):
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Optimized Portfolio')
    plt.title("Backtested Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = np.mean(portfolio_returns) * 252
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    print(f"\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

backtest_portfolio(returns, optimal_weights)
