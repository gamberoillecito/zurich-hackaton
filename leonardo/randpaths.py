import pandas as pd
import numpy as np

# Load your data
url = "https://raw.githubusercontent.com/gamberoillecito/zurich-hackaton/refs/heads/main/giacomo/selected_assets.json?token=GHSAT0AAAAAADDUFJGKE25LO26KPNAM6A5I2A7YWHA"
df = pd.read_csv(url, index_col=0, parse_dates=True)

# Define the portfolio generator
def generate_random_portfolio(asset_data: pd.DataFrame, day: str) -> pd.Series:
    if day not in asset_data.index:
        raise ValueError(f"Day {day} not found in asset data.")

    values = asset_data.loc[day]
    weights = np.random.rand(len(values))
    weights /= weights.sum()
    portfolio_value = np.dot(weights, values)
    portfolio = pd.Series(weights, index=values.index)
    portfolio['Total Value'] = portfolio_value
    return portfolio

# Example usage
portfolio = generate_random_portfolio(df, '2025-03-01')
print(portfolio)
