import json
import random
from collections import defaultdict

# Leggi il dataset dal file JSON
with open("marc\selected_assets.json") as file:
    data = json.load(file)

# Raggruppiamo le aziende per industria, settore e regione geografica
grouped_by_industry_sector_region = defaultdict(list)

for ticker, details in data.items():
    industry = details['industry']
    sector = details['sector']
    region = details['region']
    grouped_by_industry_sector_region[(industry, sector, region)].append(ticker)

# Determina il numero di aziende da selezionare
num_companies_to_select = 20
total_groups = len(grouped_by_industry_sector_region)
companies_per_group = num_companies_to_select // total_groups

# Seleziona casualmente le aziende da ciascun gruppo
selected_companies = {}
for group, tickers in grouped_by_industry_sector_region.items():
    # Seleziona un numero casuale di aziende dal gruppo, rispettando la quantità desiderata
    selected_tickers = random.sample(tickers, min(companies_per_group, len(tickers)))
    selected_companies.update({ticker: data[ticker] for ticker in selected_tickers})

# Se il numero totale di aziende selezionate non raggiunge i 100, aggiungi aziende casuali
while len(selected_companies) < num_companies_to_select:
    remaining = num_companies_to_select - len(selected_companies)
    # Aggiungi aziende casuali dalla lista rimanente
    remaining_tickers = [ticker for ticker in data if ticker not in selected_companies]
    additional_tickers = random.sample(remaining_tickers, remaining)
    selected_companies.update({ticker: data[ticker] for ticker in additional_tickers})

# Salva i risultati in un nuovo file JSON
with open("marc/selected_companies_diverse.json", "w") as file:
    json.dump(selected_companies, file, indent=4)

# Verifica il numero di aziende selezionate
print(f"Numero di aziende selezionate: {len(selected_companies)}")
