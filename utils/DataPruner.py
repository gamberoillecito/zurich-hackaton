import json
import random
from collections import defaultdict

class DataPruner:
    def __init__(self, data):
        self.data = data
        print(self.data.keys())

    def select_assets(self, num_companies_to_select):
        '''Given an input dataset, uses different heuristics
        to select `num_companies_to_select` assets'''
        # Group companies by industry, sector, and geographical region
        grouped_by_industry_sector_region = defaultdict(list)

        self.evaluation_date = self.data['evaluation_date']

        self.data['data'] = self.data['assets'] 
        for ticker, details in self.data.items():
            industry = details['industry']
            sector = details['sector']
            region = details['region']
            grouped_by_industry_sector_region[(industry, sector, region)].append(ticker)

        # Determine the number of companies to select
        total_groups = len(grouped_by_industry_sector_region)
        companies_per_group = num_companies_to_select // total_groups

        # Randomly select companies from each group
        selected_companies = {}
        for group, tickers in grouped_by_industry_sector_region.items():
            # Select a random number of companies from the group, respecting the desired amount
            selected_tickers = random.sample(tickers, min(companies_per_group, len(tickers)))
            selected_companies.update({ticker: self.data[ticker] for ticker in selected_tickers})

        # If the total number of selected companies is less than 100, add random companies
        while len(selected_companies) < num_companies_to_select:
            remaining = num_companies_to_select - len(selected_companies)
            # Add random companies from the remaining list
            remaining_tickers = [ticker for ticker in self.data if ticker not in selected_companies]
            additional_tickers = random.sample(remaining_tickers, remaining)
            selected_companies.update({ticker: self.data[ticker] for ticker in additional_tickers})
        return {'evaluation_date': self.evaluation_date, 'assets': selected_companies}
