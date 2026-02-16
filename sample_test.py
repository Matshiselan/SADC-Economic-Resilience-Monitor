"""
Sample test script for SADC Economic Resilience Monitor pipeline.
This script demonstrates a minimal end-to-end test using mock data.
"""
import pandas as pd
import numpy as np

from src.tools import (
    fetch_sadc_data, profile_missingness, plot_missing_data_heatmap, detect_structural_breaks,
    compute_fiscal_risk_score, rule_based_risk_flag, monte_carlo_macro_shock,
    recommend_policy_actions, generate_ministerial_brief, plot_risk_heatmap
)

# 1. Fetch real macro data for SADC countries from World Bank API
sadc_iso_codes = [
    'ZWE',  # Zimbabwe
    'ZMB',  # Zambia
    'MWI',  # Malawi
    'MOZ',  # Mozambique
    'NAM',  # Namibia
    'BWA',  # Botswana
    'ZAF',  # South Africa
    'AGO',  # Angola
    'TZA',  # Tanzania
]

print('Fetching World Bank macro-fiscal data (1970-2024)...')
wb_data = fetch_sadc_data(sadc_iso_codes, start=1970, end=2025)

print('Fetched data shape:', wb_data.shape)
print(wb_data.head())

# Convert World Bank API output to flat DataFrame
flat_rows = []
for (country, indicator), row in wb_data.iterrows():
    for year in wb_data.columns:
        flat_rows.append({
            'country': country,
            'indicator': indicator,
            'year': int(year.replace('YR', '')),
            'value': row[year]
        })
flat_df = pd.DataFrame(flat_rows)

# Pivot to wide format: one row per country-year, columns for each indicator
wide_df = flat_df.pivot_table(index=['country', 'year'], columns='indicator', values='value').reset_index()
wide_df = wide_df.rename(columns={
    'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
    'FP.CPI.TOTL.ZG': 'inflation',
    'GC.DOD.TOTL.GD.ZS': 'debt_gdp',
    'GC.BAL.CASH.GD.ZS': 'fiscal_balance',
    'BN.CAB.XOKA.GD.ZS': 'current_account',
    'FI.RES.TOTL.MO': 'reserves_months'
})
print('Wide DataFrame head:')
print(wide_df.head())


# 2. Data Quality Checks
print('Missingness profile:')
print(profile_missingness(wide_df))
plot_missing_data_heatmap(wide_df)

# Compute 'growth_shock' and 'reserves_risk' columns for all rows
wide_df['growth_shock'] = wide_df['gdp_growth'] - wide_df['gdp_growth'].mean()
wide_df['reserves_risk'] = wide_df['reserves_months']

# 3. Risk Score & Flag (example for one country/year)
sample_row = wide_df.iloc[0].copy()
frs = compute_fiscal_risk_score(sample_row)
flag = rule_based_risk_flag(sample_row)
print(f"Sample FRS: {frs}, Risk Flag: {flag}")

# 4. Policy Recommendation
policy = recommend_policy_actions(sample_row)
print(f"Policy recommendation: {policy}")

# 5. Scenario Simulation (example: inflation shock)
shocks = {'inflation': 5}
sims = monte_carlo_macro_shock(wide_df, shocks, n_sim=10)
print(f'Simulated scenario (first run):\n{sims[0][["inflation"]].head()}')

# 6. Report Generation (text and heatmap for Zambia)
zambia_rows = wide_df[wide_df['country'] == 'ZWE']
if not zambia_rows.empty:
    zambia_frs = zambia_rows['gdp_growth'].mean() if 'gdp_growth' in zambia_rows else 0
    zambia_flags = [rule_based_risk_flag(row) for _, row in zambia_rows.iterrows()]
    zambia_policies = [recommend_policy_actions(row) for _, row in zambia_rows.iterrows()]
    generate_ministerial_brief(
        country='ZWE',
        risk_score=zambia_frs,
        flags=zambia_flags,
        scenarios='Inflation shock +5pp',
        recommendations=zambia_policies,
        filename='sample_ministerial_brief.txt'
    )

    # Risk heatmap
    if 'gdp_growth' in wide_df:
        risk_matrix = wide_df.pivot(index='year', columns='country', values='gdp_growth').values
        plot_risk_heatmap(risk_matrix, sadc_iso_codes, filename='sample_risk_heatmap.png')

print('Sample ministerial brief and risk heatmap generated.')
