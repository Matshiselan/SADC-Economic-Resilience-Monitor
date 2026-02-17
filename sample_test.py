"""
Sample test script for SADC Economic Resilience Monitor pipeline.
This script demonstrates a minimal end-to-end test using mock data.
"""

import pandas as pd
import numpy as np
from tabulate import tabulate

from src.tools import (
    fetch_sadc_data, profile_missingness, plot_missing_data_heatmap, detect_structural_breaks,
    compute_fiscal_risk_score, rule_based_risk_flag, monte_carlo_macro_shock,
    recommend_policy_actions, generate_ministerial_brief, plot_risk_heatmap
)

# 1. Fetch real macro data for SADC countries from World Bank API
sadc_countries = [
        ('AGO', 'Angola'),
        ('BWA', 'Botswana'),
        ('COM', 'Comoros'),
        ('COD', 'Democratic Republic of the Congo'),
        ('SWZ', 'Eswatini'),
        ('LSO', 'Lesotho'),
        ('MDG', 'Madagascar'),
        ('MWI', 'Malawi'),
        ('MUS', 'Mauritius'),
        ('MOZ', 'Mozambique'),
        ('NAM', 'Namibia'),
        ('SEY', 'Seychelles'),
        ('ZAF', 'South Africa'),
        ('TZA', 'Tanzania'),
        ('ZMB', 'Zambia'),
        ('ZWE', 'Zimbabwe'),
    ]
sadc_iso_codes = [iso for iso, name in sadc_countries]
iso_to_name = dict(sadc_countries)

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
missingness = profile_missingness(wide_df)
print(missingness)

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
if isinstance(sims, list):
    # Convert list of DataFrames to a single DataFrame
    sims = pd.concat(sims, ignore_index=True)
print(f'Simulated scenario (first run):\n{sims[["inflation"]].head()}')



# 6. Report Generation (executive summary, pretty table, and notes for each country)
import os
report_path = os.path.join('reports', 'sadc_ministerial_brief.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    for country in sadc_iso_codes:
        country_rows = wide_df[wide_df['country'] == country]
        if country_rows.empty:
            continue
        country_name = iso_to_name.get(country, country)
        frs_mean = country_rows['gdp_growth'].mean() if 'gdp_growth' in country_rows else 0


        # Build table data for appendix and executive summary
        table_data = []
        sim_rows = sims[sims['country'] == country] if 'country' in sims.columns else sims
        sim_by_year = sim_rows.groupby('year').first() if 'year' in sim_rows.columns else None
        for _, row in country_rows.iterrows():
            year = int(row['year'])
            flag = rule_based_risk_flag(row)
            key_drivers = f"Debt: {row.get('debt_gdp', 'NA'):.1f}, Inf: {row.get('inflation', 'NA'):.1f}, Growth: {row.get('gdp_growth', 'NA'):.1f}"
            scenario = "N/A"
            if sim_by_year is not None and year in sim_by_year.index:
                sim_infl = sim_by_year.loc[year]['inflation'] if 'inflation' in sim_by_year.columns else None
                if sim_infl is not None:
                    scenario = f"Inflation after shock: {sim_infl:.2f}%"
            policy = recommend_policy_actions(row)
            table_data.append([year, flag, key_drivers, scenario, policy])
        headers = ["Year", "Flag", "Key Drivers", "Scenario Analysis", "Policy Recommendation"]
        table_str = tabulate(table_data, headers=headers, tablefmt="github", showindex=False)

        # Save all detailed report sections as appendix
        appendix = []
        appendix.append(f"Fiscal Risk Score (mean gdp_growth): {frs_mean:.2f}\n")
        appendix.append(table_str + "\n")
        appendix.append("Column Key:\n")
        appendix.append("Year: Calendar year of observation\n")
        appendix.append("Flag: Early warning risk flag (HIGH/MEDIUM/LOW)\n")
        appendix.append("Key Drivers: Main macro-fiscal indicators\n")
        appendix.append("Scenario Analysis: Description of simulated macro shock\n")
        appendix.append("Policy Recommendation: Automated policy advice based on risk pattern\n")


        # Data Quality Report: save to separate file
        dq_report = []
        dq_report.append(f"Data Quality Auditor Report: {country_name}\n")
        missing = profile_missingness(country_rows)
        dq_report.append("Missing Data (%):\n")
        dq_report.append(missing.to_string() + "\n\n")
        dq_report.append("Structural Breaks (by column):\n")
        for col in ['gdp_growth', 'inflation', 'debt_gdp', 'fiscal_balance', 'current_account', 'reserves_months']:
            if col in country_rows.columns:
                try:
                    breaks = detect_structural_breaks(country_rows, col)
                    dq_report.append(f"{col}: breakpoints at years {[int(country_rows.iloc[b]['year']) for b in breaks['breakpoints']]}\n")
                except Exception as e:
                    dq_report.append(f"{col}: error ({e})\n")
        dq_report.append("\n")
        dq_report.append("Chow Test F-statistics (by column):\n")
        for col in ['gdp_growth', 'inflation', 'debt_gdp', 'fiscal_balance', 'current_account', 'reserves_months']:
            if col in country_rows.columns and country_rows[col].notnull().sum() > 10:
                try:
                    split = len(country_rows) // 2
                    f_stat = chow_test(country_rows, col, split)
                    dq_report.append(f"{col}: F = {f_stat:.2f}\n")
                except Exception as e:
                    dq_report.append(f"{col}: error ({e})\n")
        dq_report.append("\n")
        dq_report.append("Outlier Count (z-score > 3):\n")
        for col in ['gdp_growth', 'inflation', 'debt_gdp', 'fiscal_balance', 'current_account', 'reserves_months']:
            if col in country_rows.columns:
                try:
                    outliers = detect_outliers(country_rows, col, method="zscore", threshold=3.0)
                    count = outliers.sum()
                    dq_report.append(f"{col}: {count}\n")
                except Exception as e:
                    dq_report.append(f"{col}: error ({e})\n")
        dq_report.append("\n")
        dq_report.append("Rolling Variance (window=5, last value):\n")
        for col in ['gdp_growth', 'inflation', 'debt_gdp', 'fiscal_balance', 'current_account', 'reserves_months']:
            if col in country_rows.columns:
                try:
                    rv = rolling_variance(country_rows, col, window=5)
                    last = rv.dropna().iloc[-1] if not rv.dropna().empty else float('nan')
                    dq_report.append(f"{col}: {last:.2f}\n")
                except Exception as e:
                    dq_report.append(f"{col}: error ({e})\n")
        dq_report.append("\n")
        dq_report.append("Metadata Quality Score:\n")
        dummy_metadata = {"source": "World Bank", "units": "various", "frequency": "annual", "last_updated": "2025"}
        try:
            score = score_metadata_quality(country_rows, dummy_metadata)
            dq_report.append(f"Score: {score:.2f}\n")
        except Exception as e:
            dq_report.append(f"error ({e})\n")
        dq_report.append("\n")

        dq_report_path = os.path.join('reports', 'data_quality_audit.txt')
        with open(dq_report_path, 'a', encoding='utf-8') as dqf:
            dqf.write(f"\n{'='*60}\nData Quality Summary: {country_name}\n{'='*60}\n{''.join(dq_report)}")

        # Write appendix to file
        f.write(f"\n{'='*60}\nAppendix: {country_name}\n{'='*60}\n")
        f.write(''.join(appendix))

        # Placeholder for Ollama agent executive summary
        f.write(f"\n{'='*60}\nOllama Executive Summary: {country_name}\n{'='*60}\n")
        f.write("[This section will be generated by the Ollama agent based on the appendix above.]\n\n")

print(f'Report generated for all countries: {report_path}')
