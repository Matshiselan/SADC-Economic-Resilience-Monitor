# =========================================================
# SADC Economic Resilience Monitor - Agent Tools
# Arranged by agent workflow for clarity and maintainability
# =========================================================

# -----------------------------
# 1. Data Harvester Agent
# -----------------------------
from typing import Any, List, Dict, Tuple
import wbgapi as wb
import pandas as pd

# Macro-fiscal indicator codes
INDICATORS = {
	"NY.GDP.MKTP.KD.ZG": "gdp_growth",
	"FP.CPI.TOTL.ZG": "inflation",
	"GC.DOD.TOTL.GD.ZS": "debt_gdp",
	"GC.BAL.CASH.GD.ZS": "fiscal_balance",
	"BN.CAB.XOKA.GD.ZS": "current_account",
	"FI.RES.TOTL.MO": "reserves_months"
}

def fetch_sadc_data(countries: List[str], start: int = 2005, end: int = 2025) -> pd.DataFrame:
	"""
	Fetches macro-fiscal indicators for SADC countries from the World Bank API.
	"""
	df = wb.data.DataFrame(
		INDICATORS.keys(),
		economy=countries,
		time=range(start, end)
	)
	return df.rename(columns=INDICATORS)

def harvest_data(countries: List[Any]) -> Any:
	"""
	Collect macro-fiscal and economic data for the given SADC countries.
	Returns raw data (format TBD).
	"""
	pass

# -----------------------------
# 2. Data Quality Auditor Agent
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def profile_missingness(df: pd.DataFrame) -> pd.Series:
	"""Returns the percentage of missing values per column."""
	return df.isnull().mean() * 100

def plot_missing_data_heatmap(df: pd.DataFrame, figsize=(10, 6)) -> None:
	"""Plots a heatmap of missing data for visual inspection."""
	plt.figure(figsize=figsize)
	plt.imshow(df.isnull(), aspect='auto', cmap='viridis', interpolation='none')
	plt.xlabel('Variables')
	plt.ylabel('Observations')
	plt.title('Missing Data Heatmap')
	plt.colorbar(label='Missing')
	plt.show()

def detect_structural_breaks(df: pd.DataFrame, column: str) -> Dict[str, Any]:
	"""Detects structural breaks in a time series column using rolling mean/variance."""
	window = min(5, len(df)//2)
	rolling_mean = df[column].rolling(window=window).mean()
	rolling_std = df[column].rolling(window=window).std()
	diffs = rolling_mean.diff().abs()
	breakpoints = diffs[diffs > diffs.mean() + 2*diffs.std()].index.tolist()
	return {"breakpoints": breakpoints, "rolling_mean": rolling_mean, "rolling_std": rolling_std}

def chow_test(df: pd.DataFrame, column: str, split_index: int) -> float:
	"""Performs the Chow test for a structural break at split_index."""
	y = df[column].dropna().values
	X = np.arange(len(y)).reshape(-1, 1)
	X = add_constant(X)
	split = split_index
	model_full = OLS(y, X).fit()
	model_1 = OLS(y[:split], X[:split]).fit()
	model_2 = OLS(y[split:], X[split:]).fit()
	rss_full = sum(model_full.resid ** 2)
	rss_1 = sum(model_1.resid ** 2)
	rss_2 = sum(model_2.resid ** 2)
	k = X.shape[1]
	n1 = split
	n2 = len(y) - split
	f_stat = ((rss_full - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (n1 + n2 - 2 * k))
	return f_stat

def detect_outliers(df: pd.DataFrame, column: str, method: str = "zscore", threshold: float = 3.0) -> pd.Series:
	"""Detects outliers in a column using z-score or IQR method."""
	if method == "zscore":
		z = (df[column] - df[column].mean()) / df[column].std()
		return z.abs() > threshold
	elif method == "iqr":
		q1 = df[column].quantile(0.25)
		q3 = df[column].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		return (df[column] < lower) | (df[column] > upper)
	else:
		raise ValueError("Unknown method: choose 'zscore' or 'iqr'")

def robust_zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.5) -> pd.Series:
	"""Detects outliers using the robust z-score (median absolute deviation)."""
	median = df[column].median()
	mad = np.median(np.abs(df[column] - median))
	if mad == 0:
		return pd.Series([False]*len(df), index=df.index)
	robust_z = 0.6745 * (df[column] - median) / mad
	return robust_z.abs() > threshold

def rolling_variance(df: pd.DataFrame, column: str, window: int = 5) -> pd.Series:
	"""Computes rolling variance for vintage drift detection."""
	return df[column].rolling(window=window).var()

def select_imputation_strategy(df: pd.DataFrame, column: str) -> str:
	"""Suggests an imputation strategy based on missingness and data type."""
	missing_pct = df[column].isnull().mean()
	if missing_pct < 0.05:
		return "linear_interpolation"
	elif missing_pct < 0.2:
		return "mean_imputation"
	else:
		return "flag_and_review"

def score_metadata_quality(df: pd.DataFrame, metadata: Dict[str, Any]) -> float:
	"""Scores metadata quality (completeness, consistency, documentation)."""
	required_fields = ["source", "units", "frequency", "last_updated"]
	present = sum(1 for f in required_fields if f in metadata and metadata[f])
	return present / len(required_fields)

def audit_data(raw_data: Any) -> Any:
	"""
	Audit and validate the quality, completeness, and consistency of harvested data.
	Returns validated data (format TBD).
	"""
	pass

# -----------------------------
# 3. Economic Risk Modeler Agent
# -----------------------------
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# from lightgbm import LGBMClassifier  # Uncomment if using LightGBM
# from sksurv.linear_model import CoxPHSurvivalAnalysis  # For survival models
# from econml.dml import CausalForestDML  # For causal forest
# import pymc3 as pm  # For Bayesian state-space models
import shap

def compute_fiscal_risk_score(row, weights=None):
	"""
	Computes the Fiscal Risk Score (FRS) as a weighted sum of risk factors.
	"""
	if weights is None:
		weights = {'debt_gdp': 0.3, 'inflation': 0.2, 'growth_shock': 0.3, 'reserves_risk': 0.2}
	return (
		weights['debt_gdp'] * row['debt_gdp'] +
		weights['inflation'] * row['inflation'] +
		weights['growth_shock'] * row['growth_shock'] +
		weights['reserves_risk'] * row['reserves_risk']
	)

def statistical_early_warning(X, y):
	"""
	Tier 1: Fit a logistic regression for early warning signal.
	"""
	model = LogisticRegression()
	model.fit(X, y)
	return model

def ml_risk_model(X, y):
	"""
	Tier 2: Fit an XGBoost model for risk prediction.
	"""
	model = xgb.XGBClassifier(eval_metric='logloss')
	model.fit(X, y)
	explainer = shap.Explainer(model, X)
	return model, explainer

# --- Advanced: Causal ML, Bayesian, Dynamic Factor (stubs) ---
def causal_forest_policy_impact(X, y, T):
	"""Tier 3: Estimate policy impact using causal forest (stub)."""
	pass  # Implement with econml or similar

def bayesian_state_space_model(data):
	"""Tier 3: Bayesian state-space model for fiscal risk (stub)."""
	pass  # Implement with pymc3 or similar

def dynamic_factor_model(data):
	"""Tier 3: Dynamic factor model for macro risk (stub)."""
	pass  # Implement with statsmodels or similar

def rule_based_risk_flag(row):
	"""Example rule-based risk flag logic for macro-fiscal stress."""
	if (
		row['debt_gdp'] > 70 and
		row['inflation'] > 15 and
		row['gdp_growth'] < 2
	):
		return 'HIGH'
	elif (
		row['debt_gdp'] > 60 or
		row['inflation'] > 10 or
		row['gdp_growth'] < 3
	):
		return 'MEDIUM'
	else:
		return 'LOW'

def probabilistic_risk_flag(prob: float, high_thresh=0.8, med_thresh=0.5):
	"""Probabilistic risk flag assignment based on model output probability."""
	if prob >= high_thresh:
		return 'HIGH'
	elif prob >= med_thresh:
		return 'MEDIUM'
	else:
		return 'LOW'

def model_risk(validated_data: Any) -> List[Any]:
	"""
	Compute quarterly fiscal risk scores and early warning flags using validated data.
	Returns a list of RiskScore objects.
	"""
	pass

def generate_early_warning_flags(validated_data: Any) -> List[Any]:
	"""
	Generate early warning flags for macro-fiscal stress (6–12 month horizon).
	Returns a list of EarlyWarningFlag objects.
	"""
	pass

# -----------------------------
# 4. Scenario Simulator Agent
# -----------------------------
import random

def monte_carlo_macro_shock(df, shock_dict, n_sim=1000, seed=42):
	"""
	Runs Monte Carlo simulations applying macro shocks to input DataFrame.
	"""
	np.random.seed(seed)
	sims = []
	for _ in range(n_sim):
		sim_df = df.copy()
		for col, shock in shock_dict.items():
			sim_df[col] += np.random.normal(loc=shock, scale=abs(shock)*0.2 if shock != 0 else 0.1)
		sims.append(sim_df)
	return sims

def bayesian_var_simulation(df, shocks: dict, n_draws=1000):
	"""Stub for Bayesian VAR scenario simulation (gold standard for macro policy analysis)."""
	pass

def local_projection_scenario(df, shock_col: str, shock_value: float, horizon: int = 8):
	"""Stub for local projections (Jordà method) to estimate impulse responses."""
	pass

def summarize_scenario_probabilities(baseline: float, adverse: float, policy: float) -> str:
	"""Formats scenario output for reporting."""
	return (
		f"Probability of fiscal distress (baseline): {baseline:.0%}\n"
		f"Under adverse shock: {adverse:.0%}\n"
		f"Under policy adjustment: {policy:.0%}"
	)

def simulate_scenarios(validated_data: Any) -> Any:
	"""
	Simulate macro-fiscal scenarios and stress tests for 6–12 month horizons.
	Returns scenario outputs (format TBD).
	"""
	pass

# -----------------------------
# 5. Policy Intelligence Agent
# -----------------------------
POLICY_KB = [
	{
		"pattern": "high_debt_low_growth",
		"criteria": lambda r: r["debt_gdp"] > 70 and r["gdp_growth"] < 2,
		"action": "fiscal consolidation",
		"reference": "Zambia (2015–2018): Phased fiscal consolidation of 1.5–2% of GDP."
	},
	{
		"pattern": "high_inflation",
		"criteria": lambda r: r["inflation"] > 15,
		"action": "monetary tightening",
		"reference": "IMF-supported programs: Aggressive rate hikes to anchor expectations."
	},
	{
		"pattern": "low_reserves",
		"criteria": lambda r: r["reserves_months"] < 3,
		"action": "FX management",
		"reference": "Mozambique (2016): FX interventions and import compression."
	},
	{
		"pattern": "growth_shock",
		"criteria": lambda r: r["growth_shock"] < -2,
		"action": "targeted stimulus",
		"reference": "Botswana (2009): Countercyclical fiscal stimulus."
	},
]

def recommend_policy_actions(row) -> str:
	"""
	Matches country macro conditions to policy templates and provides explainable recommendations.
	"""
	for entry in POLICY_KB:
		if entry["criteria"](row):
			return (
				f"Based on similarities with {entry['reference']}, "
				f"a {entry['action']} is recommended."
			)
	return "No specific policy template matched. Recommend expert review."

def generate_policy_playbook(scenarios: Any) -> List[Any]:
	"""
	Generate policy response playbooks and recommendations based on scenario outputs.
	Returns a list of PolicyPlaybook objects.
	"""
	pass

# -----------------------------
# 6. Report Generator Agent
# -----------------------------
import json

def generate_ministerial_brief(country, risk_score, flags, scenarios, recommendations, filename="ministerial_brief.txt"):
	"""
	Generates a concise 2-page ministerial brief (text format for demo).
	"""
	with open(filename, "w") as f:
		f.write(f"Executive Summary\n=================\n")
		f.write(f"Country: {country}\nFiscal Risk Score: {risk_score}\nFlags: {flags}\n\n")
		f.write(f"Key Drivers: {', '.join(flags)}\n\n")
		f.write(f"Scenario Analysis: {scenarios}\n\n")
		f.write(f"Policy Recommendations: {recommendations}\n\n")
		f.write(f"Data Quality Notes: See annex.\n")

def generate_technical_annex(data_quality, shap_values, filename="technical_annex.txt"):
	"""
	Generates a technical annex with data quality and SHAP explainability details.
	"""
	with open(filename, "w") as f:
		f.write("Technical Annex\n===============\n")
		f.write(f"Data Quality Notes: {data_quality}\n\n")
		f.write(f"SHAP Key Drivers: {shap_values}\n")

def generate_dashboard_json(country, risk_score, flags, scenarios, recommendations, filename="dashboard.json"):
	"""
	Outputs dashboard data as JSON for web or BI tools.
	"""
	dashboard = {
		"country": country,
		"risk_score": risk_score,
		"flags": flags,
		"scenarios": scenarios,
		"recommendations": recommendations
	}
	with open(filename, "w") as f:
		json.dump(dashboard, f, indent=2)

def plot_risk_heatmap(risk_matrix, countries, filename="risk_heatmap.png"):
	"""
	Plots and saves a risk heatmap for SADC countries.
	"""
	plt.figure(figsize=(10, 6))
	plt.imshow(risk_matrix, cmap="Reds", aspect="auto")
	plt.xticks(range(len(countries)), countries, rotation=45)
	plt.yticks([])
	plt.colorbar(label="Fiscal Risk Score")
	plt.title("SADC Country Fiscal Risk Heatmap")
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()

def generate_executive_summary(country, risk_score, flags):
	return f"{country}: Fiscal Risk Score {risk_score}, Flags: {', '.join(flags)}"

def generate_country_risk_map(risk_scores):
	return {r['country']: r['score'] for r in risk_scores}

def generate_key_drivers(shap_values):
	return f"Key drivers (SHAP): {shap_values}"

def generate_scenario_analysis(scenarios):
	return f"Scenario analysis: {scenarios}"

def generate_policy_recommendations(recommendations):
	return f"Policy recommendations: {recommendations}"

def generate_data_quality_notes(notes):
	return f"Data quality notes: {notes}"

def generate_policy_brief_report(
	risk_scores: List[Any],
	warning_flags: List[Any],
	playbooks: List[Any]
) -> List[Any]:
	"""
	Produce automated PDF/HTML policy briefs for stakeholders.
	Returns a list of PolicyBriefReport objects.
	"""
	pass
# --- Policy Intelligence Agent: Knowledge-Based Recommendations ---
POLICY_KB = [
	{
		"pattern": "high_debt_low_growth",
		"criteria": lambda r: r["debt_gdp"] > 70 and r["gdp_growth"] < 2,
		"action": "fiscal consolidation",
		"reference": "Zambia (2015–2018): Phased fiscal consolidation of 1.5–2% of GDP."
	},
	{
		"pattern": "high_inflation",
		"criteria": lambda r: r["inflation"] > 15,
		"action": "monetary tightening",
		"reference": "IMF-supported programs: Aggressive rate hikes to anchor expectations."
	},
	{
		"pattern": "low_reserves",
		"criteria": lambda r: r["reserves_months"] < 3,
		"action": "FX management",
		"reference": "Mozambique (2016): FX interventions and import compression."
	},
	{
		"pattern": "growth_shock",
		"criteria": lambda r: r["growth_shock"] < -2,
		"action": "targeted stimulus",
		"reference": "Botswana (2009): Countercyclical fiscal stimulus."
	},
]

def recommend_policy_actions(row) -> str:
	"""
	Matches country macro conditions to policy templates and provides explainable recommendations.
	Args:
		row: pd.Series or dict with macro indicators
	Returns:
		String with recommended action and historical reference
	"""
	for entry in POLICY_KB:
		if entry["criteria"](row):
			return (
				f"Based on similarities with {entry['reference']}, "
				f"a {entry['action']} is recommended."
			)
	return "No specific policy template matched. Recommend expert review."
from typing import Tuple
import random

# --- Scenario Simulator Agent: Advanced Counterfactuals ---
def monte_carlo_macro_shock(df, shock_dict, n_sim=1000, seed=42):
	"""
	Runs Monte Carlo simulations applying macro shocks to input DataFrame.
	Args:
		df: pd.DataFrame of macro variables
		shock_dict: dict of {col: shock_value} to apply (e.g., {"inflation": 5})
		n_sim: number of simulations
		seed: random seed
	Returns:
		List of simulated DataFrames
	"""
	np.random.seed(seed)
	sims = []
	for _ in range(n_sim):
		sim_df = df.copy()
		for col, shock in shock_dict.items():
			sim_df[col] += np.random.normal(loc=shock, scale=abs(shock)*0.2 if shock != 0 else 0.1)
		sims.append(sim_df)
	return sims

def bayesian_var_simulation(df, shocks: dict, n_draws=1000):
	"""
	Stub for Bayesian VAR scenario simulation (gold standard for macro policy analysis).
	Args:
		df: pd.DataFrame of macro variables
		shocks: dict of {col: shock_value}
		n_draws: number of posterior draws
	Returns:
		Placeholder for Bayesian VAR output
	"""
	# Implement with pymc3, statsmodels, or similar
	pass

def local_projection_scenario(df, shock_col: str, shock_value: float, horizon: int = 8):
	"""
	Stub for local projections (Jordà method) to estimate impulse responses.
	Args:
		df: pd.DataFrame
		shock_col: variable to shock
		shock_value: magnitude of shock
		horizon: periods to project
	Returns:
		Placeholder for impulse response estimates
	"""
	# Implement with statsmodels or custom regression
	pass

def summarize_scenario_probabilities(baseline: float, adverse: float, policy: float) -> str:
	"""
	Formats scenario output for reporting.
	Args:
		baseline: Probability of fiscal distress (baseline)
		adverse: Probability under adverse shock
		policy: Probability under policy adjustment
	Returns:
		Formatted string summary
	"""
	return (
		f"Probability of fiscal distress (baseline): {baseline:.0%}\n"
		f"Under adverse shock: {adverse:.0%}\n"
		f"Under policy adjustment: {policy:.0%}"
	)
# Tools for SADC Economic Resilience Monitor

from typing import Any, List
from src.models import Country, RiskScore, EarlyWarningFlag, PolicyPlaybook, PolicyBriefReport

# --- Data Harvester Agent: Enhanced Macro-Fiscal Data Fetching ---
import wbgapi as wb
import pandas as pd

INDICATORS = {
	"NY.GDP.MKTP.KD.ZG": "gdp_growth",
	"FP.CPI.TOTL.ZG": "inflation",
	"GC.DOD.TOTL.GD.ZS": "debt_gdp",
	"GC.BAL.CASH.GD.ZS": "fiscal_balance",
	"BN.CAB.XOKA.GD.ZS": "current_account",
	"FI.RES.TOTL.MO": "reserves_months"
}

def fetch_sadc_data(countries: List[str], start: int = 2005, end: int = 2025) -> pd.DataFrame:
	"""
	Fetches macro-fiscal indicators for SADC countries from the World Bank API.
	Args:
		countries: List of ISO country codes (e.g., ['ZWE', 'ZMB'])
		start: Start year (default 2005)
		end: End year (default 2025)
	Returns:
		DataFrame with macro-fiscal indicators, columns renamed for clarity.
	"""
	df = wb.data.DataFrame(
		INDICATORS.keys(),
		economy=countries,
		time=range(start, end)
	)
	return df.rename(columns=INDICATORS)

def harvest_data(countries: List[Country]) -> Any:
	"""
	Collect macro-fiscal and economic data for the given SADC countries.
	Returns raw data (format TBD).
	"""
	pass

def audit_data(raw_data: Any) -> Any:
	"""
	Audit and validate the quality, completeness, and consistency of harvested data.
	Returns validated data (format TBD).
	"""
	pass

def model_risk(validated_data: Any) -> List[RiskScore]:
	"""
	Compute quarterly fiscal risk scores and early warning flags using validated data.
	Returns a list of RiskScore objects.
	"""
	pass

def generate_early_warning_flags(validated_data: Any) -> List[EarlyWarningFlag]:
	"""
	Generate early warning flags for macro-fiscal stress (6–12 month horizon).
	Returns a list of EarlyWarningFlag objects.
	"""
	pass

def simulate_scenarios(validated_data: Any) -> Any:
	"""
	Simulate macro-fiscal scenarios and stress tests for 6–12 month horizons.
	Returns scenario outputs (format TBD).
	"""
	pass

def generate_policy_playbook(scenarios: Any) -> List[PolicyPlaybook]:
	"""
	Generate policy response playbooks and recommendations based on scenario outputs.
	Returns a list of PolicyPlaybook objects.
	"""
	pass

def generate_policy_brief_report(
	risk_scores: List[RiskScore],
	warning_flags: List[EarlyWarningFlag],
	playbooks: List[PolicyPlaybook]
) -> List[PolicyBriefReport]:
	"""
	Produce automated PDF/HTML policy briefs for stakeholders.
	Returns a list of PolicyBriefReport objects.
	"""
	pass





# --- Economic Risk Modeler Agent: Core Intelligence ---
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# from lightgbm import LGBMClassifier  # Uncomment if using LightGBM
# from sksurv.linear_model import CoxPHSurvivalAnalysis  # For survival models
# from econml.dml import CausalForestDML  # For causal forest
# import pymc3 as pm  # For Bayesian state-space models
import shap

def compute_fiscal_risk_score(row, weights=None):
	"""
	Computes the Fiscal Risk Score (FRS) as a weighted sum of risk factors.
	Args:
		row: pd.Series with keys ['debt_gdp', 'inflation', 'growth_shock', 'reserves_risk']
		weights: dict with weights for each factor (default: equal weights)
	Returns:
		FRS value (float)
	"""
	if weights is None:
		weights = {'debt_gdp': 0.3, 'inflation': 0.2, 'growth_shock': 0.3, 'reserves_risk': 0.2}
	return (
		weights['debt_gdp'] * row['debt_gdp'] +
		weights['inflation'] * row['inflation'] +
		weights['growth_shock'] * row['growth_shock'] +
		weights['reserves_risk'] * row['reserves_risk']
	)

def statistical_early_warning(X, y):
	"""
	Tier 1: Fit a logistic regression for early warning signal.
	Args:
		X: pd.DataFrame of features
		y: pd.Series of binary outcomes (e.g., crisis event)
	Returns:
		Fitted model
	"""
	model = LogisticRegression()
	model.fit(X, y)
	return model

def ml_risk_model(X, y):
	"""
	Tier 2: Fit an XGBoost model for risk prediction.
	Args:
		X: pd.DataFrame of features
		y: pd.Series of binary outcomes
	Returns:
		Fitted XGBoost model, SHAP explainer
	"""
	model = xgb.XGBClassifier(eval_metric='logloss')
	model.fit(X, y)
	explainer = shap.Explainer(model, X)
	return model, explainer


# --- Advanced: Causal ML, Bayesian, Dynamic Factor (stubs) ---
def causal_forest_policy_impact(X, y, T):
	"""
	Tier 3: Estimate policy impact using causal forest (stub).
	Args:
		X: Features, y: outcome, T: treatment
	Returns:
		Placeholder for causal forest model
	"""
	pass  # Implement with econml or similar

def bayesian_state_space_model(data):
	"""
	Tier 3: Bayesian state-space model for fiscal risk (stub).
	Args:
		data: pd.DataFrame
	Returns:
		Placeholder for Bayesian model
	"""
	pass  # Implement with pymc3 or similar

def dynamic_factor_model(data):
	"""
	Tier 3: Dynamic factor model for macro risk (stub).
	Args:
		data: pd.DataFrame
	Returns:
		Placeholder for dynamic factor model
	"""
	pass  # Implement with statsmodels or similar


def rule_based_risk_flag(row):
	"""
	Example rule-based risk flag logic for macro-fiscal stress.
	Args:
		row: pd.Series with keys ['debt_gdp', 'inflation', 'gdp_growth']
	Returns:
		String risk flag: 'HIGH', 'MEDIUM', 'LOW', or None
	"""
	if (
		row['debt_gdp'] > 70 and
		row['inflation'] > 15 and
		row['gdp_growth'] < 2
	):
		return 'HIGH'
	elif (
		row['debt_gdp'] > 60 or
		row['inflation'] > 10 or
		row['gdp_growth'] < 3
	):
		return 'MEDIUM'
	else:
		return 'LOW'

def probabilistic_risk_flag(prob: float, high_thresh=0.8, med_thresh=0.5):
	"""
	Probabilistic risk flag assignment based on model output probability.
	Args:
		prob: Probability of risk event (float 0-1)
		high_thresh: Threshold for 'HIGH' flag
		med_thresh: Threshold for 'MEDIUM' flag
	Returns:
		String risk flag: 'HIGH', 'MEDIUM', 'LOW'
	"""
	if prob >= high_thresh:
		return 'HIGH'
	elif prob >= med_thresh:
		return 'MEDIUM'
	else:
		return 'LOW'



import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# --- Recommended Data Quality Checks ---
def plot_missing_data_heatmap(df: pd.DataFrame, figsize=(10, 6)) -> None:
	"""
	Plots a heatmap of missing data for visual inspection.
	"""
	plt.figure(figsize=figsize)
	plt.imshow(df.isnull(), aspect='auto', cmap='viridis', interpolation='none')
	plt.xlabel('Variables')
	plt.ylabel('Observations')
	plt.title('Missing Data Heatmap')
	plt.colorbar(label='Missing')
	plt.show()

def robust_zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.5) -> pd.Series:
	"""
	Detects outliers using the robust z-score (median absolute deviation).
	Returns a boolean Series where True indicates an outlier.
	"""
	median = df[column].median()
	mad = np.median(np.abs(df[column] - median))
	if mad == 0:
		return pd.Series([False]*len(df), index=df.index)
	robust_z = 0.6745 * (df[column] - median) / mad
	return robust_z.abs() > threshold

def chow_test(df: pd.DataFrame, column: str, split_index: int) -> float:
	"""
	Performs the Chow test for a structural break at split_index.
	Returns the F-statistic.
	"""
	y = df[column].dropna().values
	X = np.arange(len(y)).reshape(-1, 1)
	X = add_constant(X)
	split = split_index
	model_full = OLS(y, X).fit()
	model_1 = OLS(y[:split], X[:split]).fit()
	model_2 = OLS(y[split:], X[split:]).fit()
	rss_full = sum(model_full.resid ** 2)
	rss_1 = sum(model_1.resid ** 2)
	rss_2 = sum(model_2.resid ** 2)
	k = X.shape[1]
	n1 = split
	n2 = len(y) - split
	f_stat = ((rss_full - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (n1 + n2 - 2 * k))
	return f_stat

def rolling_variance(df: pd.DataFrame, column: str, window: int = 5) -> pd.Series:
	"""
	Computes rolling variance for vintage drift detection.
	Returns a Series of rolling variances.
	"""
	return df[column].rolling(window=window).var()
import numpy as np
from typing import Dict

# --- Data Quality Auditor Agent: Donor-Grade Data Auditing ---
def profile_missingness(df: pd.DataFrame) -> pd.Series:
	"""
	Returns the percentage of missing values per column.
	"""
	return df.isnull().mean() * 100

def detect_structural_breaks(df: pd.DataFrame, column: str) -> Dict[str, Any]:
	"""
	Detects structural breaks in a time series column using rolling mean/variance.
	Returns breakpoints and summary stats.
	"""
	# Simple rolling mean/variance change detection (placeholder for advanced methods)
	window = min(5, len(df)//2)
	rolling_mean = df[column].rolling(window=window).mean()
	rolling_std = df[column].rolling(window=window).std()
	diffs = rolling_mean.diff().abs()
	breakpoints = diffs[diffs > diffs.mean() + 2*diffs.std()].index.tolist()
	return {"breakpoints": breakpoints, "rolling_mean": rolling_mean, "rolling_std": rolling_std}

def detect_outliers(df: pd.DataFrame, column: str, method: str = "zscore", threshold: float = 3.0) -> pd.Series:
	"""
	Detects outliers in a column using z-score or IQR method.
	Returns a boolean Series where True indicates an outlier.
	"""
	if method == "zscore":
		z = (df[column] - df[column].mean()) / df[column].std()
		return z.abs() > threshold
	elif method == "iqr":
		q1 = df[column].quantile(0.25)
		q3 = df[column].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		return (df[column] < lower) | (df[column] > upper)
	else:
		raise ValueError("Unknown method: choose 'zscore' or 'iqr'")

def select_imputation_strategy(df: pd.DataFrame, column: str) -> str:
	"""
	Suggests an imputation strategy based on missingness and data type.
	"""
	missing_pct = df[column].isnull().mean()
	if missing_pct < 0.05:
		return "linear_interpolation"
	elif missing_pct < 0.2:
		return "mean_imputation"
	else:
		return "flag_and_review"

def score_metadata_quality(df: pd.DataFrame, metadata: Dict[str, Any]) -> float:
	"""
	Scores metadata quality (completeness, consistency, documentation).
	Returns a score between 0 and 1.
	"""
	required_fields = ["source", "units", "frequency", "last_updated"]
	present = sum(1 for f in required_fields if f in metadata and metadata[f])
	return present / len(required_fields)
