from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class Country(BaseModel):
	name: str
	iso_code: Optional[str] = None

class RiskScore(BaseModel):
	country: Country
	quarter: str  # e.g., '2026Q1'
	score: float
	details: Optional[str] = None

class EarlyWarningFlag(BaseModel):
	country: Country
	flag_type: str  # e.g., 'Fiscal Stress', 'External Shock'
	description: str
	horizon_months: int

class PolicyPlaybook(BaseModel):
	country: Country
	scenario: str
	recommendations: List[str]

class PolicyBriefReport(BaseModel):
	country: Country
	quarter: str
	risk_score: RiskScore
	warning_flags: List[EarlyWarningFlag]
	playbook: PolicyPlaybook
	generated_at: date
