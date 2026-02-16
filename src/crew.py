import os
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.tools import *
from src.models import Country, RiskScore, EarlyWarningFlag, PolicyPlaybook, PolicyBriefReport

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

@CrewBase
class SADCResilienceCrew:
    agents_config_path = os.path.join(CONFIG_DIR, 'agents.yaml')
    tasks_config_path = os.path.join(CONFIG_DIR, 'tasks.yaml')

    def __init__(self, countries=None):
        self.countries = countries
        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

    @agent
    def data_harvester_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_harvester'],
            tools=[fetch_sadc_data, harvest_data],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @agent
    def data_quality_auditor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_quality_auditor'],
            tools=[profile_missingness, detect_structural_breaks, detect_outliers, robust_zscore_outliers, select_imputation_strategy, score_metadata_quality, audit_data],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @agent
    def economic_risk_modeler_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['economic_risk_modeler'],
            tools=[compute_fiscal_risk_score, statistical_early_warning, ml_risk_model, rule_based_risk_flag, probabilistic_risk_flag, model_risk, generate_early_warning_flags],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @agent
    def scenario_simulator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['scenario_simulator'],
            tools=[monte_carlo_macro_shock, bayesian_var_simulation, local_projection_scenario, simulate_scenarios, summarize_scenario_probabilities],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @agent
    def policy_intelligence_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_intelligence'],
            tools=[recommend_policy_actions, generate_policy_playbook],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @agent
    def report_generator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'],
            tools=[generate_ministerial_brief, generate_technical_annex, generate_dashboard_json, plot_risk_heatmap, generate_executive_summary, generate_country_risk_map, generate_key_drivers, generate_scenario_analysis, generate_policy_recommendations, generate_data_quality_notes, generate_policy_brief_report],
            allow_delegation=False,
            max_iter=3,
            verbose=True
        )

    @task
    def data_harvesting_task(self) -> Task:
        task_config = self.tasks_config['data_harvesting']
        return Task(
            description=task_config['description'],
            agent=self.data_harvester_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def data_quality_audit_task(self) -> Task:
        task_config = self.tasks_config['data_quality_audit']
        return Task(
            description=task_config['description'],
            agent=self.data_quality_auditor_agent(),
            depends_on=['data_harvesting_task'],
            expected_output=task_config['expected_output']
        )

    @task
    def risk_modeling_task(self) -> Task:
        task_config = self.tasks_config['risk_modeling']
        return Task(
            description=task_config['description'],
            agent=self.economic_risk_modeler_agent(),
            depends_on=['data_quality_audit_task'],
            expected_output=task_config['expected_output']
        )

    @task
    def scenario_simulation_task(self) -> Task:
        task_config = self.tasks_config['scenario_simulation']
        return Task(
            description=task_config['description'],
            agent=self.scenario_simulator_agent(),
            depends_on=['risk_modeling_task'],
            expected_output=task_config['expected_output']
        )

    @task
    def policy_intelligence_task(self) -> Task:
        task_config = self.tasks_config['policy_intelligence']
        return Task(
            description=task_config['description'],
            agent=self.policy_intelligence_agent(),
            depends_on=['scenario_simulation_task'],
            expected_output=task_config['expected_output']
        )

    @task
    def report_generation_task(self) -> Task:
        task_config = self.tasks_config['report_generation']
        return Task(
            description=task_config['description'],
            agent=self.report_generator_agent(),
            depends_on=['policy_intelligence_task'],
            expected_output=task_config['expected_output']
        )

@CrewBase
class SADCPolicyBriefCrew(SADCResilienceCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [
            self.data_harvesting_task(),
            self.data_quality_audit_task(),
            self.risk_modeling_task(),
            self.scenario_simulation_task(),
            self.policy_intelligence_task(),
            self.report_generation_task()
        ]
        agents = [
            self.data_harvester_agent(),
            self.data_quality_auditor_agent(),
            self.economic_risk_modeler_agent(),
            self.scenario_simulator_agent(),
            self.policy_intelligence_agent(),
            self.report_generator_agent()
        ]
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )