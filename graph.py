import networkx as nx
import matplotlib.pyplot as plt

# Define agent connections
edges = [
    ('Supervisor', 'DataHarvester'),
    ('Supervisor', 'DataQualityAuditor'),
    ('Supervisor', 'EconomicRiskModeler'),
    ('Supervisor', 'ScenarioSimulator'),
    ('Supervisor', 'PolicyIntelligence'),
    ('Supervisor', 'ReportGenerator'),
    ('DataHarvester', 'DataQualityAuditor'),
    ('DataQualityAuditor', 'EconomicRiskModeler'),
    ('EconomicRiskModeler', 'ScenarioSimulator'),
    ('ScenarioSimulator', 'PolicyIntelligence'),
    ('PolicyIntelligence', 'ReportGenerator')
]

G = nx.DiGraph()
G.add_edges_from(edges)

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
plt.title('Agent Linkage Graph')
plt.savefig('reports/agent_linkage_graph.png')
plt.show()
print('Agent linkage graph saved as reports/agent_linkage_graph.png')

