import importlib
from graphviz import Digraph

def visualize_agents():
    dot = Digraph(comment='Multi-Agent Workflow')

    # Import the agents module
    agents_module = importlib.import_module('examples.airline.configs.agents')

    # List of agent names to import
    agent_names = [
        'triage_agent',
        'flight_modification',
        'flight_cancel',
        'flight_change',
        'lost_baggage'
    ]

    # Dictionary to hold agent objects
    agents = {}

    # Load agent objects
    for name in agent_names:
        agent = getattr(agents_module, name)
        agents[agent.name] = agent

    # Add nodes and edges to the graph
    for agent_name, agent in agents.items():
        dot.node(agent_name, agent_name)
        for function in agent.functions:
            function_name = function.__name__
            dot.node(function_name, function_name, shape='box')
            dot.edge(agent_name, function_name)

            # Check if the function is a transfer function
            if function_name.startswith('transfer_to_'):
                # Extract the target agent name from the function name
                target_agent_name = function_name.replace('transfer_to_', '').replace('_', ' ').title() + ' Agent'
                if target_agent_name in agents:
                    dot.edge(function_name, target_agent_name)

    # Render the graph
    dot.render('multi_agent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_agents()