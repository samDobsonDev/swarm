from graphviz import Digraph
from examples.airline.configs.agents import triage_agent
from swarm.multiagent import Agent

def visualize_agents(initial_agent: Agent):
    dot = Digraph(graph_attr={'rankdir': 'LR', 'splines': 'polyline'})
    agents = {initial_agent.name: initial_agent}  # Dictionary to hold agent objects
    agent_queue = [initial_agent]  # Queue for processing agents

    # Process each agent in the queue
    while agent_queue:
        current_agent = agent_queue.pop(0)
        dot.node(current_agent.name, current_agent.name, shape='ellipse', style='filled', fillcolor='lightblue')

        for function in current_agent.functions:
            function_name = function.__name__
            dot.node(function_name, function_name, shape='box', style='rounded,filled', fillcolor='lightgrey')
            dot.edge(current_agent.name, function_name)

            # Check if the function returns an Agent
            result = function()
            if isinstance(result, Agent):
                if result.name not in agents:
                    agents[result.name] = result
                    agent_queue.append(result)
                # Add the edge even if the agent is already in the graph
                dot.edge(function_name, result.name)

    # Render the graph
    dot.render('multi_agent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_agents(triage_agent)