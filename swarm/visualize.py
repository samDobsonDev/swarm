from graphviz import Digraph
from examples.airline.configs.agents import triage_agent
from swarm.multiagent import Agent

def visualize_agents(initial_agent: Agent):
    dot = Digraph(graph_attr={
        'rankdir': 'LR',
        'splines': 'true',
        'nodesep': '0.5',
        'concentrate': 'true'
    })
    agents = {initial_agent.name: initial_agent}  # Dictionary to hold agent objects
    agent_queue = [initial_agent]  # Queue for processing agents

    # Initialize a set to track added function-agent edges
    added_function_agent_edges = set()

    # Process each agent in the queue
    while agent_queue:
        current_agent = agent_queue.pop(0)
        dot.node(current_agent.name, current_agent.name, shape='ellipse', style='filled', fillcolor='lightblue')



        for function in current_agent.functions:
            function_name = function.__name__
            dot.node(function_name, function_name, shape='box', style='rounded,filled', fillcolor='lightgrey')

            # Add edge from agent to function
            dot.edge(current_agent.name, function_name)

            # Check if the function returns an Agent
            result = function()
            if isinstance(result, Agent):
                if result.name not in agents:
                    agents[result.name] = result
                    agent_queue.append(result)

                # Create a unique edge identifier for function-agent pair
                function_agent_id = (function_name, result.name)

                # Add the edge only if it hasn't been added yet
                if function_agent_id not in added_function_agent_edges:
                    dot.edge(function_name, result.name)
                    added_function_agent_edges.add(function_agent_id)

    # Render the graph
    dot.render('multi_agent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_agents(triage_agent)