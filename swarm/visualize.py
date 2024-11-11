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
    agents = {initial_agent.name: initial_agent}
    agent_queue = [initial_agent]
    added_function_agent_edges = set()
    while agent_queue:
        current_agent = agent_queue.pop(0)
        dot.node(current_agent.name, current_agent.name, shape='ellipse', style='filled', fillcolor='lightblue')
        for function in current_agent.functions:
            function_name = function.__name__
            dot.node(function_name, function_name, shape='box', style='rounded,filled', fillcolor='lightgrey')
            dot.edge(current_agent.name, function_name)
            result = function()
            if isinstance(result, Agent):
                if result.name not in agents:
                    agents[result.name] = result
                    agent_queue.append(result)
                function_agent_id = (function_name, result.name)
                if function_agent_id not in added_function_agent_edges:
                    dot.edge(function_name, result.name)
                    added_function_agent_edges.add(function_agent_id)
    dot.render('multi_agent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_agents(triage_agent)