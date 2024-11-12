from graphviz import Digraph
from examples.airline.configs.agents import *
from swarm.multiagent import Agent

def get_unique_color():
    # Define a list of bold, easily distinguishable colors
    colors = [
        "#FF0000",  # Red
        "#33FF57",  # Green
        "#3357FF",  # Blue
        "#FF33A1",  # Pink
        "#FF8C33",  # Orange
        "#33FFF5",  # Cyan
        "#8D33FF",  # Purple
        "#FFD433",  # Yellow
        "#33FF8C",  # Lime
        "#FF3333"   # Bright Red
    ]
    # Use a static variable to keep track of the current color index
    if not hasattr(get_unique_color, "index"):
        get_unique_color.index = 0
    # Get the current color and update the index
    color = colors[get_unique_color.index]
    get_unique_color.index = (get_unique_color.index + 1) % len(colors)
    return color

def visualize_agents(initial_agent: Agent):
    dot = Digraph(graph_attr={
        'rankdir': 'LR',
        'splines': 'ortho',
        'nodesep': '0.5',
    })
    agents = {initial_agent.name: initial_agent}
    agent_queue = [initial_agent]
    agent_colors = {}
    while agent_queue:
        current_agent = agent_queue.pop(0)
        if current_agent.name not in agent_colors:
            agent_colors[current_agent.name] = get_unique_color()
        current_color = agent_colors[current_agent.name]
        dot.node(current_agent.name, current_agent.name, shape='ellipse', style='filled', fillcolor=current_color)
        for function in current_agent.functions:
            function_name = function.__name__
            dot.node(function_name, function_name, shape='box', style='rounded,filled', fillcolor='lightgrey')
            dot.edge(current_agent.name, function_name, color=current_color)
            result = function()
            if isinstance(result, Agent):
                if result.name not in agents:
                    agents[result.name] = result
                    agent_queue.append(result)
                dot.edge(function_name, result.name, color=current_color)
    dot.render('multi_agent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_agents(triage_agent)