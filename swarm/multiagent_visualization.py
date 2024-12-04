from graphviz import Digraph

def visualize_multiagent_workflow():
    dot = Digraph(graph_attr={
        'rankdir': 'LR',
        'splines': 'ortho',
        'nodesep': '0.5',
    })

    # Define the steps in the multiagent process with descriptions
    steps = {
        "User Input": "User types a message,\nadded to events history.",
        "Determine Agent": "Determine the appropriate\nAgent to handle the request.",
        "Call LLM": "Call the LLM with\nevents history and\nAgent's tools/functions.",
        "LLM Tool Decision": "LLM decides whether\nto use a tool\nor respond directly.",
        "Tool Call": "Call the tool/function\nand add response\nto events history.",
        "LLM Response": "LLM returns a normal\nmessage, added to\nevents history.",
        "Return to User": "Return the response\nto the user.",
    }

    # Add nodes for each step with descriptions using HTML-like labels
    for step, description in steps.items():
        label = f"""<
            <table border="0" cellborder="0">
                <tr><td><font point-size="30"><b>{step}</b></font></td></tr>
                <tr><td><font point-size="20">{description.replace('\n', '<br/>')}</font></td></tr>
            </table>
        >"""
        dot.node(step, label=label, shape='box', style='filled')

    # Add edges to represent the process flow
    dot.edge("User Input", "Determine Agent")
    dot.edge("Determine Agent", "Call LLM")
    dot.edge("Call LLM", "LLM Tool Decision")
    dot.edge("LLM Tool Decision", "Tool Call", label="If tool needed")
    dot.edge("Tool Call", "Call LLM", label="Tool response")
    dot.edge("LLM Tool Decision", "LLM Response", label="If no tool needed")
    dot.edge("LLM Response", "Return to User")
    dot.edge("Return to User", "User Input", label="Cycle repeats")

    dot.render('multiagent_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_multiagent_workflow()