from graphviz import Digraph

def visualize_rag_process():
    dot = Digraph(graph_attr={
        'rankdir': 'LR',
        'splines': 'ortho',
        'nodesep': '0.5',
    })

    # Define the steps in the RAG process with descriptions
    steps = {
        "Collect Data": "Collect some data\nfrom various sources.",
        "Split into Chunks": "Split data into\nmeaningful chunks\nfor model ingestion.",
        "Embed Chunks": "Pass chunks into\nan embedding model\nto convert them\ninto vectors.",
        "Store Embeddings": "Store these embeddings\nin a Tensor or\na Vector Database.",
        "User Query": "User asks\na question.",
        "Embed Query": "Convert the user's\nquestion into a vector\nusing the embedding model.",
        "Semantic Search": "Perform semantic search\nto find relevant\ndata chunks.",
        "Pass to LLM": "Pass the query and\nrelevant data to\nan LLM for\nresponse generation."
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
    dot.edge("Collect Data", "Split into Chunks")
    dot.edge("Split into Chunks", "Embed Chunks")
    dot.edge("Embed Chunks", "Store Embeddings")
    dot.edge("User Query", "Embed Query")
    dot.edge("Embed Query", "Semantic Search")
    dot.edge("Semantic Search", "Pass to LLM")
    dot.edge("Store Embeddings", "Semantic Search")

    dot.render('rag_process_workflow', format='png', cleanup=True)

if __name__ == "__main__":
    visualize_rag_process()