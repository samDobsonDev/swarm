import json

from swarm import Swarm, Agent
from swarm.util import debug_print

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant": # Only process messages from the assistant (agent)
            continue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ") # Print the agent's name in blue
        if message["content"]: # Print the message content if available
            print(message["content"])
        tool_calls = message.get("tool_calls") or [] # Print tool calls in purple, if any
        if len(tool_calls) > 1:
            print()  # Add a newline if there are multiple tool calls
        for tool_call in tool_calls:
            f = tool_call["function"] # Extract the function name
            name, args = f["name"], f["arguments"] # Extract the arguments
            arg_str = json.dumps(json.loads(args)).replace(":", "=") # Format the arguments for display
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})") # Print the tool call in purple

def run_demo_loop(
    starting_agent: Agent, context_variables = None, debug = False
) -> None:
    client = Swarm() # Initialize a Swarm client instance
    print("Starting Swarm CLI 🐝")
    messages = [] # Initialize the conversation history
    agent = starting_agent  # Set the starting agent
    total_tokens_used = 0  # Initialize a variable to track total tokens
    while True:
        user_input = input("\033[90mUser\033[0m: ") # Prompt the user for input...
        messages.append({"role": "user", "content": user_input}) # ...and append it to the conversation history
        response = client.run(agent = agent, messages = messages, context_variables = context_variables, debug = debug)  # Run the client with the current agent, conversation history and context variables
        pretty_print_messages(response.messages)
        total_tokens_used += response.tokens_used # Update the total tokens used
        debug_print(debug, f"Total tokens used in this session: {total_tokens_used}") # Print the total tokens used so far in the session
        messages.extend(response.messages) # Extend the conversation history with the agent's response
        agent = response.agent # Update the current agent if a transfer occurred
