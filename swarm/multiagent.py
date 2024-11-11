# Standard library imports
import copy
import json
import logging
import inspect
from collections import defaultdict
from typing import List, Callable, Union, Optional
from pydantic import BaseModel
from datetime import datetime

# OpenAI Package/library imports
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

def debug_print(debug: bool, *args: str) -> None:
    if not debug: return  # If debugging is not enabled, exit the function
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time formatted as a timestamp
    message = " ".join(map(str, args))  # Convert all arguments to strings and join them into a single message
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")  # Print the timestamp and message with color formatting

# Define a type alias for a function that can take any number of arguments and return either a string, an Agent instance, or a dictionary
AgentFunction = Callable[..., Union[str, "Agent", dict]]

def function_to_json(func: AgentFunction) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {  # Map Python types to JSON schema types
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    try:
        signature = inspect.signature(func)  # Get the function's signature
    except ValueError as e:
        raise ValueError(  # Raise an error if the signature cannot be obtained
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    parameters = {}  # Initialize a dictionary to hold parameter details
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")  # Get the parameter type
        except KeyError as e:
            raise KeyError(  # Raise an error if the type annotation is unknown
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}  # Add parameter type to the dictionary
    required = [
        param.name  # Extract the name of the parameter
        for param in signature.parameters.values()  # Iterate over each parameter in the function's signature
        if param.default is param.empty  # Include only parameters without default values
    ]
    return {  # Return the function signature as a JSON-serializable dictionary
        "type": "function",
        "function": {
            "name": func.__name__,  # Function name
            "description": func.__doc__ or "",  # Function description from docstring
            "parameters": {
                "type": "object",
                "properties": parameters,  # Parameter details
                "required": required,  # Required parameters
            },
        },
    }

class Agent(BaseModel):
    """
    Represents an agent with specific attributes and behaviors.

    Attributes:
        name (str): The name of the agent.
        model (str): The model type used by the agent.
        instructions (Union[str, Callable[[], str]]): Instructions or a function returning instructions for the agent.
        functions (List[AgentFunction]): A list of functions the agent can perform.
        tool_choice (str): The tool choice for the agent, if any.
        parallel_tool_calls (bool): Whether the agent can make parallel tool calls.
    """
    name: str = "Agent"
    model: str = "gpt-4o"
    # Use Union to allow instructions to be either a string or a callable that takes a dictionary input and returns a string
    instructions: Union[str, Callable[[dict], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    # Gives the option for the model to pick multiple tools in a single response, indicating that they should be called in parallel.
    parallel_tool_calls: bool = True

class Response(BaseModel):
    """
    Represents a response from an agent.

    Attributes:
        messages (List): A list of messages in the response.
        agent (Optional[Agent]): The agent that generated the response.
        context_variables (dict): Context variables associated with the response.
        tokens_used (int): The number of tokens used in generating the response.
    """
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}
    tokens_used: int = 0

class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Optional[Agent]): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """
    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}

__CTX_VARS_NAME__ = "context_variables"

def handle_function_result(raw_result, debug) -> Result:  # Define a function to process the raw_result and return a Result object
    match raw_result:  # Use pattern matching to handle different types of results
        case Result() as result:  # If the result is already a Result object
            return result  # Return it directly
        case Agent() as agent:  # If the result is an Agent object
            return Result(  # Create a new Result object
                value=json.dumps({"assistant": agent.name}),  # Serialize the agent's name as JSON
                agent=agent,  # Include the agent in the Result
            )
        case _:  # For any other type of result
            try:
                return Result(value=str(raw_result))  # Attempt to convert the result to a string and return it as a Result
            except Exception as e:  # If conversion fails
                error_message = f"Failed to cast response to string: {raw_result}. Make sure agent functions return a string or Result object. Error: {str(e)}"  # Prepare an error message
                debug_print(debug, error_message)  # Print the error message if debugging is enabled
                raise TypeError(error_message)  # Raise a TypeError with the error message


def handle_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],  # List of tool calls to process. This is provided by the model
    functions: List[AgentFunction],  # List of functions to active agent has as its disposal
    context_variables: dict,  # Context variables to pass to functions
    debug: bool,  # Debug flag for logging
) -> Response:  # Returns a Response object
    function_map = {f.__name__: f for f in functions}  # Map function names to function objects
    tools_response = Response(messages=[], agent=None, context_variables={})  # Initialize an empty Response object
    for tool_call in tool_calls:  # Iterate over each tool call
        name = tool_call.function.name  # Get the function name from the tool call
        if name not in function_map:  # Check if the function is not in the map
            debug_print(debug, f"Tool {name} not found in function map.")  # Log missing tool
            tools_response.messages.append(  # Add an error message to the tools_response
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue  # Skip to the next tool call
        args = json.loads(tool_call.function.arguments)  # Parse the function arguments model tools_response JSON
        debug_print(debug, f"Processing tool call: {name} with arguments {args}")  # Log the tool call processing
        func = function_map[name]  # Retrieve the function object from the map
        if callable(func) and hasattr(func, '__code__'):  # Ensure func is callable and has a __code__ attribute
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:  # Check if the function accepts context variables
                args[__CTX_VARS_NAME__] = context_variables  # Add context variables to the arguments
        raw_result = function_map[name]()  # Call the function with arguments. TODO: Make it so functions can accepts arguments like normal
        result: Result = handle_function_result(raw_result, debug)  # Process the function result
        tools_response.messages.append(  # Add the function result to the tools_response messages
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            }
        )
        tools_response.context_variables.update(result.context_variables)  # Update context variables in the tools_response
        if result.agent:  # Check if the result includes an agent
            tools_response.agent = result.agent  # Set the agent in the tools_response
    return tools_response  # Return the constructed tools_response

class Swarm:
    def __init__(self, client = None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,  # The agent to interact with
        history: List,  # The list of messages (conversation history)
        context_variables: dict,  # Contextual variables for the conversation
        model_override: str,  # Optional model override
        debug: bool,  # Whether to enable debug mode
    ) -> ChatCompletion:
        context_variables = defaultdict(str,context_variables)  # Initialize context variables with default string values
        instructions = (  # Determine the instructions/system message for the agent
            agent.instructions(context_variables)  # Call the instructions function if it's callable
            if callable(agent.instructions)
            else agent.instructions  # Use the instructions string directly if not callable
        )
        messages = [{"role": "system","content": instructions}] + history  # Prepend system instructions to the message history
        tools = [function_to_json(func = f) for f in agent.functions]  # Convert agent functions to JSON format for tools
        for tool in tools:  # Iterate over each tool
            params = tool["function"]["parameters"]  # Access the function parameters
            params["properties"].pop(__CTX_VARS_NAME__, None)  # Remove context variables from properties
            if __CTX_VARS_NAME__ in params["required"]:  # Check if context variables are required
                params["required"].remove(__CTX_VARS_NAME__)  # Remove context variables from required list
        create_params = {  # Prepare parameters for the chat completion request
            "model": model_override or agent.model,  # Use model override if provided, otherwise use agent's model
            "messages": messages,  # Include the messages
            "tools": tools or None,  # Include tools if available
            "tool_choice": agent.tool_choice,  # Specify the tool choice
        }
        if tools:  # If tools are available...
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls  # ...allow parallel tool calls
        # Debug print the full create_params object
        debug_print(debug, "Getting chat completion for...:", json.dumps(create_params, indent=4))  # Log the parameters if debugging
        return self.client.chat.completions.create( **create_params)  # Make the chat completion request with the prepared parameters

    """
    This is the current implementation of agent transfer/handover:
    
    1. Call LLM
    2. LLM responds specifying it wants to use a tool or multiple tools
    3. We invoke the LLM's chosen tools and produce a Response object. When initializing this object we set the Agent field to None.
    For each tool we call, we use the result of the tool call to produce a Result object. Each Result contains a value and a potential Agent. 
    The Response object has a messages field. We use the Result objects to help construct fake "LLM" response messages which we add to the conversation history.
    Each message adheres to the following schema:
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "tool_name": name, # Name of the function called, e.g. "transfer_to_triage"
        "content": result.value, # This will be the agent's name, e.g. "Triage Agent"
    }
    5. When all of the tools have been invoked from the LLM's response (Step 2), we then add the Response.messages to the conversation history. This includes the 
    result of each tool call which provides context to the LLM going forward. If Response.agent is populated and different the current active_agent, we change the active_agent to said agent.
    6. Go through this loop again and again until all agent transfers and tool calls have been made (Steps 1-5).
    7. At this point we'll call the LLM one more time with the new conversation history. At this point no tools will be invoked and
    we'll just receive a normal response from the LLM, which we will go back to the user with. During this call, the LLM will
    have context an all agent transfers and tool calls that have happened, which their resulting values. This gives it all the context
    it needs to produce a response for the user.
    
    TODO: Each agent receives the full conversation history thus far as it is constantly amended to by each agent. The only thing that changes is the system message
    that the agent receives. This means that agents are potentially receiving unnecessary information from the conversation history in order to complete their task.
    I need to devise a way in which when we transfer to another agent, it only receives the context it needs from the conversation history, and not the entire thing.
    This way, we can keep the agent on task and minimize hallucinations. 
    
    One way to approach this might be having the agents share some overall state, rather than simply amending the conversation history with more and more
    messages (that includes tools calls, results of tool calls, agent transfer and normal conversation history) like we're doing now, which looks bloated.
    This state can include any agent transfers, tool calls and their results by agents, as well as the conversation history. This state
    would also be in chronological order to provide context to the LLM as to when tool calls, agent transfers or normal conversation occurred.
    This means we essentially provide a timeline to the LLM. In this approach, the user wouldn't be communicating exclusively with the top-level/centralized agent in the conversation, such as a Triage Agent, but rather
    would be communicating with whatever the active agent is at the time. 
    
    However, there is a downside to this approach. Take staff in a store as an example. Let's say a customer walks in a says to the store manager "Hi there, I need to return this device, it doesn't work. Can I get a replacement?".
    The store manager analyses the situation and the device, where they concur that the item is indeed damaged, the battery is broken. The store manager decides to hand this over to a supervisor, to see if the purchase is eligible for
    a replacement, or if the only option is a refund. The supervisor doesn't need to know the details of the item and how it's damaged, just that the store manager has asked them to check if the purchase is eligible
    for a replacement or refund, indicating that the store manager has previously checked the item and approved the return. The supervisor performs their checks and determines that the customer can receive a replacement device.
    The supervisor hands this over to the stock clerk to locate the replacement product. The stock clerk doesn't need to know context of what this item is for, or that it's a replacement for a damaged device, or how the device
    was damaged, just that it needs to locate the replacement product. See where I'm going with this? The context provided to each staff member in the store is limited to only essential information in order to allow the
    staff member to complete their task. They don't need to know the entire story from when the customer entered the store. The above approach neglects this, and allows the stock clerk to gain insight into how the product the
    customer initially bought was damaged, which is just unnecessary information and doesn't impact the task of the stock clerk in anyway, other than just overloading it with irrelevant information. 
    It isn't in the best interest of a sub-agent to know the entire context of the session, just the relevant details to ensure the agent stays on task and is as efficient as possible
    """
    def run(
        self,
        agent: Agent,  # The agent to interact with
        messages: List,  # The list of messages (conversation history)
        context_variables = None,  # Contextual variables for the conversation
        model_override: str = None,  # Optional model override
        debug: bool = False,  # Whether to enable debug mode
        max_sequential_turns: int = float("inf"),  # The amount of sequential turns the assistant can have before returning back to the user, default to positive infinity
        execute_tools: bool = True,  # Whether to execute tool calls
    ) -> Response:
        logging.basicConfig(level=logging.ERROR)
        # Initialize context_variables as an empty dictionary if not provided
        if context_variables is None:
            context_variables = {}
        active_agent = agent  # Set the active agent
        context_variables = copy.deepcopy(context_variables)  # Deep copy context variables
        history = copy.deepcopy(messages)  # Deep copy the message history
        init_len = len(messages)  # Initial length of the message history
        tokens_used = 0  # Initialize token usage counter
        # While the length of the conversation history - the initial length of the conversation history is less than max_sequential_turns, and we have an active agent...
        while len(history) - init_len < max_sequential_turns and active_agent:
            completion: ChatCompletion = self.get_chat_completion(  # Get ChatCompletion with current history and active agent
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
            )
            message: ChatCompletionMessage = completion.choices[0].message  # Extract the first ChatCompletionMessage from the ChatCompletion
            debug_print(debug, "Received completion:", str(message))  # Print the received ChatCompletionMessage
            message.sender = active_agent.name  # Set the sender of the ChatCompletionMessage, this is used in pretty_print_messages() in repl.py
            history.append(json.loads(message.model_dump_json()))  # Append the message to the history by converting it to JSON
            tokens_used += completion.usage.total_tokens  # Update tokens used in interaction (prompt + response)
            if not message.tool_calls or not execute_tools:  # Check if there are no tool calls in the ChatCompletionMessage, or if tools should not be executed
                debug_print(debug, "Ending turn.")  # Debug print ending turn
                break  # Exit the while loop
            tools_response: Response = handle_tool_calls(  # Call chosen tools and retrieve result
                tool_calls = message.tool_calls, functions = active_agent.functions, context_variables = context_variables, debug = debug
            )
            history.extend(tools_response.messages)  # Extend history with tool results
            context_variables.update(tools_response.context_variables)  # Update context variables
            if tools_response.agent and tools_response.agent != active_agent:  # If there's a new agent, and it's different to the current...
                active_agent = tools_response.agent  # ...switch to the new agent

        # Keep going until we don't call any more tools, and we break on line 286 above. We'll always have an active_agent.
        # That condition is just there so we continue the loop
        return Response(
            messages=history[init_len:],  # Messages from the initial length of history to the end of the array (i.e, all new messages generated in this interaction)
            agent=active_agent,  # The active agent
            context_variables=context_variables,  # Updated context variables
            tokens_used=tokens_used  # Total tokens used
        )

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
    client = Swarm() # Initialize a Swarm instance
    print("Starting Scurri AI Concierge...")
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
        messages.extend(response.messages) # Extend the conversation history with the agent's responses
        agent = response.agent # Update the current agent if a transfer occurred
