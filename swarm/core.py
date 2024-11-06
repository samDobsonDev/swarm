# Standard library imports
import copy
import json
import logging
from collections import defaultdict
from typing import List, Any, Generator

# Package/library imports
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

# Local imports
from .util import function_to_json, debug_print
from .types import Agent, AgentFunction, Response, Result

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
    partial_response = Response(messages=[], agent=None, context_variables={})  # Initialize an empty Response object
    for tool_call in tool_calls:  # Iterate over each tool call
        name = tool_call.function.name  # Get the function name from the tool call
        if name not in function_map:  # Check if the function is not in the map
            debug_print(debug, f"Tool {name} not found in function map.")  # Log missing tool
            partial_response.messages.append(  # Add an error message to the response
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue  # Skip to the next tool call
        args = json.loads(tool_call.function.arguments)  # Parse the function arguments model response JSON
        debug_print(debug, f"Processing tool call: {name} with arguments {args}")  # Log the tool call processing
        func = function_map[name]  # Retrieve the function object from the map
        if callable(func) and hasattr(func, '__code__'):  # Ensure func is callable and has a __code__ attribute
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:  # Check if the function accepts context variables
                args[__CTX_VARS_NAME__] = context_variables  # Add context variables to the arguments
        raw_result = function_map[name]()  # Call the function with arguments. TODO: Make it so functions can accepts arguments like normal
        result: Result = handle_function_result(raw_result, debug)  # Process the function result
        partial_response.messages.append(  # Add the function result to the response messages
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            }
        )
        partial_response.context_variables.update(result.context_variables)  # Update context variables in the response
        if result.agent:  # Check if the result includes an agent
            partial_response.agent = result.agent  # Set the agent in the response
    return partial_response  # Return the constructed response

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
    ) -> ChatCompletionMessage | Stream[ChatCompletionChunk]:  # Return type can be a ChatCompletionMessage or a stream of ChatCompletionChunks
        context_variables = defaultdict(str,context_variables)  # Initialize context variables with default string values
        instructions = (  # Determine the instructions for the agent
            agent.instructions(context_variables)  # Call the instructions function if it's callable
            if callable(agent.instructions)
            else agent.instructions  # Use the instructions string directly if not callable
        )
        messages = [{"role": "system","content": instructions}] + history  # Prepend system instructions to the message history
        tools = [function_to_json(f) for f in agent.functions]  # Convert agent functions to JSON format for tools
        # Hide context_variables from model
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

    def run(
        self,
        agent: Agent,  # The agent to interact with
        messages: List,  # The list of messages (conversation history)
        context_variables = None,  # Contextual variables for the conversation
        model_override: str = None,  # Optional model override
        debug: bool = False,  # Whether to enable debug mode
        max_sequential_turns: int = float("inf"),  # The amount of sequential turns the assistant can have before returning back to the user, default to positive infinity
        execute_tools: bool = True,  # Whether to execute tool calls
    ) -> Generator[dict[str, str] | dict[str, Response] | Any, Any, None] | Response:
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
            completion: ChatCompletionMessage = self.get_chat_completion(  # Get chat completion with current history and active agent
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
            )
            message = completion.choices[0].message  # Extract the message from the completion
            debug_print(debug, "Received completion:", message)  # Debug print the received message
            message.sender = active_agent.name  # Set the sender of the message, this is used in pretty_print_messages in repl.py
            history.append(  # Append the message to the history
                json.loads(message.model_dump_json())  # Convert message to JSON
            )
            tokens_used += completion.usage.total_tokens  # Update tokens used in interaction (prompt + response)
            if not message.tool_calls or not execute_tools:  # Check if there are no tool calls in the response or tools should not be executed
                debug_print(debug, "Ending turn.")  # Debug print ending turn
                break  # Exit the loop
            partial_response = handle_tool_calls(  # Handle tool calls
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)  # Extend history with partial response messages
            context_variables.update(partial_response.context_variables)  # Update context variables
            if partial_response.agent and partial_response.agent != active_agent:  # Check if there's a new agent and it's different
                active_agent = partial_response.agent  # Switch to the new agent
        return Response(  # Return the response
            messages=history[init_len:],  # Messages from the initial length to the end of the array
            agent=active_agent,  # The active agent
            context_variables=context_variables,  # Updated context variables
            tokens_used=tokens_used  # Total tokens used
        )
