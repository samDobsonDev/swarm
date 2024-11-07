# Standard library imports
import copy
import json
import logging
from collections import defaultdict
from typing import List, Any, Generator

# Package/library imports
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionMessage, ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

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
    tool_result = Response(messages=[], agent=None, context_variables={})  # Initialize an empty Response object
    for tool_call in tool_calls:  # Iterate over each tool call
        name = tool_call.function.name  # Get the function name from the tool call
        if name not in function_map:  # Check if the function is not in the map
            debug_print(debug, f"Tool {name} not found in function map.")  # Log missing tool
            tool_result.messages.append(  # Add an error message to the response
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
        tool_result.messages.append(  # Add the function result to the response messages
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            }
        )
        tool_result.context_variables.update(result.context_variables)  # Update context variables in the response
        if result.agent:  # Check if the result includes an agent
            tool_result.agent = result.agent  # Set the agent in the response
    return tool_result  # Return the constructed response

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
            tool_results: Response = handle_tool_calls(  # Call chosen tools and retrieve result
                tool_calls = message.tool_calls, functions = active_agent.functions, context_variables = context_variables, debug = debug
            )
            history.extend(tool_results.messages)  # Extend history with tool results
            context_variables.update(tool_results.context_variables)  # Update context variables
            if tool_results.agent and tool_results.agent != active_agent:  # If there's a new agent, and it's different to the current...
                active_agent = tool_results.agent  # ...switch to the new agent
        return Response(
            messages=history,  # Messages from the initial length of history to the end of the array (i.e, all new messages generated in this interaction)
            agent=active_agent,  # The active agent
            context_variables=context_variables,  # Updated context variables
            tokens_used=tokens_used  # Total tokens used
        )

    def run_test(
            self,
            agent: Agent,  # The triage agent
            messages: List,  # The list of messages (conversation history)
            context_variables=None,  # Contextual variables for the conversation
            model_override: str = None,  # Optional model override
            debug: bool = False,  # Whether to enable debug mode
            max_sequential_turns: int = float("inf"),
            # The amount of sequential turns the assistant can have before returning back to the user, default to positive infinity
            execute_tools: bool = True,  # Whether to execute tool calls
    ) -> Response:
        logging.basicConfig(level=logging.ERROR)
        if context_variables is None:
            context_variables = {}
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        tokens_used = 0

        # Main loop for the triage agent
        while len(history) < max_sequential_turns:
            completion: ChatCompletion = self.get_chat_completion(
                agent=agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
            )
            message: ChatCompletionMessage = completion.choices[0].message
            debug_print(debug, "Received completion:", str(message))
            message.sender = agent.name
            history.append(json.loads(message.model_dump_json()))
            tokens_used += completion.usage.total_tokens

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # Handle tool calls and agent handovers
            tool_results: Response = handle_tool_calls(
                tool_calls=message.tool_calls,
                functions=agent.functions,
                context_variables=context_variables,
                debug=debug,
            )

            # If a new agent needs to be invoked, handle it in a separate chat completion
            if tool_results.agent and tool_results.agent != agent:
                # Isolate context for the new agent
                agent_context = {"task": tool_results.messages}
                agent_response = self.run_test(
                    agent=tool_results.agent,
                    # TODO: Need to modify this so that the Agent has at least some context about what's going on in the conversation and why it's been invoked, rather than nothing...
                    messages=[],
                    context_variables=agent_context,
                    model_override=model_override,
                    debug=debug,
                    max_sequential_turns=max_sequential_turns,
                    execute_tools=execute_tools,
                )
                # Compile results back to the triage agent
                tool_results.messages.extend(agent_response.messages)
                context_variables.update(agent_response.context_variables)
            history.extend(tool_results.messages)
            context_variables.update(tool_results.context_variables)

            '''
            TODO: Need to modify this so sub-agent completions and tool calls aren't added to the Triage Agent's history.
            The Triage Agent doesn't need to know about the inner workings of the Agents that it invokes, just the final result it produces...
            Essentially we need to maintain a new history for each agent we invoke until we're finished and need to go back to the Triage agent, when the histories can be wiped.
            This is very close though...
            '''

        return Response(
            messages=history,
            agent=agent,
            context_variables=context_variables,
            tokens_used=tokens_used
        )
