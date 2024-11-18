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
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

# BAML imports
from swarm.baml_client.types import AgentName
from swarm.baml_client.sync_client import b as baml

def debug_print(debug: bool, *args: str) -> None:
    if not debug: return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")

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
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default is param.empty
    ]
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
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
    instructions: Union[str, Callable[[dict], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

class Response(BaseModel):
    """
    Represents a response from an agent.

    Attributes:
        events (List): A list of events that occurred during this interaction.
        agent (Optional[Agent]): The Agent we need to transfer to.
        input_tokens_used (int): The number of input tokens used in generating the response.
        output_tokens_used (int): The number of output tokens used in generating the response.
    """
    events: List = []
    agent: Optional[Agent] = None
    input_tokens_used: int = 0
    output_tokens_used: int = 0

class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Optional[Agent]): The Agent instance returned by a transfer tool, if applicable.
    """
    value: str = ""
    agent: Optional[Agent] = None

__CTX_VARS_NAME__ = "context_variables"

def handle_function_result(raw_result, debug) -> Result:
    match raw_result:
        case Result() as result:
            return result
        case Agent() as agent:
            return Result(
                value=json.dumps({"agent": agent.name}),
                agent=agent,
            )
        case _:
            try:
                return Result(value=str(raw_result))
            except Exception as e:
                error_message = f"Failed to cast response to string: {raw_result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                # debug_print(debug, error_message)
                raise TypeError(error_message)


def handle_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],
    agent: Agent,
    debug: bool,
) -> Response:
    function_map = {f.__name__: f for f in agent.functions}
    tools_response = Response(messages=[], agent=None)
    for tool_call in tool_calls:
        name = tool_call.function.name
        if name not in function_map:
            # debug_print(debug, f"Tool {name} not found in function map.")
            tools_response.events.append(
                {
                    "originator": agent.name,
                    "event": "tool_output",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found."
                }
            )
            continue
        args = json.loads(tool_call.function.arguments)
        func = function_map[name]
        num_args = 0
        if callable(func) and hasattr(func, '__code__'):
            num_args = func.__code__.co_argcount
        if not args and num_args == 0:
            # debug_print(debug, f"Calling tool: {name} with no arguments")
            raw_result = func()
        else:
            # debug_print(debug, f"Calling tool: {name} with arguments {args}")
            raw_result = func(args)
        result: Result = handle_function_result(raw_result, debug)
        tools_response.events.append(
            {
                "originator": agent.name,
                "event": "tool_output",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value
            }
        )
        if result.agent:
            tools_response.agent = result.agent
    return tools_response

class ContextManager:
    def __init__(self):
        self.shared_state = {"centralized": {}, "agent_specific": defaultdict(dict)}

    def update_shared_state(self, agent_name, context):
        self.shared_state["agent_specific"][agent_name].update(context)

    def get_context_for_agent(self, agent_name):
        return {
            "centralized": self.shared_state["centralized"],
            "agent_specific": self.shared_state["agent_specific"].get(agent_name, {})
        }


def format_event(event) -> str:
    event_type = event["event"]
    if event_type == "user_message":
        return f"User: {event['content']}"
    elif event_type == "assistant_message":
        return f"Assistant: {event['content']}"
    elif event_type == "tool_call":
        return f"Tool Call: {event['tool_name']} with arguments {event['arguments']}"
    elif event_type == "tool_output":
        return f"Tool Output from {event['tool_name']}: {event['content']}"
    else:
        return f"Unknown event type: {event_type}"


class Swarm:
    def __init__(self,  agents: List[Agent], client=None):
        if not client:
            client = OpenAI()
        self.client = client
        self.context_manager = ContextManager()
        if not agents or len(agents) == 0:
            raise ValueError("You must provide at least one agent")
        self.agents = agents

    def get_chat_completion(
        self,
        agent: Agent,
        events: List,
        model_override: str,
        debug: bool,
    ) -> ChatCompletion:
        agent_instructions = (
            agent.instructions()
            if callable(agent.instructions)
            else agent.instructions
        )
        previous_events_content = "\n".join(
            format_event(event) for event in events[:-1]
        )
        latest_event_content = f"This is the latest event:\n{format_event(events[-1])}"
        full_agent_instructions = agent_instructions
        if previous_events_content:
            full_agent_instructions += (
                    "\nEvents so far in the conversation (this includes user messages, "
                    "assistant messages, agent tool calls and agent transfers):\n" +
                    previous_events_content
            )
        full_agent_instructions += "\n\n" + latest_event_content
        messages = [
            {"role": "system", "content": full_agent_instructions}
        ]
        tools = [function_to_json(func = f) for f in agent.functions]
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }
        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        # debug_print(debug, "Getting chat completion for...:", json.dumps(create_params, indent=3))
        return self.client.chat.completions.create( **create_params)

    def run(
        self,
        events: List,
        model_override: str = None,
        debug: bool = False,
        max_sequential_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        logging.basicConfig(level=logging.ERROR)
        # TODO: Before we call this, we need to confirm that the last event in events is a "user_message"
        agent_name: AgentName = baml.DecideAgentForEvents(events)
        active_agent = next((agent for agent in self.agents if agent.name == agent_name), None)
        if not active_agent:
            raise ValueError(f"No agent found with the name {agent_name}")
        debug_print(debug, "Chosen Agent:", active_agent.name)
        events_history = copy.deepcopy(events)
        init_len = len(events)
        total_input_tokens = 0
        total_output_tokens = 0
        while len(events_history) - init_len < max_sequential_turns:
            completion: ChatCompletion = self.get_chat_completion(
                agent=active_agent,
                events=events_history,
                model_override=model_override,
                debug=debug,
            )
            message: ChatCompletionMessage = completion.choices[0].message
            # debug_print(debug, "Received completion:", str(message))
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    events_history.append({
                        "originator": active_agent.name,
                        "event": "tool_call",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })
            else:
                events_history.append({
                    "originator": active_agent.name,
                    "event": "assistant_message",
                    "content": message.content,
                })
            total_input_tokens += completion.usage.prompt_tokens
            total_output_tokens += completion.usage.completion_tokens
            if not message.tool_calls or not execute_tools:
                break
            tools_response: Response = handle_tool_calls(
                tool_calls = message.tool_calls, agent = active_agent, debug = debug
            )
            events_history.extend(tools_response.events)
            if tools_response.agent:
                active_agent = tools_response.agent
        return Response(
            events = events_history[init_len:],
            agent = active_agent,
            input_tokens_used = total_input_tokens,
            output_tokens_used = total_output_tokens
        )

def pretty_print_events(events):
    for event in events:
        originator = event["originator"]
        event_type = event["event"]
        if event_type == "user_message":
            print(f"\033[90mUser\033[0m: {event['content']}")
        elif event_type == "tool_call":
            tool_name = event["tool_name"]
            arguments = event["arguments"]
            print(f"\033[94m{originator}\033[0m called tool \033[95m{tool_name}\033[0m with arguments {arguments}")
        elif event_type == "assistant_message":
            print(f"\033[94m{originator}\033[0m: {event['content']}")
        elif event_type == "tool_output":
            tool_name = event["tool_name"]
            content = event["content"]
            print(f"\033[94m{originator}\033[0m tool output from \033[95m{tool_name}\033[0m: {content}")
        else:
            print(f"\033[91mUnknown event type\033[0m: {event_type} from {originator}")

def run_demo_loop(
    agents: List[Agent], debug = False
) -> None:
    client = Swarm(agents = agents)
    print("Starting Scurri AI Concierge...")
    events = []
    total_input_tokens_used = 0
    total_output_tokens_used = 0
    while True:
        user_input = input("\033[90mUser\033[0m: ")
        events.append({
            "originator": "user",
            "event": "user_message",
            "content": user_input
        })
        response = client.run(events = events, debug = debug)
        pretty_print_events(response.events)
        debug_print(debug, "Ending turn.")
        total_input_tokens_used += response.input_tokens_used
        total_output_tokens_used += response.output_tokens_used
        debug_print(
            debug,
            f"Total input tokens used in this session: {total_input_tokens_used}. Total output tokens used in this session: {total_output_tokens_used}. Total tokens used in this session: {total_input_tokens_used + total_output_tokens_used}"
        )
        events.extend(response.events)
