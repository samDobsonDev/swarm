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
        tokens_used (int): The number of tokens used in generating the response.
    """
    events: List = []
    agent: Optional[Agent] = None
    tokens_used: int = 0

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
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )
        case _:
            try:
                return Result(value=str(raw_result))
            except Exception as e:
                error_message = f"Failed to cast response to string: {raw_result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                debug_print(debug, error_message)
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
            debug_print(debug, f"Tool {name} not found in function map.")
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
            debug_print(debug, f"Calling tool: {name} with no arguments")
            raw_result = func()
        else:
            debug_print(debug, f"Calling tool: {name} with arguments {args}")
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
    def __init__(self, client = None):
        if not client:
            client = OpenAI()
        self.client = client
        self.context_manager = ContextManager()

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
        user_message_content = "\n".join(
            format_event(event) for event in events
        )
        messages = [
            {"role": "system", "content": agent_instructions},
            {"role": "user", "content": user_message_content}
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
        debug_print(debug, "Getting chat completion for...:", json.dumps(create_params, indent=3))
        return self.client.chat.completions.create( **create_params)

    """
    This is the current implementation of agent transfer/handover:
    
    1. Call LLM
    2. LLM responds specifying it wants to use a tool or multiple tools, as well as the parameters to pass to said tool/s
    3. We invoke the LLM's chosen tools and produce a Response object. When initializing this object we set the Agent field to None.
    For each tool we call, we use the result of the tool call to produce a Result object. Each Result contains a value and a potential Agent, if the tool returned one.
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
    
    NOTE: In this design, the user communicates with whatever the active agent is at the time. There isn't a 
    centralized agent that the user communicates with exclusively and that hands over to other agents with the intentioned that they will
    relay any information they uncover back to this centralized agent. This approach solves the problem of an agentic system where there is only
    one agent, which is that above around 5-10 tool options, the agent can begin to make mistakes on which tool to pick and when. With this 
    multi-agent approach, each agent has a specific domain they work in with a set selection of tools.
    
    TODO: Currently, each agent receives the full conversation history thus far as it is constantly amended to by each agent. The only thing that changes is the system message
    that each agent receives. This means that agents are potentially receiving unnecessary information from the conversation history in order to complete their task.
    I need to devise a way in which when we transfer to another agent, it only receives the context it needs and not the conversation history.
    This way, we can keep each agent on task and minimize hallucinations. 
    
    One way to approach this might be having the agents share some overall state, which is similar to what we do now with amending the conversation history with more and more
    messages (that includes tools calls, results of tool calls, agent transfer and normal conversation history).
    The difference is this state will be more concentrated and include less bloat, and can include any agent transfers, tool calls and their results by agents, as well as the conversation history. 
    This state would also be in chronological order to provide context to the LLM as to when tool calls, agent transfers or normal conversation occurred.
    This means we essentially provide a timeline to the LLM. In this approach, similarly to the current approach, the user wouldn't be communicating exclusively with the top-level/centralized agent 
    in the conversation, but rather would be communicating with whatever the active agent is at the time. 
    
    However, there is a downside to this solution. Take staff in a store as an example. Let's say a customer walks in a says to the store manager "Hi there, I need to return this device, it doesn't work. Can I get a replacement?".
    The store manager analyses the situation and the device, where they concur that the item is indeed damaged, the battery is broken. The store manager decides to hand this over to a supervisor, to see if the purchase is eligible for
    a replacement, or if the only option is a refund. The supervisor doesn't need to know the details of the item and how it's damaged, just that the store manager has asked them to check if the purchase is eligible
    for a replacement or refund, indicating that the store manager has previously checked the item and approved the return. The supervisor performs their checks and determines that the customer can receive a replacement device.
    The supervisor hands this over to the stock clerk to locate the replacement product. The stock clerk doesn't need to know context of what this item is for, or that it's a replacement for a damaged device, or how the device
    was damaged, just that it needs to locate the replacement product. See where I'm going with this? The context provided to each staff member in the store is limited to only essential information in order to allow the
    staff member to complete their task. They don't need to know the entire story from when the customer entered the store. The above approach neglects this, and allows the stock clerk to gain insight into how the product the
    customer initially bought was damaged, which is just unnecessary information and doesn't impact the task of the stock clerk in anyway, other than just overloading it with irrelevant information. 
    It isn't in the best interest of a sub-agent to know the entire context of the session, just the relevant details to ensure the agent stays on task and is as efficient as possible.
    
    So taking this into account, what is a suitable approach? What agent/s should the user communicate with (centralized vs active agent)? How should we construct the context that each sub-agent receives as it is transferred to? 
    """
    def run(
        self,
        agent: Agent,
        events: List,
        model_override: str = None,
        debug: bool = False,
        max_sequential_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        logging.basicConfig(level=logging.ERROR)
        active_agent = agent
        events_history = copy.deepcopy(events)
        init_len = len(events)
        tokens_used = 0
        while len(events_history) - init_len < max_sequential_turns and active_agent:
            # agent_context = self.context_manager.get_context_for_agent(active_agent.name)
            completion: ChatCompletion = self.get_chat_completion(
                agent=active_agent,
                events=events_history,
                model_override=model_override,
                debug=debug,
            )
            message: ChatCompletionMessage = completion.choices[0].message
            debug_print(debug, "Received completion:", str(message))
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
            tokens_used += completion.usage.total_tokens
            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break
            tools_response: Response = handle_tool_calls(
                tool_calls = message.tool_calls, agent = active_agent, debug = debug
            )
            events_history.extend(tools_response.events)
            if tools_response.agent:
                active_agent = tools_response.agent
        return Response(
            events=events_history[init_len:],
            agent=active_agent,
            tokens_used=tokens_used
        )

def pretty_print_messages(events) -> None:
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
    agent: Agent, debug = False
) -> None:
    client = Swarm()
    print("Starting Scurri AI Concierge...")
    events = []
    agent = agent
    total_tokens_used = 0
    while True:
        user_input = input("\033[90mUser\033[0m: ")
        events.append({
            "originator": "user",
            "event": "user_message",
            "content": user_input
        })
        response = client.run(agent= agent, events = events, debug = debug)
        pretty_print_messages(response.events)
        total_tokens_used += response.tokens_used
        debug_print(debug, f"Total tokens used in this session: {total_tokens_used}")
        events.extend(response.events)
        agent = response.agent
