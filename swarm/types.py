from typing import List, Callable, Union, Optional

# We use pydantic to enforce type safety
from pydantic import BaseModel

# Define a type alias for a function that can take any number of arguments and return either a string, an Agent instance, or a dictionary
AgentFunction = Callable[..., Union[str, "Agent", dict]]

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