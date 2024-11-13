import copy
import json
import logging

from typing import List
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from swarm import Response
from swarm.multiagent import debug_print, handle_tool_calls, Agent

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
            events=history,
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
            agent_context = {"task": tool_results.events}
            agent_response = self.run_test(
                agent=tool_results.agent,
                # TODO: Each Agent receives the latest message from the user. They don't need to know anything else they've said in the past, or the outputs of other agents and their tools
                messages=[],
                context_variables=agent_context,
                model_override=model_override,
                debug=debug,
                max_sequential_turns=max_sequential_turns,
                execute_tools=execute_tools,
            )
            # Compile results back to the triage agent
            tool_results.events.extend(agent_response.events)
            context_variables.update(agent_response.context_variables)
        history.extend(tool_results.events)
        context_variables.update(tool_results.context_variables)

        '''
        TODO: Need to modify this so sub-agent completions and tool calls aren't added to the Triage Agent's history.
        The Triage Agent doesn't need to know about the inner workings of the Agents that it invokes, just the final result it produces...
        Essentially we need to maintain a new history for each agent we invoke until we're finished (no more Agent transfers or tool calls) and need to go back to the Triage agent, when the histories can be wiped.
        This is very close though...
        '''

    return Response(
        messages=history,
        agent=agent,
        context_variables=context_variables,
        tokens_used=tokens_used
    )