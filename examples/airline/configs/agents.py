from examples.airline.configs.tools import *
from examples.airline.data.routines.baggage.policies import *
from examples.airline.data.routines.flight_modification.policies import *
from examples.airline.data.routines.prompts import STARTER_PROMPT

from swarm import Agent

def transfer_to_flight_modification():
    return flight_modification

def transfer_to_flight_cancellation():
    return flight_cancellation


def transfer_to_flight_change():
    return flight_change


def transfer_to_lost_baggage():
    return lost_baggage


def transfer_to_triage():
    """Call this function when a user needs to be transferred to a different agent and a different policy.
    For instance, if a user is asking about a topic that is not handled by the current agent, call this function.
    """
    return triage_agent

context_variables = {
    "customer_context": """Here is what you know about the customer's details:
1. CUSTOMER_ID: customer_12345
2. NAME: John Doe
3. PHONE_NUMBER: (123) 456-7890
4. EMAIL: johndoe@example.com
5. STATUS: Premium
6. ACCOUNT_STATUS: Active
7. BALANCE: $0.00
8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
""",
    "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024.""",
}

def triage_instructions():
    customer_context = context_variables.get("customer_context", None)
    flight_context = context_variables.get("flight_context", None)
    return f"""You are to triage a users request, and call a tool to transfer to the right intent.
    Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.
    You dont need to know specifics, just the topic of the request.
    When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it.
    Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user.
    The customer context is here: {customer_context}, and flight context is here: {flight_context}"""


triage_agent = Agent(
    name="Triage",
    instructions=triage_instructions,
    functions=[transfer_to_flight_modification, transfer_to_lost_baggage],
)

flight_modification = Agent(
    name="Flight Modification Agent",
    instructions="""You are a Flight Modification Agent for a customer service airlines company.
      You are an expert customer service agent deciding which sub intent the user should be referred to.
You already know the intent is for flight modification related question. First, look at message history and see if you can determine if the user wants to cancel or change their flight.
Ask user clarifying questions until you know whether or not it is a cancel request or change flight request. Once you know, call the appropriate transfer function. Either ask clarifying questions, or call one of your functions, every time.""",
    functions=[transfer_to_flight_cancellation, transfer_to_flight_change, transfer_to_triage],
    parallel_tool_calls=False,
)

def general_instructions():
    customer_context = context_variables.get("customer_context", None)
    flight_context = context_variables.get("flight_context", None)
    return f"""You are a helpful customer service assistant for an airline company.
    Your role is to handle general inquiries and pleasantries that do not require specific routing to other agents.
    You should provide helpful and friendly responses to the user's questions.
    We can aid the user with flight changes, flight cancellations and lost baggage requests.
    The customer context is here: {customer_context}, and flight context is here: {flight_context}"""

general_agent = Agent(
    name="GeneralAgent",
    instructions=general_instructions,
    functions=[],
)

flight_change = Agent(
    name="FlightChangeAgent",
    instructions=STARTER_PROMPT + FLIGHT_CHANGE_POLICY,
    functions=[
        escalate_to_agent,
        change_flight,
        valid_to_change_flight,
        case_resolved,
        check_flight_availability
    ],
)

flight_cancellation = Agent(
    name="FlightCancellationAgent",
    instructions=STARTER_PROMPT + FLIGHT_CANCELLATION_POLICY,
    functions=[
        escalate_to_agent,
        initiate_refund,
        initiate_flight_credits,
        case_resolved,
    ],
)

lost_baggage = Agent(
    name="LostBaggageAgent",
    instructions=STARTER_PROMPT + LOST_BAGGAGE_POLICY,
    functions=[
        escalate_to_agent,
        initiate_baggage_search,
        case_resolved,
    ],
)
