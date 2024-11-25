from examples.scurri.tools import *
from swarm.es.elasticsearch_client import ElasticSearchClient
from swarm.es.repositories import UserRepository, OrderRepository

from swarm import Agent

# Initialize Elasticsearch client and repositories
es_client = ElasticSearchClient("http://54.154.188.102:9200")
user_repo = UserRepository(es_client)
order_repo = OrderRepository(es_client)

def get_customer_context():
    user = user_repo.find_user_by_email("footasylum", "footasylum", email)
    if user:
        verified_status = "Verified" if user.get('verifiedChannels') else "Not Verified"
        return f"""Here is what you know about the customer's details:
1. NAME: {user.get('firstName')} {user.get('lastName')}
2. EMAIL: {user.get('email')}
3. VERIFIED STATUS: {verified_status}
"""
    return "Customer details not found."

customer_context = get_customer_context()

RETAILER_STARTER_PROMPT = "You are a knowledgeable and compassionate customer support assistant for the fashion brand Footasylum."

def general_instructions():
    return RETAILER_STARTER_PROMPT + f"""Your role is to handle general inquiries and pleasantries.
        The customer context is here: {customer_context}"""

def stock_alert_instructions():
    return RETAILER_STARTER_PROMPT + f"""Your role is to manage stock alerts for the customer.
        The customer context is here: {customer_context}"""

def order_instructions():
    return RETAILER_STARTER_PROMPT + f"""You are an Order Agent.
        Your role is to manage orders for the customer.
        The customer context is here: {customer_context}"""

general_agent = Agent(
    name="GeneralAgent",
    instructions=general_instructions,
    functions=[],
)

stock_alert_agent = Agent(
    name="StockAlertAgent",
    instructions=stock_alert_instructions,
    functions=[get_stock_alerts_tool, add_stock_alert_tool, remove_stock_alert_tool],
)

order_agent = Agent(
    name="OrderAgent",
    instructions=order_instructions,
    functions=[get_completed_orders_tool, get_incomplete_orders_tool, get_latest_order_tool]
)