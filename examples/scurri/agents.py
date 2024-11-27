from examples.scurri.prompts import *
from examples.scurri.tools import *
from swarm.es.elasticsearch_client import ElasticSearchClient
from swarm.es.repositories import UserRepository, OrderRepository

from swarm import Agent

# Initialize Elasticsearch client and repositories
es_client = ElasticSearchClient("http://54.154.188.102:9200")
user_repo = UserRepository(es_client)
order_repo = OrderRepository(es_client)

def general_instructions():
    return RETAILER_STARTER_PROMPT + GENERAL_AGENT_POLICY

def stock_alert_instructions():
    return RETAILER_STARTER_PROMPT + STOCK_ALERT_POLICY

def order_and_shipment_instructions():
    return RETAILER_STARTER_PROMPT + ORDER_AND_SHIPMENT_POLICY

def verification_instructions():
    return RETAILER_STARTER_PROMPT + VERIFICATION_PROCESS_POLICY

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

order_and_shipment_agent = Agent(
    name="OrderAndShipmentAgent",
    instructions=order_and_shipment_instructions,
    functions=[get_latest_order_tool, get_shipment_details_tool]
)

verification_agent = Agent(
    name="VerificationAgent",
    instructions=verification_instructions,
    functions=[generate_verification_pin_tool, verify_user_tool, check_verification_status_tool],
)