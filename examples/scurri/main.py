from examples.scurri.agents import *
from swarm import run_demo_loop

if __name__ == "__main__":
    run_demo_loop(agents = [
        general_agent,
        stock_alert_agent,
        order_and_shipment_agent,
        verification_agent
    ], debug = True)