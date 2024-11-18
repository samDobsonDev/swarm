from configs.agents import *
from swarm import run_demo_loop

if __name__ == "__main__":
    run_demo_loop(agents = [
        general_agent,
        flight_change,
        flight_cancellation,
        lost_baggage
    ], debug = True)
