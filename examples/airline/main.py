from configs.agents import *
from swarm import run_demo_loop

if __name__ == "__main__":
    run_demo_loop(agent= triage_agent, debug = True)
