from swarm import Swarm, Agent

client = Swarm()

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, events=messages)

print(response.events[-1]["content"])
