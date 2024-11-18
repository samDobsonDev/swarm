import unittest

from examples.airline.configs.agents import *
from swarm.multiagent import Swarm, Agent, Response

class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"originator": "user", "event": "user_message", "content": "I need help with my account settings."},
            {"originator": "AccountAgent", "event": "assistant_message", "content": "Sure, I can help with that. What seems to be the issue?"},
            {"originator": "user", "event": "user_message", "content": "I can't update my billing information."},
            {"originator": "BillingAgent", "event": "tool_call", "tool_name": "UpdateBillingInfo", "arguments": "{\"action\": \"update_billing_info\"}"},
            {"originator": "BillingAgent", "event": "tool_output", "tool_name": "UpdateBillingInfo", "content": "Billing information updated successfully."},
            {"originator": "BillingAgent", "event": "assistant_message", "content": "Your billing information has been updated successfully. Let me know if you need any more help."},
            {"originator": "user", "event": "user_message", "content": "Cheers!"},
            {"originator": "GeneralAgent", "event": "assistant_message", "content": "You're welcome! If you have any other questions, feel free to ask."},
            {"originator": "user", "event": "user_message", "content": "I have another question about my lost baggage. And I also need to cancel my flight."}
        ]

    def test_get_chat_completion(self):
        swarm = Swarm(agents = [general_agent, flight_change, flight_cancellation, lost_baggage])
        run: Response = swarm.run(
            events = self.events,
        )
        print("New Events:", run.events)
        print("Chosen agent:", run.agent)
        print("Input Tokens Used:", run.input_tokens_used)
        print("Output Tokens Used:", run.output_tokens_used)

if __name__ == '__main__':
    unittest.main()