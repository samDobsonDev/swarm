generator target {
  // Valid values: "python/pydantic", "typescript", "ruby/sorbet"
  output_type "python/pydantic"
  
  // Where the generated code will be saved (relative to baml_src/)
  output_dir "../swarm"
  
  // What interface you prefer to use for the generated code (sync/async)
  // Both are generated regardless of the choice, just modifies what is exported
  // at the top level
  default_client_mode "sync"
  
  // Version of runtime to generate code for (should match installed baml-py version)
  version "0.68.0"
}

client<llm> GPT4o {
  provider openai
  options {
    model gpt-4o
    api_key env.OPENAI_API_KEY
  }
}

enum AgentName {
  GeneralAgent @description("Handles pleasantries and general inquiries we don't need to route to other agents.")
  FlightCancellationAgent @description("Handles flight cancellation requests.")
  FlightChangeAgent @description("Handles flight change requests.")
  LostBaggageAgent @description("Handles lost baggage requests.")
}

class Event {
  originator string
  event string
  tool_name string?
  content string?
  arguments string? // This is a JSON string
}

function DecideAgentForEvents(events: Event[]) -> AgentName {
  client GPT4o

  prompt #"
    Based on the following list of events that have occurred in the conversation thus far with the user, decide which ONE agent is best suited to handle the situation. Even if multiple agents are needed, only choose the most relevant one:

    Previous Events:
    {% for event in events[:-1] %}
    {% set originator = event.originator %}
    {% set event_type = event.event %}
    {% if event_type == "user_message" %}
      User: {{ event.content }}
    {% elif event_type == "tool_call" %}
      {{ originator }}: called tool "{{ event.tool_name }}" with arguments {{ event.arguments }}
    {% elif event_type == "assistant_message" %}
      {{ originator }}: {{ event.content }}
    {% elif event_type == "tool_output" %}
      {{ event.tool_name }}: {{ event.content }}
    {% else %}
      Unknown event type: {{ event_type }} from {{ originator }}
    {% endif %}
    {% endfor %}

    Latest Event:
    {% set latest_event = events[-1] %}
    {% set originator = latest_event.originator %}
    {% set event_type = latest_event.event %}
    {% if event_type == "user_message" %}
      User: {{ latest_event.content }}
    {% elif event_type == "tool_call" %}
      {{ originator }}: called tool "{{ latest_event.tool_name }}" with arguments {{ latest_event.arguments }}
    {% elif event_type == "assistant_message" %}
      {{ originator }}: {{ latest_event.content }}
    {% elif event_type == "tool_output" %}
      {{ latest_event.tool_name }}: {{ latest_event.content }}
    {% else %}
      Unknown event type: {{ event_type }} from {{ originator }}
    {% endif %}

    {{ ctx.output_format }}

    Response:
  "#
}

test DecideAgentForEventsComplexTest {
  functions [DecideAgentForEvents]
  args {
    events [
      {
        originator "user",
        event "user_message",
        content "I need help with my account settings."
      },
      {
        originator "AccountAgent",
        event "assistant_message",
        content "Sure, I can help with that. What seems to be the issue?"
      },
      {
        originator "user",
        event "user_message",
        content "I can't update my billing information."
      },
      {
        originator "BillingAgent",
        event "tool_call",
        tool_name "UpdateBillingInfo",
        arguments "{\"action\": \"update_billing_info\"}"
      },
      {
        originator "BillingAgent",
        event "tool_output",
        tool_name "UpdateBillingInfo",
        content "Billing information updated successfully."
      },
      {
        originator "BillingAgent",
        event "assistant_message",
        content "Your billing information has been updated successfully. Let me know if you need any more help."
      },
      {
        originator "user",
        event "user_message",
        content "Cheers!"
      },
      {
        originator "GeneralAgent",
        event "assistant_message",
        content "You're welcome! If you have any other questions, feel free to ask."
      },
      {
        originator "user",
        event "user_message",
        content "I have another question about my lost baggage. And I also need to cancel my flight."
      }
    ]
  }
}