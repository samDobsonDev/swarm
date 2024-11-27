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
  version "0.69.0"
}

client<llm> GPT4o {
  provider openai
  options {
    model gpt-4o
    api_key env.OPENAI_API_KEY
  }
}

enum AgentName {
  GeneralAgent @description("Handles pleasantries and general inquiries that we don't need to route to other agents.")
  StockAlertAgent @description("Handles stock alert requests.")
  OrderAndShipmentAgent @description("Handles order and shipment requests.")
  VerificationAgent @description("Handles verification and email change requests.")
}

class Event {
  originator string
  event string
  tool_name string?
  content string?
  arguments string? // This is a JSON string
}

enum SecurityMetric {
  MaliciousContent
  Spam
  Threat
  InappropriateRequest
  ContextualRelevance
  CompetitorMention
}

class Analysis {
  metrics SecurityMetric[]
  details string
}

class Evaluation {
  agent AgentName
  analysis Analysis
}

function Evaluate(events: Event[]) -> Evaluation {
  client GPT4o

  prompt #"
    You are a chatbot for the retailer Footasylum, designed to assist with all things retail, including order updates, stock alerts, tracking shipments, and more.

    Please evaluate the following events, including the latest event, which is a message from the user, for any security concerns.

    Additionally, based on the following list of events that have occurred in the conversation thus far with the user, decide which ONE agent is best suited to handle the situation. Even if multiple agents are needed, only choose the most relevant one:

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

test EvaluateComplexTest {
  functions [Evaluate]
  args {
    events [
      {
        originator "user",
        event "user_message",
        content "Can you tell me if the Nike Air Max is in stock?"
      },
      {
        originator "StockAlertAgent",
        event "tool_call",
        tool_name "CheckStock",
        arguments "{\"product\": \"Nike Air Max\"}"
      },
      {
        originator "StockAlertAgent",
        event "tool_output",
        tool_name "CheckStock",
        content "Nike Air Max is available in sizes 8, 9, and 10."
      },
      {
        originator "StockAlertAgent",
        event "assistant_message",
        content "The Nike Air Max is available in sizes 8, 9, and 10. Would you like to place an order?"
      },
      {
        originator "user",
        event "user_message",
        content "Yes, please order size 9."
      },
      {
        originator "OrderAndShipmentAgent",
        event "tool_call",
        tool_name "PlaceOrder",
        arguments "{\"product\": \"Nike Air Max\", \"size\": \"9\"}"
      },
      {
        originator "OrderAndShipmentAgent",
        event "tool_output",
        tool_name "PlaceOrder",
        content "Order placed successfully for Nike Air Max size 9."
      },
      {
        originator "user",
        event "user_message",
        content "It would be a shame if something happened to your website's uptime."
      },
    ]
  }
}