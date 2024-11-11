generator target {
    // Valid values: "python/pydantic", "typescript", "ruby/sorbet"
    output_type "python/pydantic"
    
    // Where the generated code will be saved (relative to baml_src/)
    output_dir "../"
    
    // What interface you prefer to use for the generated code (sync/async)
    // Both are generated regardless of the choice, just modifies what is exported
    // at the top level
    default_client_mode "sync"
    
    // Version of runtime to generate code for (should match installed baml-py version)
    version "0.67.0"
}

client<llm> GPT4o {
  provider openai
  options {
    model gpt-4o
    api_key env.OPENAI_API_KEY
  }
} 

enum MyClass {
    Refund @alias("k1")
    @description("Customer wants to refund a product")

    CancelOrder @alias("k2")
    @description("Customer wants to cancel an order")

    TechnicalSupport @alias("k3")
    @description("Customer needs help with a technical issue unrelated to account creation or login")

    AccountIssue @alias("k4")
    @description("Specifically relates to account-login or account-creation")

    Question @alias("k5")
    @description("Customer has a question")
}

function ClassifyMessageWithSymbol(input: string) -> MyClass[] {
  client GPT4o

  prompt #"
    Classify the following INPUT into ONE
    of the following categories:

    INPUT: {{ input }}

    {{ ctx.output_format }}

    Response:
  "#
}

test Test1 {
  functions [ClassifyMessageWithSymbol]
  args {
    input "I want to refund my item and cancel my other order"
  }
}