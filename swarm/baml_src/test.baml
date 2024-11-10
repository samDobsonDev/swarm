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

class Resume {
  name string
  education Education[] @description("Extract in the same order listed")
  skills string[] @description("Only include programming languages")
}

class Education {
  school string
  degree string
  year int
}

function ExtractResume(resume_text: string) -> Resume {
  // see clients.baml
  client GPT4o

  // The prompt uses Jinja syntax. Change the models or this text and watch the prompt preview change!
  prompt #"
    Parse the following resume and return a structured representation of the data in the schema below.

    Resume:
    ---
    {{ resume_text }}
    ---

    {# special macro to print the output instructions. #}
    {{ ctx.output_format }}
  "#
}

test Test1 {
  functions [ExtractResume]
  args {
    resume_text #"
      John Doe

      Education
      - University of California, Berkeley
        - B.S. in Computer Science
        - 2020

      Skills
      - Python
      - Java
      - C++
    "#
  }
}