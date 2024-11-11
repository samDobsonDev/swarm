from baml_client.sync_client import b as baml
from baml_client.types import MyClass

resume_text = """
      John Doe

      Education
      - University of California, Berkeley
        - B.S. in Computer Science
        - 2020

      Skills
      - Python
      - Java
      - C++
    """

def example(u: str) -> MyClass:
  response = baml.ClassifyMessageWithSymbol(u)
  return response

if __name__ == "__main__":
    print(example(resume_text))
