from typing import List

from baml_client.sync_client import b as baml
from baml_client.types import MyClass

resume_text = """
      I want to refund my item. I also want to cancel the order I made last week
    """

def example(u: str) -> List[MyClass]:
  response = baml.ClassifyMessageWithSymbol(u)
  return response

if __name__ == "__main__":
    print(example(resume_text))
