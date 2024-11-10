from baml_client.sync_client import b
from baml_client.types import Resume

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

def example(raw_resume: str) -> Resume: 
  response = b.ExtractResume(raw_resume)
  return response

if __name__ == "__main__":
    print(example(resume_text))
