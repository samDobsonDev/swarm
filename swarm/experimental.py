import importlib
import json
import os
import shlex
import argparse
import uuid
import time
import pandas as pd
import qdrant_client
import logging

from typing import Optional, Any, List, Literal, Dict
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client.http import models as rest

TRIAGE_MESSAGE_PROMPT = "Given the following message: {}, select which assistant of the following is best suited to handle it: {}. Respond with JUST the name of the assistant, nothing else"
TRIAGE_SYSTEM_PROMPT = "You are an assistant who triages requests and selects the best assistant to handle that request."
EVAL_GROUNDTRUTH_PROMPT = "Given the following completion: {}, and the expected completion: {}, select whether the completion and expected completion are the same in essence. Correctness does not mean they are the same verbatim, but that the ANSWER is the same. For example: 'The answer, after calculating, is 4' and '4' would be the same. But 'it is 5' and 'the answer is 12' would be different. Respond with ONLY 'true' or 'false'"
EVAL_ASSISTANT_PROMPT = "Given the following assistant name: {}, and the expected assistant name: {}, select whether the assistants are the same. Minor formatting differences, or extra characters are OK, but the words should be the same. Respond with ONLY 'true' or 'false'"
EVAL_PLANNING_PROMPT = "Given the following plan: {}, and the expected plan: {}, select whether the plan and expected plan are the same in essence. Correctness does not mean they are the same verbatim, but that the content is the same with just minor formatting differences. Respond with ONLY 'true' or 'false'"
ITERATE_PROMPT = "Your task to complete is {}. You previously generated the following plan: {}. The steps completed, and the output of those steps, are here: {}. IMPORTANT: Given the outputs of the previous steps, use that to create a revised plan, using the following planning prompt."
EVALUATE_TASK_PROMPT = """Your task was {}. The steps you completed, and the output of those steps, are here: {}. IMPORTANT: Output the following, 'true' or 'false' if you successfully completed the task. Even if your plan changed from original plan, evaluate if the new plan and output
correctly satisfied the given task. Additionally, output a message for the user, explaining whya task was successfully completed, or why it failed. Example:
Task: "Tell a joke about cars. Translate it to Spanish"
Original Plan: [{{tool: "tell_joke", args: {{input: "cars"}}, {{tool: "translate", args: {{language: "Spanish"}}]
Steps Completed: [{{tool: "tell_joke", args: {{input: "cars", output: "Why did the car stop? It ran out of gas!"}}, {{tool: "translate", args: {{language: "Spanish", output: "¿Por qué se detuvo el coche? ¡Se quedó sin gas!"}}]
OUTPUT: ['true','The joke was successfully told and translated to Spanish.']
MAKE SURE THAT OUTPUT IS a list, bracketed by square brackets, with the first element being either 'true' or 'false', and the second element being a string message."""

# IMPORTANT: If you are missing
# any information, or do not have all the required arguments for the tools you are planning, just return your response in double quotes.
# to tell user what information you would need for the request.
#local_engine_vars
LOCAL_PLANNER_PROMPT = """
You are a planner for the Swarm framework.
Your job is to create a properly formatted JSON plan step by step, to satisfy the task given.
Create a list of subtasks based off the [TASK] provided. Your FIRST THOUGHT should be, do I need to call a tool here to answer
or fulfill the user's request. First, think through the steps of the plan necessary. Make sure to carefully look over the tools you are given access to to decide this.
If you are confident that you do not need a tool to respond, either just in conversation or to ask for clarification or more information, respond to the prompt in a concise, but conversational, tone in double quotes. Do not explain that you do not need a tool.
If you DO need tools, create a list of subtasks. Each subtask must be from within the [AVAILABLE TOOLS] list. DO NOT use any tools that are not in the list.
Make sure you have all information needed to call the tools you use in your plan.
Base your decisions on which tools to use from the description and the name and arguments of the tool.
Always output the arguments of the tool, even when arguments is an empty dictionary. MAKE SURE YOU USE ALL REQUIRED ARGUMENTS.
The plan should be as short as possible.

For example:

[AVAILABLE TOOLS]
{{
  "tools": [
    {{
      "type": "function",
      "function": {{
        "name": "lookup_contact_email",
        "description": "Looks up a contact and retrieves their email address",
        "parameters": {{
          "type": "object",
          "properties": {{
            "name": {{
              "type": "string",
              "description": "The name to look up"
            }}
          }},
          "required": ["name"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "email_to",
        "description": "Email the input text to a recipient",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The text to email"
            }},
            "recipient": {{
              "type": "string",
              "description": "The recipient's email address. Multiple addresses may be included if separated by ';'."
            }}
          }},
          "required": ["input", "recipient"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "translate",
        "description": "Translate the input to another language",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The text to translate"
            }},
            "language": {{
              "type": "string",
              "description": "The language to translate to"
            }}
          }},
          "required": ["input", "language"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "summarize",
        "description": "Summarize input text",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The text to summarize"
            }}
          }},
          "required": ["input"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "joke",
        "description": "Generate a funny joke",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The input to generate a joke about"
            }}
          }},
          "required": ["input"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "brainstorm",
        "description": "Brainstorm ideas",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The input to brainstorm about"
            }}
          }},
          "required": ["input"]
        }}
      }}
    }},
    {{
      "type": "function",
      "function": {{
        "name": "poe",
        "description": "Write in the style of author Edgar Allen Poe",
        "parameters": {{
          "type": "object",
          "properties": {{
            "input": {{
              "type": "string",
              "description": "The input to write about"
            }}
          }},
          "required": ["input"]
        }}
      }}
    }}
  ]
}}

[TASK]
"Tell a joke about cars. Translate it to Spanish"

[OUTPUT]
[
    {{"tool": "joke","args":{{"input": "cars"}}}},
    {{"tool": "translate", "args": {{"language": "Spanish"}}
  ]

[TASK]
"Tomorrow is Valentine's day. I need to come up with a few date ideas. She likes Edgar Allen Poe so write using his style. E-mail these ideas to my significant other. Translate it to French."

[OUTPUT]
[{{"tool": "brainstorm","args":{{"input": "Valentine's Day Date Ideas"}}}},
    {{"tool": "poe", "args": {{}}}},
    {{"tool": "email_to", "args": {{"recipient": "significant_other@example.com"}},
    {{"tool": "translate", "args": {{"language": "French"}}]

[AVAILABLE TOOLS]
{tools}

[TASK]
{task}

[OUTPUT]
"""

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = '\033[90m'

test_root = 'tests'
test_file = 'test_prompts.jsonl'
tasks_path = 'swarm_tasks.json'

# Options are 'assistants' or 'local'
engine_name = 'local'
max_iterations = 5
persist = False

class Task:
    def __init__(self, description, iterate=False, evaluate=False, assistant='user_interface'):
        self.id = str(uuid.uuid4())
        self.description = description
        self.assistant = assistant
        self.iterate: bool = iterate
        self.evaluate: bool = evaluate


class EvaluationTask(Task):
    def __init__(self, description, assistant,iterate, evaluate, groundtruth, expected_assistant, eval_function, expected_plan):
        super().__init__(description=description, assistant=assistant,iterate=iterate, evaluate=evaluate)
        self.groundtruth = groundtruth
        self.expected_assistant = expected_assistant
        self.expected_plan = expected_plan
        self.eval_function = eval_function


def get_completion(
    client,
    messages: list[dict[str, str]],
    model: str = "gpt-4-0125-preview",
    max_tokens=2000,
    temperature=0.7,
    tools=None,
    stream=False
):
    # Prepare the request parameters
    request_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools and isinstance(tools, list):
        request_params["tools"] = tools  # Tools are already in dictionary format

    # Make the API call with the possibility of streaming
    if stream:
        completion = client.chat.completions.create(**request_params)
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        for chunk in completion:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk.choices[0].delta.content  # extract the message
            collected_messages.append(chunk_message)  # save the message
            print(chunk_message, end="")  # print the message
            # yield chunk_message  # Yield each part of the completion as it arrives
        return collected_messages  # Returns the whole completion
    else:
        completion = client.chat.completions.create(**request_params)
        return completion.choices[0].message  # Returns the whole completion

def is_dict_empty(d):
    return all(not v for v in d.values())

class Assistant(BaseModel):
    log_flag: bool
    name: Optional[str] = None
    instance: Optional[Any] = None
    tools: Optional[list] = None
    current_task_id: str = None
    sub_assistants: Optional[list] = None
    runs: list = []
    context: Optional[dict] = {}
    planner: str = 'sequential' #default to sequential

    def initialize_history(self):
        self.context['history'] = []

    def add_user_message(self, message):
        self.context['history'].append({'task_id':self.current_task_id,'role':'user','content':message})

    def add_assistant_message(self, message):
        self.context['history'].append({'task_id':self.current_task_id,'role':'assistant','content':message})

    def add_tool_message(self, message):
        self.context['history'].append({'task_id':self.current_task_id,'role':'user','tool':message})

    def print_conversation(self):
        print(f"\n{Colors.GREY}Conversation with Assistant: {self.name}{Colors.ENDC}\n")
        # Group messages by run_id
        messages_by_task_id = {}
        for message in self.context['history']:
            task_id = message['task_id']
            if task_id not in messages_by_task_id:
                messages_by_task_id[task_id] = []
            messages_by_task_id[task_id].append(message)
        # Print messages for each run_id
        for task_id, messages in messages_by_task_id.items():
            print(f"{Colors.OKCYAN}Task ID: {task_id}{Colors.ENDC}")
            for message in messages:
                if 'role' in message and message['role'] == 'user':
                    print(f"{Colors.OKBLUE}User:{Colors.ENDC} {message['content']}")
                elif 'tool' in message:
                    tool_message = message['tool']
                    tool_args = ', '.join([f"{arg}: {value}" for arg, value in tool_message['args'].items()])
                    print(f"{Colors.OKGREEN}Tool:{Colors.ENDC} {tool_message['tool']}({tool_args})")
                elif 'role' in message and message['role'] == 'assistant':
                    print(f"{Colors.HEADER}Assistant:{Colors.ENDC} {message['content']}")
            print("\n")

    @staticmethod
    def evaluate(client, task, plan_log):
        """Evaluates the assistant's performance on a task"""
        output = get_completion(client, [{'role': 'user', 'content': EVALUATE_TASK_PROMPT.format(task.description, plan_log)}])
        output.content = output.content.replace("'",'"')
        try:
            return json.loads(output.content)
        except json.JSONDecodeError:
            print("An error occurred while decoding the JSON.")
            return None

    def save_conversation(self,test=False):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if not test:
            filename = f'logs/session_{timestamp}.json'
        else:
            filename = f'tests/test_runs/test_{timestamp}.json'
        with open(filename, 'w') as file:
            json.dump(self.context['history'], file)

    def pass_context(self,assistant):
        """Passes the context of the conversation to the assistant"""
        assistant.context['history'] = self.context['history']

class Parameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = Field(None, alias='choices')

class FunctionParameters(BaseModel):
    type: Literal['object']  # Ensuring it's always 'object'
    properties: Dict[str, Parameter] = {}
    required: Optional[List[str]] = None

class FunctionTool(BaseModel):
    name: str
    description: Optional[str]
    parameters: FunctionParameters

class Tool(BaseModel):
    type: str
    function: Optional[FunctionTool]
    human_input: Optional[bool] = False

class AssistantsEngine:
    def __init__(self,client,tasks):
        self.client = client
        self.assistants = []
        self.tasks = tasks
        self.thread = self.initialize_thread()

    def initialize_thread(self):
        # Create a Thread for the user's conversation
        thread = self.client.beta.threads.create()
        return thread

    def reset_thread(self):
        # Create a Thread for the user's conversation
        self.thread = self.client.beta.threads.create()

    def load_all_assistants(self):
        base_path = 'assistants'
        tools_base_path = 'tools'
        # Load individual tool definitions from the tools directory
        tool_defs = {}
        for tool_dir in os.listdir(tools_base_path):
            if '__pycache__' in tool_dir:
                continue
            tool_dir_path = os.path.join(tools_base_path, tool_dir)
            if os.path.isdir(tool_dir_path):
                tool_json_path = os.path.join(tool_dir_path, 'tool.json')
                if os.path.isfile(tool_json_path):
                    with open(tool_json_path, 'r') as file:
                        # Assuming the JSON file contains a list of tool definitions
                        tool_def = json.load(file)
                        tool_defs[tool_def['function']['name']] = tool_def['function']
        # Load assistants and their tools
        for assistant_dir in os.listdir(base_path):
            if '__pycache__' in assistant_dir:
                continue
            assistant_config_path = os.path.join(base_path, assistant_dir, "assistant.json")
            if os.path.exists(assistant_config_path):
                with open(assistant_config_path, "r") as file:
                    assistant_config = json.load(file)[0]
                    assistant_name = assistant_config.get('name', assistant_dir)
                    log_flag = assistant_config.pop('log_flag', False)
                    # List of tool names from the assistant's config
                    assistant_tools_names = assistant_config.get('tools', [])
                    # Build the list of tool definitions for this assistant
                    assistant_tools = [tool_defs[name] for name in assistant_tools_names if name in tool_defs]
                    # Create or update the assistant instance
                    existing_assistants = self.client.beta.assistants.list()
                    loaded_assistant = next((a for a in existing_assistants if a.name == assistant_name), None)
                    if loaded_assistant:
                        assistant_tools = [{'type': 'function', 'function': tool_defs[name]} for name in assistant_tools_names if name in tool_defs]
                        assistant_config['tools'] = assistant_tools
                        assistant_config['name']=assistant_name
                        loaded_assistant = self.client.beta.assistants.create(**assistant_config)
                        print(f"Assistant '{assistant_name}' created.\n")
                    asst_object = Assistant(name=assistant_name, log_flag=log_flag, instance=loaded_assistant, tools=assistant_tools)
                    self.assistants.append(asst_object)


    def initialize_and_display_assistants(self):
            """
            Loads all assistants and displays their information.
            """
            self.load_all_assistants()
            for asst in self.assistants:
                print(f'\n{Colors.HEADER}Initializing assistant:{Colors.ENDC}')
                print(f'{Colors.OKBLUE}Assistant name:{Colors.ENDC} {Colors.BOLD}{asst.name}{Colors.ENDC}')
                if asst.instance and hasattr(asst.instance, 'tools'):
                    print(f'{Colors.OKGREEN}Tools:{Colors.ENDC} {asst.instance.tools} \n')
                else:
                    print(f"{Colors.OKGREEN}Tools:{Colors.ENDC} Not available \n")


    def get_assistant(self, assistant_name):
        for assistant in self.assistants:
            if assistant.name == assistant_name:
                return assistant
        print('No assistant found')
        return None

    def triage_request(self, message, test_mode):
        """
        Analyze the user message and delegate it to the appropriate assistant.
        """
        #determine the appropriate assistant for the message
        assistant_name = self.determine_appropriate_assistant(message)
        assistant = self.get_assistant(assistant_name)
        if assistant:
            print(
            f"{Colors.OKGREEN}\nSelected Assistant:{Colors.ENDC} {Colors.BOLD}{assistant.name}{Colors.ENDC}"
            )
            assistant.add_assistant_message('Selected Assistant: '+assistant.name)
            return assistant
        #else
        if not test_mode:
            print('No assistant found')
        return None


    def determine_appropriate_assistant(self, message):
        triage_message = [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}, {
            "role": "user",
            "content": TRIAGE_MESSAGE_PROMPT.format(message, [asst.instance for asst in self.assistants]),
        }]
        response = get_completion(self.client, triage_message)
        return response.content


    def run_request(self, request, assistant,test_mode):
        """
        Run the request with the selected assistant and monitor its status.
        """
        # Add message to thread
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=request
        )
        # Initialize run
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=assistant.instance.id
        )
        # Monitor the run status in a loop
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            if run.status in ["queued", "in_progress"]:
                time.sleep(2)  # Wait before checking the status again
                if not test_mode:
                    print('waiting for run')
            elif run.status == "requires_action":
                tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
                self.handle_tool_call(tool_call, run)
                # Re-submitting the tool outputs and continue the loop
            elif run.status in ["completed","expired", "cancelling", "cancelled", "failed"]:
                if not test_mode:
                    print(f'\nrun {run.status}')
                break
        if assistant.log_flag:
            self.store_messages()
        # Retrieve and return the response (only if completed)
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        assistant_response = next((msg for msg in messages.data if msg.role == 'assistant' and msg.content), None)
        if assistant_response:
            assistant_response_text = assistant_response.content[0].text.value
            if not test_mode:
                print(f"{Colors.RED}Response:{Colors.ENDC} {assistant_response_text}", "\n")
            return assistant_response_text
        return "No response from the assistant."


    def handle_tool_call(self, tool_call, run):
        tool_name = tool_call.function.name
        tool_dir = os.path.join(os.getcwd(), 'tools', tool_name)
        handler_path = os.path.join(tool_dir, 'handler.py')
        # Dynamically import the handler function from the handler.py file
        if os.path.isfile(handler_path):
            spec = importlib.util.spec_from_file_location(f"{tool_name}_handler", handler_path)
            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)
            tool_handler = getattr(tool_module, tool_name+ '_assistants')
            # Prepare the arguments for the handler function
            handler_args = {'tool_id': tool_call.id}
            tool_args = json.loads(tool_call.function.arguments)
            for arg_name, arg_value in tool_args.items():
                if arg_value is not None:
                    handler_args[arg_name] = arg_value
            # Call the handler function with arguments
            print(f"{Colors.HEADER}Running Tool:{Colors.ENDC} {tool_name}")
            print(handler_args)
            tool_response = tool_handler(**handler_args)
            # Submit the tool response back to the thread
            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=run.id,
                tool_outputs=[
                    {
                        "tool_call_id": tool_call.id,
                        "output": json.dumps({"result": tool_response}),
                    }
                ],
            )
        else:
            print(f"No handler found for tool {tool_name}")

    def store_messages(self, filename="threads/thread_data.json"):
        thread = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        # Extract the required fields from each message in the thread
        messages = []
        for message in thread.data:
            role = message.role
            run_id = message.run_id
            assistant_id = message.assistant_id
            thread_id = message.thread_id
            created_at = message.created_at
            content_value = message.content[0].text.value
            messages.append({
                'role': role,
                'run_id': run_id,
                'assistant_id': assistant_id,
                'thread_id': thread_id,
                'created_at': created_at,
                'content': content_value
            })
        try:
            with open(filename, 'r') as file:
                existing_threads = json.load(file)
        except:
            existing_threads = []
        # Convert the OpenAI object to a serializable format (e.g., a dictionary)
        # Append new threads
        existing_threads.append(messages)
        # Save back to the file
        try:
            with open(filename, 'w') as file:
                json.dump(existing_threads, file, indent=4)
        except Exception as e:
            print(f"Error while saving to file: {e}")


    def run_task(self, task,test_mode):
        """
        Processes a given task. If the assistant is set to 'auto', it determines the appropriate
        assistant using triage_request. Otherwise, it uses the specified assistant.
        """
        if not test_mode:
            print(
        f"{Colors.OKCYAN}User Query:{Colors.ENDC} {Colors.BOLD}{task.description}{Colors.ENDC}"
            )
        else:
            print(
        f"{Colors.OKCYAN}Test:{Colors.ENDC} {Colors.BOLD}{task.description}{Colors.ENDC}"
            )
        if task.assistant == 'auto':
            # Triage the request to determine the appropriate assistant
            assistant = self.triage_request(task.description,test_mode)
        else:
            # Fetch the specified assistant
            assistant = self.get_assistant(task.assistant)
            print(
            f"{Colors.OKGREEN}\nSelected Assistant:{Colors.ENDC} {Colors.BOLD}{assistant.name}{Colors.ENDC}"
            )
        if test_mode:
            task.assistant = assistant.name if assistant else "None"
        if not assistant:
            if not test_mode:
                print(f"No suitable assistant found for the task: {task.description}")
            return None
        # Run the request with the determined or specified assistant
        self.reset_thread()
        return self.run_request(task.description, assistant,test_mode)

    def deploy(self, client,test_mode=False,test_file_path=None):
        """
        Processes all tasks in the order they are listed in self.tasks.
        """
        #Initialize swarm first
        self.client = client
        if test_mode and test_file_path:
            print("\nTesting the swarm\n\n")
            self.load_test_tasks(test_file_path)
        else:
            print("\n🐝🐝🐝 Deploying the swarm 🐝🐝🐝\n\n")
        self.initialize_and_display_assistants()
        total_tests = 0
        groundtruth_tests = 0
        assistant_tests = 0
        for task in self.tasks:
            output = self.run_task(task,test_mode)
            if test_mode and hasattr(task, 'groundtruth'):
                total_tests += 1
                response = get_completion(self.client,[{"role":"user","content":EVALUATE_TASK_PROMPT.format(output,task.groundtruth)}])
                if response.content=='True':
                    groundtruth_tests += 1
                    print(f"{Colors.OKGREEN}✔ Groundtruth test passed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.groundtruth}{Colors.OKBLUE}, Got: {Colors.ENDC}{output}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}✘ Test failed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.groundtruth}{Colors.OKBLUE}, Got: {Colors.ENDC}{output}{Colors.ENDC}")
                if task.assistant==task.expected_assistant:
                    print(f"{Colors.OKGREEN}✔ Correct assistant assigned for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
                    assistant_tests += 1
                else:
                    print(f"{Colors.RED}✘ Incorrect assistant assigned for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
        if test_mode:
            print(f"\n{Colors.OKGREEN}Passed {groundtruth_tests} groundtruth tests out of {total_tests} tests. Success rate: {groundtruth_tests/total_tests*100}%{Colors.ENDC}\n")
            print(f"{Colors.OKGREEN}Passed {assistant_tests} assistant tests out of {total_tests} tests. Success rate: {groundtruth_tests/total_tests*100}%{Colors.ENDC}\n")
            print("Completed testing the swarm\n\n")
        else:
            print("🍯🐝🍯 Swarm operations complete 🍯🐝🍯\n\n")



    def load_test_tasks(self, test_file_path):
        self.tasks = []  # Clear any existing tasks
        with open(test_file_path, 'r') as file:
            for line in file:
                test_case = json.loads(line)
                task = EvaluationTask(description=test_case['text'],
                            assistant=test_case.get('assistant', 'auto'),
                            groundtruth=test_case['groundtruth'],
                            expected_assistant=test_case['expected_assistant'])
                self.tasks.append(task)

class Run:
    def __init__(self,assistant,request,client):
        self.assistant = assistant
        self.request = request
        self.client = client
        self.status = None
        self.response = None

    def initiate(self, planner):
        self.status = 'in_progress'
        if planner=='sequential':
            plan = self.generate_plan()
            return plan

    def generate_plan(self,task=None):
        if not task:
            task = self.request
        completion = get_completion(self.client,[{'role':'user','content':LOCAL_PLANNER_PROMPT.format(tools=self.assistant.tools,task=task)}])
        response_string = completion.content
        #Parse out just list in case
        try: # see if plan
            start_pos = response_string.find('[')
            end_pos = response_string.rfind(']')
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                response_truncated = response_string[start_pos:end_pos+1]
                response_formatted = json.loads(response_truncated)
                return response_formatted
            else:
                try:
                    response_formatted = json.loads(response_string)
                    return response_formatted
                except:
                    return "Response not in correct format"
        except:
            return response_string

class LocalEngine:
    def __init__(self, client, tasks, persist = False):
        self.client = client
        self.assistants = []
        self.last_assistant = None
        self.persist = persist
        self.tasks = tasks
        self.tool_functions = []
        self.global_context = {}

    def load_tools(self):
        tools_path = 'tools'
        self.tool_functions = []
        for tool_dir in os.listdir(tools_path):
            dir_path = os.path.join(tools_path, tool_dir)
            if os.path.isdir(dir_path):
                for tool_name in os.listdir(dir_path):
                    if tool_name.endswith('.json'):
                        with open(os.path.join(dir_path, tool_name), 'r') as file:
                            try:
                                tool_def = json.load(file)
                                tool = Tool(type=tool_def['type'], function=tool_def['function'], human_input=tool_def.get('human_input', False))
                                self.tool_functions.append(tool)
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON for tool {tool_name}: {e}")

    def load_all_assistants(self):
        base_path = 'assistants'
        self.load_tools()
        # tool_defs = {tool.function.name: tool.function.dict() for tool in self.tool_functions}
        for assistant_dir in os.listdir(base_path):
            if '__pycache__' in assistant_dir:
                continue
            assistant_config_path = os.path.join(base_path, assistant_dir, "assistant.json")
            if os.path.exists(assistant_config_path):
                try:
                    with open(assistant_config_path, "r") as file:
                        assistant_config = json.load(file)[0]
                        assistant_tools_names = assistant_config.get('tools', [])
                        assistant_name = assistant_config.get('name', assistant_dir)
                        assistant_tools = [tool for tool in self.tool_functions if tool.function.name in assistant_tools_names]
                        log_flag = assistant_config.pop('log_flag', False)
                        sub_assistants = assistant_config.get('assistants', None)
                        planner = assistant_config.get('planner', 'sequential') #default is sequential
                        print(f"Assistant '{assistant_name}' created.\n")
                        asst_object = Assistant(name=assistant_name, log_flag=log_flag, instance=None, tools=assistant_tools, sub_assistants=sub_assistants, planner=planner)
                        asst_object.initialize_history()
                        self.assistants.append(asst_object)
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error loading assistant configuration from {assistant_config_path}: {e}")


    def initialize_and_display_assistants(self):
            """
            Loads all assistants and displays their information.
            """
            self.load_all_assistants()
            self.initialize_global_history()
            for asst in self.assistants:
                print(f'\n{Colors.HEADER}Initializing assistant:{Colors.ENDC}')
                print(f'{Colors.OKBLUE}Assistant name:{Colors.ENDC} {Colors.BOLD}{asst.name}{Colors.ENDC}')
                if asst.tools:
                    print(f'{Colors.OKGREEN}Tools:{Colors.ENDC} {[tool.function.name for tool in asst.tools]} \n')
                else:
                    print(f"{Colors.OKGREEN}Tools:{Colors.ENDC} No tools \n")


    def get_assistant(self, assistant_name):
        for assistant in self.assistants:
            if assistant.name == assistant_name:
                return assistant
        print('No assistant found')
        return None

    def triage_request(self, assistant, message):
        """
        Analyze the user message and delegate it to the appropriate assistant.
        """
        assistant_name = None
        # Determine the appropriate assistant for the message
        if assistant.sub_assistants is not None:
            assistant_name = self.determine_appropriate_assistant(assistant, message)
            if not assistant_name:
                print('No appropriate assistant determined')
                return None
            assistant_new = self.get_assistant(assistant_name)
            if not assistant_new:
                print(f'No assistant found with name: {assistant_name}')
                return None
            assistant.pass_context(assistant_new)
            # Pass along context: if the assistant is a sub-assistant, pass along the context of the parent assistant
        else:
            assistant_new = assistant
        # If it's a new assistant, so a sub assistant
        if assistant_name and assistant_name != assistant.name:
            print(
                f"{Colors.OKGREEN}Selecting sub-assistant:{Colors.ENDC} {Colors.BOLD}{assistant_new.name}{Colors.ENDC}"
            )
            assistant.add_assistant_message(f"Selecting sub-assistant: {assistant_new.name}")
        else:
            print(
                f"{Colors.OKGREEN}Assistant:{Colors.ENDC} {Colors.BOLD}{assistant_new.name}{Colors.ENDC}"
            )
        return assistant_new


    def determine_appropriate_assistant(self, assistant, message):
        triage_message = [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}, {
            "role": "user",
            "content": TRIAGE_MESSAGE_PROMPT.format(
                message,
                [(asst.name, asst.tools) for asst in
                 [assistant] + [asst for asst in self.assistants if asst.name in assistant.sub_assistants]]),
        }]
        response = get_completion(self.client, triage_message)
        return response.content

    def initiate_run(self, task, assistant,test_mode):
        """
        Run the request with the selected assistant and monitor its status.
        """
        run = Run(assistant, task.description, self.client)
        #Update assistant with current task and run
        assistant.current_task_id = task.id
        assistant.runs.append(run)
        #Get planner
        planner = assistant.planner
        plan = run.initiate(planner)
        plan_log = {'step': [], 'step_output': []}
        if not isinstance(plan, list):
            plan_log['step'].append('response')
            plan_log['step'].append(plan)
            assistant.add_assistant_message(f"Response to user: {plan}")
            print(f"{Colors.HEADER}Response:{Colors.ENDC} {plan}")
            #add global context
            self.store_context_globally(assistant)
            return plan_log, plan_log
        original_plan = plan.copy()
        iterations = 0
        while plan and iterations< max_iterations:
            if isinstance(plan,list):
              step = plan.pop(0)
            else:
                return "Error generating plan", "Error generating plan"
            assistant.add_tool_message(step)
            human_input_flag = next((tool.human_input for tool in assistant.tools if tool.function.name == step['tool']), False)
            if step['tool']:
                print(f"{Colors.HEADER}Running Tool:{Colors.ENDC} {step['tool']}")
                if human_input_flag:
                    print(f"\n{Colors.HEADER}Tool {step['tool']} requires human input:{Colors.HEADER}")
                    print(f"{Colors.GREY}Tool arguments:{Colors.ENDC} {step['args']}\n")
                    user_confirmation = input(f"Type 'yes' to execute tool, anything else to skip: ")
                    if user_confirmation.lower() != 'yes':
                        assistant.add_assistant_message(f"Tool {step['tool']} execution skipped by user.")
                        print(f"{Colors.GREY}Skipping tool execution.{Colors.ENDC}")
                        plan_log['step'].append('tool_skipped')
                        plan_log['step_output'].append(f'Tool {step["tool"]} execution skipped by user! Task not completed.')
                        continue
                    assistant.add_assistant_message(f"Tool {step['tool']} execution approved by user.")
            tool_output = self.handle_tool_call(step)
            plan_log['step'].append(step)
            plan_log['step_output'].append(tool_output)
            if task.iterate and not is_dict_empty(plan_log) and plan:
               iterations += 1
               new_task = ITERATE_PROMPT.format(task.description, original_plan, plan_log)
               plan = run.generate_plan(new_task)
            # Store the output for the next iteration
            self.store_context_globally(assistant)
        return original_plan, plan_log

    @staticmethod
    def handle_tool_call(tool_call):
        tool_name = tool_call['tool']
        tool_dir = os.path.join(os.getcwd(), 'tools', tool_name)
        handler_path = os.path.join(tool_dir, 'handler.py')
        # Dynamically import the handler function from the handler.py file
        if os.path.isfile(handler_path):
            spec = importlib.util.spec_from_file_location(f"{tool_name}_handler", handler_path)
            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)
            tool_handler = getattr(tool_module, tool_name)
            # Call the handler function with arguments
            try:
                tool_response = tool_handler(**tool_call['args'])
            except:
                return 'Failed to execute tool'
            try:
                # assistant.add_assistant_message(tool_response.content)
                return tool_response.content
            except:
                # assistant.add_assistant_message(tool_response)
                return tool_response
        print('No tool file found')
        return 'No tool file found'

    def run_task(self, task, test_mode):
            """
            Processes a given task.
            """
            if not test_mode:
                print(
            f"{Colors.OKCYAN}User Query:{Colors.ENDC} {Colors.BOLD}{task.description}{Colors.ENDC}"
                )
            else:
                print(
            f"{Colors.OKCYAN}Test:{Colors.ENDC} {Colors.BOLD}{task.description}{Colors.ENDC}"
                )
            #Maintain assistant if persist flag is true
            if self.persist and self.last_assistant is not None:
                assistant = self.last_assistant
            else:
                assistant = self.get_assistant(task.assistant)
                assistant.current_task_id = task.id
                assistant.add_user_message(task.description)
            #triage based on current assistant
            selected_assistant = self.triage_request(assistant, task.description)
            if test_mode:
                task.assistant = selected_assistant.name if selected_assistant else "None"
            if not selected_assistant:
                if not test_mode:
                    print(f"No suitable assistant found for the task: {task.description}")
                return None
            # Run the request with the determined or specified assistant
            original_plan, plan_log = self.initiate_run(task, selected_assistant,test_mode)
            #set last assistant
            self.last_assistant = selected_assistant
            #if evaluating the task
            if task.evaluate:
                output = assistant.evaluate(self.client,task, plan_log)
                if output is not None:
                    success_flag = False
                    if not isinstance(output[0],bool):
                     success_flag = False if output[0].lower() == 'false' else bool(output[0])
                    message = output[1]
                    if success_flag:
                        print(f'\n\033[93m{message}\033[0m')
                    else:
                        print(f"{Colors.RED}{message}{Colors.ENDC}")
                    #log
                    assistant.add_assistant_message(message)
                else:
                    message = "Error evaluating output"
                    print(f"{Colors.RED}{message}{Colors.ENDC}")
                    assistant.add_assistant_message(message)
            return original_plan, plan_log


    def run_tests(self):
        total_groundtruth = 0
        total_planning = 0
        total_assistant = 0
        groundtruth_pass = 0
        planning_pass = 0
        assistant_pass = 0
        for task in self.tasks:
            original_plan, plan_log = self.run_task(task, test_mode=True)
            if task.groundtruth:
                total_groundtruth += 1
                # Assuming get_completion returns a response object with a content attribute
                response = get_completion(self.client, [{"role": "user", "content": EVAL_GROUNDTRUTH_PROMPT.format(original_plan, task.groundtruth)}])
                if response.content.lower() == 'true':
                    groundtruth_pass += 1
                    print(f"{Colors.OKGREEN}✔ Groundtruth test passed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.groundtruth}{Colors.OKBLUE}, Got: {Colors.ENDC}{original_plan}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}✘ Test failed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.groundtruth}{Colors.OKBLUE}, Got: {Colors.ENDC}{original_plan}{Colors.ENDC}")
                total_assistant += 1
                if task.assistant == task.expected_assistant:
                    assistant_pass += 1
                    print(f"{Colors.OKGREEN}✔ Correct assistant assigned. {Colors.ENDC}{Colors.OKBLUE} Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
                else:
                    print(f"{Colors.RED}✘ Incorrect assistant assigned. {Colors.ENDC}{Colors.OKBLUE} Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
            elif task.expected_plan:
                total_planning += 1
                # Assuming get_completion returns a response object with a content attribute
                response = get_completion(self.client, [{"role": "user", "content": EVAL_PLANNING_PROMPT.format(original_plan, task.expected_plan)}])
                if response.content.lower() == 'true':
                    planning_pass += 1
                    print(f"{Colors.OKGREEN}✔ Planning test passed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_plan}{Colors.OKBLUE}, Got: {Colors.ENDC}{original_plan}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}✘ Test failed for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_plan}{Colors.OKBLUE}, Got: {Colors.ENDC}{original_plan}{Colors.ENDC}")
                total_assistant += 1
                if task.assistant == task.expected_assistant:
                    assistant_pass += 1
                    print(f"{Colors.OKGREEN}✔ Correct assistant assigned.  {Colors.ENDC}{Colors.OKBLUE}Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
                else:
                    print(f"{Colors.RED}✘ Incorrect assistant assigned for. {Colors.ENDC}{Colors.OKBLUE} Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
            else:
                total_assistant += 1
                if task.assistant == task.expected_assistant:
                    assistant_pass += 1
                    print(f"{Colors.OKGREEN}✔ Correct assistant assigned for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
                else:
                    print(f"{Colors.RED}✘ Incorrect assistant assigned for: {Colors.ENDC}{task.description}{Colors.OKBLUE}. Expected: {Colors.ENDC}{task.expected_assistant}{Colors.OKBLUE}, Got: {Colors.ENDC}{task.assistant}{Colors.ENDC}\n")
        if total_groundtruth > 0:
            print(f"\n{Colors.OKGREEN}Passed {groundtruth_pass} groundtruth tests out of {total_groundtruth} tests. Success rate: {groundtruth_pass / total_groundtruth * 100}%{Colors.ENDC}\n")
        if total_planning > 0:
            print(f"{Colors.OKGREEN}Passed {planning_pass} planning tests out of {total_planning} tests. Success rate: {planning_pass / total_planning * 100}%{Colors.ENDC}\n")
        if total_assistant > 0:
            print(f"{Colors.OKGREEN}Passed {assistant_pass} assistant tests out of {total_assistant} tests. Success rate: {assistant_pass / total_assistant * 100}%{Colors.ENDC}\n")
        print("Completed testing the swarm\n\n")

    def deploy(self, client, test_mode=False, test_file_path=None):
        """
        Processes all tasks in the order they are listed in self.tasks.
        """
        self.client = client
        if test_mode and test_file_path:
            print("\nTesting the swarm\n\n")
            self.load_test_tasks(test_file_path)
            self.initialize_and_display_assistants()
            self.run_tests()
            for assistant in self.assistants:
                if assistant.name == 'user_interface':
                    assistant.save_conversation(test=True)
        else:
            print("\n🐝🐝🐝 Deploying the swarm 🐝🐝🐝\n\n")
            self.initialize_and_display_assistants()
            print("\n" + "-" * 100 + "\n")
            for task in self.tasks:
                print('Task',task.id)
                print(f"{Colors.BOLD}Running task{Colors.ENDC}")
                self.run_task(task, test_mode)
                print("\n" + "-" * 100 + "\n")
            #save the session
            for assistant in self.assistants:
                if assistant.name == 'user_interface':
                    assistant.save_conversation()
             #assistant.print_conversation()

    def load_test_tasks(self, test_file_paths):
        self.tasks = []  # Clear any existing tasks
        for f in test_file_paths:
            with open(f, 'r') as file:
                for line in file:
                    test_case = json.loads(line)
                    task = EvaluationTask(description=test_case['text'],
                                assistant=test_case.get('assistant', 'user_interface'),
                                groundtruth=test_case.get('groundtruth',None),
                                expected_plan=test_case.get('expected_plan',None),
                                expected_assistant=test_case['expected_assistant'],
                                iterate=test_case.get('iterate', False),  # Add this
                                evaluate=test_case.get('evaluate', False),
                                eval_function=test_case.get('eval_function', 'default')
                                )
                    self.tasks.append(task)

    def store_context_globally(self, assistant):
        self.global_context['history'].append({assistant.name:assistant.context['history']})

    def initialize_global_history(self):
        self.global_context['history'] = []

class Swarm:
    def __init__(self, engine_name, tasks=[], persist=False):
        self.tasks = tasks
        self.engine_name = engine_name
        self.engine = None
        self.persist = persist

    def deploy(self, test_mode=False, test_file_paths=None):
        """
        Processes all tasks in the order they are listed in self.tasks.
        """
        client = OpenAI()
        # Initialize swarm first
        if self.engine_name == 'assistants':
            print(f"{Colors.GREY}Selected engine: Assistants{Colors.ENDC}")
            self.engine = AssistantsEngine(client, self.tasks)
            self.engine.deploy(client, test_mode, test_file_paths)
        elif self.engine_name == 'local':
            print(f"{Colors.GREY}Selected engine: Local{Colors.ENDC}")
            self.engine = LocalEngine(client, self.tasks, persist=self.persist)
            self.engine.deploy(client, test_mode, test_file_paths)

    def load_tasks(self):
        self.tasks = []
        with open(tasks_path, 'r') as file:
            tasks_data = json.load(file)
            for task_json in tasks_data:
                task = Task(description=task_json['description'],
                            iterate=task_json.get('iterate', False),
                            evaluate=task_json.get('evaluate', False),
                            assistant=task_json.get('assistant', 'user_interface'))
                self.tasks.append(task)

    def add_task(self, task):
        self.tasks.append(task)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["local", "assistants"], default="local", help="Choose the engine to use.")
    parser.add_argument("--test", nargs='*', help="Run the tests.")
    parser.add_argument("--create-task", type=str, help="Create a new task with the given description.")
    parser.add_argument("task_description", type=str, nargs="?", default="", help="Description of the task to create.")
    parser.add_argument("--assistant", type=str, help="Specify the assistant for the new task.")
    parser.add_argument("--evaluate", action="store_true", help="Set the evaluate flag for the new task.")
    parser.add_argument("--iterate", action="store_true", help="Set the iterate flag for the new task.")
    parser.add_argument("--input", action="store_true", help="If we want CLI")
    return parser.parse_args()

def validate_tool(tool_definition):
    # Validate the tool using its schema
    Tool(**tool_definition)  # Uncomment if you have a schema to validate tools
    print(f"Validating tool: {tool_definition['function']['name']}")

def validate_all_tools():
    tools_path = os.path.join(os.getcwd(), 'tools')
    for tool_dir in os.listdir(tools_path):
        if '__pycache__' in tool_dir:
            continue
        tool_dir_path = os.path.join(tools_path, tool_dir)
        if os.path.isdir(tool_dir_path):
            # Validate tool.json
            tool_json_path = os.path.join(tool_dir_path, 'tool.json')
            handler_path = os.path.join(tool_dir_path, 'handler.py')
            if os.path.isfile(tool_json_path) and os.path.isfile(handler_path):
                with open(tool_json_path, 'r') as file:
                    tool_def = json.load(file)
                    tool_name_from_json = tool_def['function']['name']
                    # Check if the folder name matches the tool name in tool.json
                    if tool_name_from_json != tool_dir:
                        print(f"Mismatch in tool folder name and tool name in JSON for {tool_dir}")
                    else:
                        print(f"{tool_dir}/tool.json tool name matches folder name.")
                    # Check if the function name in handler.py matches the tool name
                    spec = importlib.util.spec_from_file_location(f"{tool_dir}_handler", handler_path)
                    tool_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(tool_module)
                    # Verify if the function exists in handler.py and matches the name
                    if hasattr(tool_module, tool_dir):
                        print(f"{tool_dir}/handler.py contains a matching function name.")
                    else:
                        print(f"{tool_dir}/handler.py does not contain a function '{tool_dir}'.")
            else:
                if not os.path.isfile(tool_json_path):
                    print(f"Missing tool.json in {tool_dir} tool folder.")
                if not os.path.isfile(handler_path):
                    print(f"Missing handler.py in {tool_dir} tool folder.")
    print('\n')

def validate_all_assistants():
    assistants_path = os.path.join(os.getcwd(), 'assistants')
    for root, dirs, files in os.walk(assistants_path):
        for file in files:
            if file.endswith('assistant.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file:
                        assistant_data = json.load(file)[0]  # Access the first dictionary in the list
                        try:
                            Assistant(**assistant_data)
                            print(f"{os.path.basename(root)} assistant validated!")
                        except:
                            Assistant(**assistant_data)
                            print(f"Assistant validation failed!")
    print('\n')

def prep_data(generate_new_embeddings=False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    client = OpenAI()
    embedding_model = "text-embedding-3-large"
    article_list = os.listdir('data')
    articles = []
    logger.info("Starting to process articles...")
    embedding_dir = 'embeddings' # Directory to store and retrieve embeddings
    os.makedirs(embedding_dir, exist_ok=True)
    for x in article_list:
        article_path = 'data/' + x
        with open(article_path) as f:
            data = json.load(f)
            articles.append(data)
            logger.info(f"Loaded article: {x}")
    for i, x in enumerate(articles):
        embedding_file = os.path.join(embedding_dir, f"{x['title']}.json")
        if generate_new_embeddings or not os.path.exists(embedding_file):
            try:
                embedding = client.embeddings.create(model=embedding_model, input=x['text'])
                articles[i].update({"embedding": embedding.data[0].embedding})
                logger.info(f"Generated embedding for article: {x['title']}")
                # Save the embedding to a file
                with open(embedding_file, 'w') as ef:
                    json.dump({"embedding": embedding.data[0].embedding}, ef)
                    logger.info(f"Saved embedding for article: {x['title']} to {embedding_file}")
            except Exception as e:
                logger.error(f"Error processing article: {x['title']}, Error: {e}")
        else:
            with open(embedding_file, 'r') as ef: # Load the existing embedding
                embedding_data = json.load(ef)
                articles[i].update({"embedding": embedding_data['embedding']})
                logger.info(f"Loaded existing embedding for article: {x['title']}")
    qdrant = qdrant_client.QdrantClient(host='localhost')
    collection_name = 'help_center'
    vector_size = len(articles[0]['embedding'])
    article_df = pd.DataFrame(articles)
    logger.info("Checking if collection exists...")
    collections = qdrant.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    if collection_name not in collection_names:
        logger.info("Collection does not exist. Creating new collection...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                'article': rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size,
                )
            }
        )
    else:
        logger.info("Collection already exists.")
    logger.info("Populating collection with vectors...")
    # Populate collection with vectors
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            rest.PointStruct(
                id=k,
                vector={
                    'article': v['embedding'],
                },
                payload=v.to_dict(),
            )
            for k, v in article_df.iterrows()
        ],
    )
    logger.info("Data preparation complete.")

def main():
    prep_data()
    args = parse_args()
    try:
        validate_all_tools()
        validate_all_assistants()
    except:
        raise Exception("Validation failed")
    swarm = Swarm(
        engine_name=engine_name, persist=persist)
    if args.test is not None:
        test_files = args.test
        if len(test_files) == 0:
            test_file_paths = [f"{test_root}/{test_file}"]
        else:
            test_file_paths = [f"{test_root}/{file}" for file in test_files]
        swarm = Swarm(engine_name='local')
        swarm.deploy(test_mode=True, test_file_paths=test_file_paths)
    elif args.input:
        # Interactive mode for adding tasks
        while True:
            print("Enter a task (or 'exit' to quit):")
            task_input = input()
            # Check for exit command
            if task_input.lower() == 'exit':
                break
            # Use shlex to parse the task description and arguments
            task_args = shlex.split(task_input)
            task_parser = argparse.ArgumentParser()
            task_parser.add_argument("description", type=str, nargs='?', default="")
            task_parser.add_argument("--iterate", action="store_true", help="Set the iterate flag for the new task.")
            task_parser.add_argument("--evaluate", action="store_true", help="Set the evaluate flag for the new task.")
            task_parser.add_argument("--assistant", type=str, default="user_interface", help="Specify the assistant for the new task.")
            # Parse task arguments
            task_parsed_args = task_parser.parse_args(task_args)
            # Create and add the new task
            new_task = Task(description=task_parsed_args.description,
                            iterate=task_parsed_args.iterate,
                            evaluate=task_parsed_args.evaluate,
                            assistant=task_parsed_args.assistant)
            swarm.add_task(new_task)
            # Deploy Swarm with the new task
            swarm.deploy()
            swarm.tasks.clear()
    else:
        # Load predefined tasks if any
        # Deploy the Swarm for predefined tasks
        swarm.load_tasks()
        swarm.deploy()
    print("\n\n🍯🐝🍯 Swarm operations complete 🍯🐝🍯\n\n")

if __name__ == "__main__":
    main()