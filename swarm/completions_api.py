import importlib
import json
import os

from typing import Optional, Literal, Dict, List
from pydantic import BaseModel, Field

from swarm.advanced_core import Assistant, Colors, TRIAGE_SYSTEM_PROMPT, TRIAGE_MESSAGE_PROMPT, get_completion, \
    EvaluationTask, EVALUATE_TASK_PROMPT

max_iterations = 5

ITERATE_PROMPT = "Your task to complete is {}. You previously generated the following plan: {}. The steps completed, and the output of those steps, are here: {}. IMPORTANT: Given the outputs of the previous steps, use that to create a revised plan, using the following planning prompt."
EVAL_GROUNDTRUTH_PROMPT = "Given the following completion: {}, and the expected completion: {}, select whether the completion and expected completion are the same in essence. Correctness does not mean they are the same verbatim, but that the ANSWER is the same. For example: 'The answer, after calculating, is 4' and '4' would be the same. But 'it is 5' and 'the answer is 12' would be different. Respond with ONLY 'true' or 'false'"
EVAL_PLANNING_PROMPT = "Given the following plan: {}, and the expected plan: {}, select whether the plan and expected plan are the same in essence. Correctness does not mean they are the same verbatim, but that the content is the same with just minor formatting differences. Respond with ONLY 'true' or 'false'"
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

def is_dict_empty(d):
    return all(not v for v in d.values())

def evaluate_assistant_on_task(client, task, plan_log):
    """Evaluates the assistant's performance on a task"""
    output = get_completion(client,
                            [{'role': 'user', 'content': EVALUATE_TASK_PROMPT.format(task.description, plan_log)}])
    output.content = output.content.replace("'", '"')
    try:
        return json.loads(output.content)
    except json.JSONDecodeError:
        print("An error occurred while decoding the JSON.")
        return None

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


class CompletionsEngine:
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
            tool_output = handle_tool_call(step)
            plan_log['step'].append(step)
            plan_log['step_output'].append(tool_output)
            if task.iterate and not is_dict_empty(plan_log) and plan:
               iterations += 1
               new_task = ITERATE_PROMPT.format(task.description, original_plan, plan_log)
               plan = run.generate_plan(new_task)
            # Store the output for the next iteration
            self.store_context_globally(assistant)
        return original_plan, plan_log

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
                output = evaluate_assistant_on_task(self.client, task, plan_log)
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