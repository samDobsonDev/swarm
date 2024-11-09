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

from typing import Optional, Any
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client.http import models as rest

TRIAGE_MESSAGE_PROMPT = "Given the following message: {}, select which assistant of the following is best suited to handle it: {}. Respond with JUST the name of the assistant, nothing else"
TRIAGE_SYSTEM_PROMPT = "You are an assistant who triages requests and selects the best assistant to handle that request."
EVALUATE_TASK_PROMPT = """Your task was {}. The steps you completed, and the output of those steps, are here: {}. IMPORTANT: Output the following, 'true' or 'false' if you successfully completed the task. Even if your plan changed from original plan, evaluate if the new plan and output
correctly satisfied the given task. Additionally, output a message for the user, explaining why a task was successfully completed, or why it failed. Example:
Task: "Tell a joke about cars. Translate it to Spanish"
Original Plan: [{{tool: "tell_joke", args: {{input: "cars"}}, {{tool: "translate", args: {{language: "Spanish"}}]
Steps Completed: [{{tool: "tell_joke", args: {{input: "cars", output: "Why did the car stop? It ran out of gas!"}}, {{tool: "translate", args: {{language: "Spanish", output: "¿Por qué se detuvo el coche? ¡Se quedó sin gas!"}}]
OUTPUT: ['true','The joke was successfully told and translated to Spanish.']
MAKE SURE THAT OUTPUT IS a list, bracketed by square brackets, with the first element being either 'true' or 'false', and the second element being a string message."""

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

# Options are 'assistants' or 'completion'
engine_name = 'completion'
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
    model: str = "gpt-4o",
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

    logging.info(f"Calling chat.completions.create with parameters: {request_params}")

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

class Swarm:
    def __init__(self, engine_name, tasks=[], persist = False):
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
            from swarm.assistants_api import AssistantsEngine
            self.engine = AssistantsEngine(client, self.tasks)
            self.engine.deploy(client, test_mode, test_file_paths)
        elif self.engine_name == 'completion':
            print(f"{Colors.GREY}Selected engine: Completion{Colors.ENDC}")
            from swarm.completions_api import CompletionsEngine
            self.engine = CompletionsEngine(client, self.tasks, persist=self.persist)
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
    client = OpenAI()
    embedding_model = "text-embedding-3-large"
    article_list = os.listdir('data')
    articles = []
    logging.info("Starting to process articles...")
    embedding_dir = 'embeddings' # Directory to store and retrieve embeddings
    os.makedirs(embedding_dir, exist_ok=True)
    for x in article_list:
        article_path = 'data/' + x
        with open(article_path) as f:
            data = json.load(f)
            articles.append(data)
            logging.info(f"Loaded article: {x}")
    for i, x in enumerate(articles):
        embedding_file = os.path.join(embedding_dir, f"{x['title']}.json")
        if generate_new_embeddings or not os.path.exists(embedding_file):
            try:
                embedding = client.embeddings.create(model=embedding_model, input=x['text'])
                articles[i].update({"embedding": embedding.data[0].embedding})
                logging.info(f"Generated embedding for article: {x['title']}")
                # Save the embedding to a file
                with open(embedding_file, 'w') as ef:
                    json.dump({"embedding": embedding.data[0].embedding}, ef)
                    logging.info(f"Saved embedding for article: {x['title']} to {embedding_file}")
            except Exception as e:
                logging.error(f"Error processing article: {x['title']}, Error: {e}")
        else:
            with open(embedding_file, 'r') as ef: # Load the existing embedding
                embedding_data = json.load(ef)
                articles[i].update({"embedding": embedding_data['embedding']})
                logging.info(f"Loaded existing embedding for article: {x['title']}")
    qdrant = qdrant_client.QdrantClient(host='localhost')
    collection_name = 'help_center'
    vector_size = len(articles[0]['embedding'])
    article_df = pd.DataFrame(articles)
    logging.info("Checking if collection exists...")
    collections = qdrant.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    if collection_name not in collection_names:
        logging.info("Collection does not exist. Creating new collection...")
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
        logging.info("Collection already exists.")
    logging.info("Populating collection with vectors...")
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
    logging.info("Data preparation complete.")

def main():
    prep_data()
    args = parse_args()
    try:
        validate_all_tools()
        validate_all_assistants()
    except:
        raise Exception("Validation failed")
    swarm = Swarm(engine_name = engine_name, persist = persist)
    if args.test is not None:
        test_files = args.test
        if len(test_files) == 0:
            test_file_paths = [f"{test_root}/{test_file}"]
        else:
            test_file_paths = [f"{test_root}/{file}" for file in test_files]
        swarm = Swarm(engine_name = 'local')
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