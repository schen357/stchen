""" Simple script to generate question set that can be answered by a given written process from the image_to_text pipeline."""
import pickle
from pathlib import Path
import pandas as pd

import anthropic
import json


PACKAGE_DIR = Path(__file__).parent
PROCESSES_PATH = PACKAGE_DIR / "flowchart_instructions.pkl"


def load_processes(path: Path = PROCESSES_PATH) -> list[str]:
    # Assumes that pickle file is available for reading
    with open(path, 'rb') as file:
        return pickle.load(file)


def parse_questions(message) -> list[str]:
    try:
        payload = message.content[-1].text.replace("json\n", "").replace("```","")
        payload_json = json.loads(payload)
        title = payload_json["process_title"]
        question_set = payload_json["question_set"]
        return title, question_set
    except:
        raise ValueError(f"### Cannot parse: {message.content}")


def generate_question_set(client, processes_list) -> pd.DataFrame:
    question_set = pd.DataFrame()
    # Set instructions for Claude to format the output clearly
    system_prompt = """
        You are creating a question set that can be answered by the reference process. Provide a short title for the process, and write as many relevant questions as you can think of. 
        Your output must be a valid JSON payload in the following format:

        {
            "process_title": "<A short, unique title that summarizes what the process is. Example: 'Order Quality Assurance'>",
            "question_set": [
                "First question here?",
                "Second question here?",
                "Third question here?"
            ]
        }
        """
    message_content = f"Please create the question set of relevant questions that can be answered by the following process: {process}"
    for process in processes_list:
        # Call the Claude API
        message = call_anthropic_api(client, system_prompt, user_message)

        title, question_list = parse_questions(message)
        question_mapping = pd.DataFrame({
            "process_title": [title],
             "process": [process],
            "question_set": [question_list]
        })
        question_set = pd.concat([question_set, question_mapping], ignore_index = True)

    return question_set


if __name__ == "__main__":
    # Initialize the Anthropic client (reads ANTHROPIC_API_KEY from your environment variables)
    client = anthropic.Anthropic()

    processes_list = load_processes()
    question_set = generate_question_set(client, processes_list)

    question_set.to_parquet("qa_generation_api/question_set.pq")
