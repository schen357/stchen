""" Simple script to generate question set that can be answered by a given written process from the image_to_text pipeline."""
import pickle
from pathlib import Path
import pandas as pd

import anthropic
import json

from api_generation.utils import call_anthropic_api


PACKAGE_DIR = Path(__file__).parent
PROCESSES_PATH = PACKAGE_DIR / "flowchart_instructions.pkl"
QUESTION_SET_PATH = PACKAGE_DIR / "question_set.pq"

# Set instructions for Claude to format the output clearly
SYSTEM_PROMPT = """
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


def load_processes(path: Path = PROCESSES_PATH) -> list[str]:
    # Assumes that pickle file is available for reading
    with open(path, 'rb') as file:
        return pickle.load(file)


def parse_questions(message) -> tuple[str, list[str]]:
    try:
        payload = message.content[-1].text.strip()
        # Claude sometimes wraps the JSON in a ```json ... ``` markdown fence
        payload = payload.removeprefix("```json").removeprefix("```").removesuffix("```")
        payload_json = json.loads(payload)
        title = payload_json["process_title"]
        question_set = payload_json["question_set"]
        return title, question_set
    except (IndexError, AttributeError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"### Cannot parse: {message.content}") from e


def generate_question_set(client, processes_list) -> pd.DataFrame:
    rows = []
    for process in processes_list:
        user_message = f"Please create the question set of relevant questions that can be answered by the following process: {process}"
        # Call the Claude API
        message = call_anthropic_api(client, SYSTEM_PROMPT, user_message)

        title, question_list = parse_questions(message)
        rows.append({
            "process_title": title,
            "process": process,
            "question_set": question_list
        })

    return pd.DataFrame(rows, columns=["process_title", "process", "question_set"])


if __name__ == "__main__":
    # Initialize the Anthropic client (reads ANTHROPIC_API_KEY from your environment variables)
    client = anthropic.Anthropic()

    processes_list = load_processes()
    question_set = generate_question_set(client, processes_list)

    question_set.to_parquet(QUESTION_SET_PATH)
