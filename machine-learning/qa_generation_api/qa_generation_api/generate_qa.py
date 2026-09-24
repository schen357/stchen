""" Simple script to generate question set that can be answered by a given written process from the image_to_text pipeline."""
import pickle
from pathlib import Path
import pandas as pd

import anthropic


PACKAGE_DIR = Path(__file__).parent
PROCESSES_PATH = PACKAGE_DIR / "flowchart_instructions.pkl"
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4000

# Set instructions for Claude to format the output clearly
SYSTEM_PROMPT = """
    You are creating a question set that can be answered by the reference process. Write as many relevant questions as you can think of. Your output should be a list of questions for the process. Only list the questions, separated by a pipe delimiter | for each new question.
    """


def load_processes(path: Path = PROCESSES_PATH) -> list[str]:
    # Assumes that pickle file is available for reading
    with open(path, 'rb') as file:
        return pickle.load(file)


def parse_questions(message) -> list[str]:
    try:
        questions = message.content[-1].text
        return questions.split("|")
    except:
        raise ValueError(f"### Cannot parse: {message.content}")


def generate_question_set(client, processes_list) -> pd.DataFrame:
    question_set = pd.DataFrame()
    for process in processes_list:
        # Call the Claude API
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Please create the question set of relevant questions that can be answered by the following process: {process}"
                }
            ]
        )

        question_list = parse_questions(message)
        question_mapping = pd.DataFrame({"question": question_list, "process": [process] * len(question_list)})
        question_set = pd.concat([question_set, question_mapping], ignore_index = True)

    return question_set


if __name__ == "__main__":
    # Initialize the Anthropic client (reads ANTHROPIC_API_KEY from your environment variables)
    client = anthropic.Anthropic()

    processes_list = load_processes()
    question_set = generate_question_set(client, processes_list)

    question_set.to_parquet("qa_generation_api/question_set.pq")
