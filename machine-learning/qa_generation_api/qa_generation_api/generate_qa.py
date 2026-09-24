import pickle
from pathlib import Path
import pandas as pd

import anthropic

# Initialize the Anthropic client (reads ANTHROPIC_API_KEY from your environment variables)
client = anthropic.Anthropic()

# Assumes that pickle file is available for reading
with open(Path(__file__).parent / "flowchart_instructions.pkl", 'rb') as file:
    processes_list = pickle.load(file)

# Set instructions for Claude to format the output clearly
system_prompt = """
You are creating a question set that can be answered by the reference process. Write as many relevant questions as you can think of. Your output should be a list of questions for the process. Only list the questions, separated by a pipe delimiter | for each new question.
"""

question_set = pd.DataFrame()
for process in processes_list:
    # Call the Claude API
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4000,
        system=system_prompt,
        messages=[
            {
                "role": "user", 
                "content": f"Please create the question set of relevant questions that can be answered by the following process: {process}"
            }
        ]
    )

    # Print the generated Q&A sets
    try:
        questions = message.content[1].text
        question_list = questions.split("|")
        question_mapping = pd.DataFrame({"question": question_list, "process": [process] * len(question_list)}) 
        question_set = pd.concat([question_set, question_mapping], ignore_index = True) 
    except:
        raise ValueError(f"### Cannot parse: {message.content}")

question_set.to_parquet("qa_generation_api/question_set.pq")
print(question_set)