This pipeline takes images and converts them to step by step text instructions using an open source HF model.

From the machine-learning/image_to_text directory in bash, run the following:  
- source .venv/bin/activate
- uv sync 
- python rag/flowchart_to_instructions.py
This generates a pickle file of verbal written instructions from flowchart images in the directory and saves the list to a pickle file.
