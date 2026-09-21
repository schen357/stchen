RAG example

From the rag directory: 

- source .venv/bin/activate
- uv sync 
- python rag/flowchart_to_instructions.py

This generates a pickle file of verbal written instructions from flowchart images in the directory and saves the list to a pickle file.

Next steps:
- Generate mapping of process files to various questions 
- Add a vector search db or create FAISS index
