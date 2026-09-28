End to end, I want to be able to interact with an LLM that uses rag to answer questions about specialized processes from flowchart images  

- image to text
    - Example of using a HuggingFace open source VLM visual language model to translate flow chart images to written text instructions of process
- api generation
    - Calls anthropic API to synthesize question sets per process from image to text.
    - Calls anthropic API again with RAG - uses different indices to supplement context
- rag sources 
    - Creates a small open source vector database using chromadb that stores my process and question answer sets for retrieval 

