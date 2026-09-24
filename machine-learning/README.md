End to end, I want to be able to interact with an LLM that uses rag to answer questions about specialized processes from flowchart images  

- image to text: Example of using a HuggingFace open source VLM visual language model to translate flow chart images to written text instructions of process
- qa generation api: Calls anthropic API to synthesize question sets per process from image to text.
- rag: creates a small open source vector database using chromadb that stores my process and question answer sets for retrieval 
- advise on processes: ask an LLM questions where it uses it's specialized process retrieval to give better answers
