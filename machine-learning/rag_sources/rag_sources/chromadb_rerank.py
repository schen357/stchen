"""
Handles embedding the docs, just need to add them to the collection.

Trying two tactics for enhancing results: 1. adding question sets to the documents (processes) 2. Broad search + reranking.
"""

import chromadb
from pathlib import Path
import pandas as pd
from sentence_transformers import CrossEncoder

PACKAGE_DIR = Path(__file__).parent
QUESTION_SET_PATH = PACKAGE_DIR / "question_set.pq"

df_processes = pd.read_parquet(QUESTION_SET_PATH)
print(df_processes.head())

# ==========================================
# Create the document collection
# ==========================================
# 1. Initialize a Persistent Client (Saves data to a local directory)
# For temporary testing in memory without saving, use: chromadb.Client()
client = chromadb.PersistentClient(path="./my_chroma_db")

# 2. Create or fetch a Collection (similar to a table in SQL)
collection = client.get_or_create_collection(name="processes")

# 3. Add documents to your collection
# Chroma automatically converts these strings into vector embeddings behind the scenes
collection.upsert(
    ids=df_processes.process_title.tolist(),
    documents=df_processes.process.tolist(),
    # metadatas=[
    #     {"category": "coding"},
    #     {"category": "databases"},
    #     {"category": "science"},
    #     {"category": "ai"}
    # ] no metadata is available here but we could add tags as needed.
)

# ==========================================
# Create a broad search of relevant results
# ==========================================
# 4. Perform a semantic (similarity) search
query = "What checks are performed as part of quality control?"
print(f"Querying for: '{query}'\n")

results = collection.query(
    query_texts=[query],
    n_results=3  # Number of closest matching documents to return
)

retrieved_docs = results['documents'][0]
# 5. Parse and print the results
for idx in range(len(retrieved_docs)):
    doc = retrieved_docs[idx]
    doc_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    metadata = results['metadatas'][0][idx]
    
    # Distance measures how 'far apart' the query is from the result. Lower is closer/better.
    print(f"Result #{idx+1} [ID: {doc_id}] (Distance score: {distance:.3f})")
    print(f"Metadata: {metadata}")
    print(f"Text: \"{doc}\"\n")


# ==========================================
# Apply a Cross-Encoder Reranker to improve results
# ==========================================
# Load a tiny, highly efficient pre-trained reranking model
# (ms-marco-MiniLM-L6-v2 is the standard default recommendation)
reranker = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", trust_remote_code=True)

# Cross-encoders expect pairs: [[query, doc1], [query, doc2], ...]
query_doc_pairs = [[query, doc] for doc in retrieved_docs]

# Generate scores (Higher scores mean more relevant)
scores = reranker.predict(query_doc_pairs)

# Combine, sort by score descending, and output
reranked_results = sorted(
    zip(retrieved_docs, scores), 
    key=lambda x: x[1], 
    reverse=True
)

print("✅ [STAGE 2] Final Reranked Ranking (By Cross-Encoder):")
for idx, (doc, score) in enumerate(reranked_results):
    print(f"  {idx+1}. [Score: {score:6.2f}] -> {doc}")