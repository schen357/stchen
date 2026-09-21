import numpy as np
import faiss

# 1. Define dimensions and generate dummy data
dimension = 64               # Dimensionality of your vectors
num_vectors = 10000          # Number of items in your database

# FAISS requires float32 arrays
data = np.random.random((num_vectors, dimension)).astype('float32')

# 2. Initialize the index using L2 (Euclidean) distance
index = faiss.IndexFlatL2(dimension)

# 3. Add vectors to the index
index.add(data)
print(f"Total vectors in index: {index.ntotal}")

# 4. Search the index
num_queries = 5
query_vectors = np.random.random((num_queries, dimension)).astype('float32')

k = 3  # Number of nearest neighbors to return
distances, indices = index.search(query_vectors, k)

# Results
print("\nNearest neighbor indices:\n", indices)
print("\nSquared L2 distances:\n", distances)