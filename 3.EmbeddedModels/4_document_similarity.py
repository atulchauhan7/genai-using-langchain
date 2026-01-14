
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Use a local embedding model (no API needed)
model = SentenceTransformer('all-MiniLM-L6-v2')


documents = [
    "The capital of India is New Delhi.",
    "Python is a popular programming language.",
    "The Eiffel Tower is located in Paris."
]

query = "Tell me about the capital city of India."

# Get embeddings
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])[0]

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get the most similar document
index = np.argmax(scores)

print(query)
print(documents[index])
