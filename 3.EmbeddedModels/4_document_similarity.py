from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "The capital of India is New Delhi.",
    "Python is a popular programming language.",
    "The Eiffel Tower is located in Paris."
]

query = "Tell me about the capital city of India."
doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, scores = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(query)
print(documents[index])
