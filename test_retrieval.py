
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Find the saved database
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing knowledge base
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name="supply_chain_policy")

# Ask a question
question = "What should we do when stock is critical?"

# Convert the question into an embedding
question_embedding = model.encode([question]).tolist()

# Search for the most relevant policy chunks
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

print("\nQuestion:", question)
print("\nRelevant information found in the policy:\n")

for document in results["documents"][0]:
    print("-", document)
    print()
