
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
POLICY_FILE = DOCUMENTS_DIR / "purchasing_policy.txt"
DB_DIR = BASE_DIR / "chroma_db"

# 2. Check that the policy document exists
if not POLICY_FILE.exists():
    raise FileNotFoundError(f"Policy file not found: {POLICY_FILE}")

# 3. Read the policy document
text = POLICY_FILE.read_text(encoding="utf-8")

# 4. Split the document into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

print(f"Document split into {len(chunks)} chunks.")

# 5. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 6. Convert each chunk into an embedding
embeddings = model.encode(chunks).tolist()

# 7. Create a local ChromaDB database
client = chromadb.PersistentClient(path=str(DB_DIR))

collection_name = "supply_chain_policy"

# Rebuild the collection each time this script runs
try:
    client.delete_collection(collection_name)
except Exception:
    pass

collection = client.create_collection(name=collection_name)

# 8. Store the chunks and their embeddings
collection.add(
    ids=[f"policy_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {"source": "purchasing_policy.txt", "chunk": i}
        for i in range(len(chunks))
    ]
)

print("Knowledge base created successfully!")
print(f"Stored {collection.count()} chunks in ChromaDB.")
print(f"Database location: {DB_DIR}")
