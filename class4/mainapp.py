from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss

# Import necessary components for RAG and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings # Using langchain_community
import os

# --- 1. SETUP: FAISS Index and Embedding Model Initialization ---

# Define a simple document set (replace with your actual document loading)
sample_docs = [
    "Giulia contributed to a Python project focused on predictive analytics for stock market trends, using libraries like Pandas and Scikit-learn.",
    "Another key project was a Flask web application that serves real-time machine learning inference, deployed using Docker containers.",
    "The third major project involved developing custom data pipelines with Apache Kafka for high-throughput data ingestion.",
    "General information: The candidate has a background in Computer Science and 5 years of professional experience.",
]

# 1.1 Load Embedding Model and Generate Chunks
# Note: This model runs locally and is often fast for smaller vectors.
print("Loading HuggingFace Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384 # The dimension for all-MiniLM-L6-v2

# Create Document objects (needed for LangChain splitters)
lc_docs = [Document(page_content=d) for d in sample_docs]

# Split into chunks (optional for this small sample, but good practice)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(lc_docs)
chunks_text = [c.page_content for c in chunks]


# 1.2 Generate Embeddings for the Chunks
print("Generating embeddings for documents...")
chunk_vectors = np.array(embedding_model.embed_documents(chunks_text), dtype='float32')

# 1.3 Initialize and Add to FAISS Index
print(f"Initializing FAISS Index (IndexFlatL2) with dim {EMBEDDING_DIM}...")
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
faiss_index.add(chunk_vectors)
print(f"FAISS Index ready with {faiss_index.ntotal} vectors.")

# --- 2. FASTAPI APPLICATION ---
app = FastAPI(title="Vector Search API with FAISS")

# Pydantic model for response validation
class SearchResult(BaseModel):
    query: str
    results: list[str]

@app.get("/search", response_model=SearchResult)
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages from the FAISS index, and return them.
    """
    
    # 2.1 Embed the query 'q'
    # The model.embed_query returns a list of floats. We convert it to a numpy array.
    query_vector_list = embedding_model.embed_query(q)
    
    # Reshape and convert the query vector for FAISS
    # Must be shape [1, dim] and dtype 'float32'
    query_vector = np.array([query_vector_list], dtype='float32')
    
    # 2.2 Perform FAISS search
    k = 3
    # distances: shape [1, k], indices: shape [1, k]
    distances, indices = faiss_index.search(query_vector, k)
    
    # 2.3 Retrieve the corresponding chunks (assuming 'chunks' list stores the text)
    results = []
    # indices[0] is the flat list of top k indices (e.g., [2, 0, 1])
    for idx in indices[0]:
        # We use chunks_text to retrieve the raw string content
        results.append(chunks_text[idx]) 
        
    return {"query": q, "results": results}

# To run this, save it as `main.py` and execute:
# uvicorn main:app --reload

#http://127.0.0.1:8000/search?q=%22Giulia%20contributed%20to%20what%20project%22