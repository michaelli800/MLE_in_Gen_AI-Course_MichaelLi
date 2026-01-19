from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import sqlite3
from typing import List, Tuple
from dataclasses import dataclass

# Import necessary components for RAG and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- DATA STRUCTURES ---

@dataclass
class DocumentMetadata:
    doc_id: int
    title: str
    author: str
    year: int
    keywords: str

@dataclass
class SearchResultItem:
    chunk_text: str
    doc_id: int
    title: str
    score: float
    source: str  # 'faiss', 'fts5', or 'hybrid'

# --- 1. SETUP: DATABASE, FAISS INDEX, AND EMBEDDING MODEL ---

# Extended document set with metadata
sample_docs_with_metadata = [
    {
        "text": "Giulia contributed to a Python project focused on predictive analytics for stock market trends, using libraries like Pandas and Scikit-learn.",
        "metadata": {"doc_id": 1, "title": "Python Analytics Project", "author": "Giulia", "year": 2023, "keywords": "Python, analytics, stock market, Pandas, Scikit-learn"}
    },
    {
        "text": "Another key project was a Flask web application that serves real-time machine learning inference, deployed using Docker containers.",
        "metadata": {"doc_id": 2, "title": "Flask ML Application", "author": "Giulia", "year": 2023, "keywords": "Flask, machine learning, Docker, web application"}
    },
    {
        "text": "The third major project involved developing custom data pipelines with Apache Kafka for high-throughput data ingestion.",
        "metadata": {"doc_id": 3, "title": "Kafka Data Pipeline", "author": "Giulia", "year": 2024, "keywords": "Apache Kafka, data pipelines, ingestion"}
    },
    {
        "text": "General information: The candidate has a background in Computer Science and 5 years of professional experience.",
        "metadata": {"doc_id": 4, "title": "Candidate Background", "author": "HR", "year": 2024, "keywords": "Computer Science, experience, background"}
    },
]

print("Loading HuggingFace Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384

# --- 1.1 Initialize SQLite Database with FTS5 ---

def init_database():
    """Initialize SQLite database with document metadata and FTS5 full-text search."""
    conn = sqlite3.connect('rag_index.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create documents metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            year INTEGER,
            keywords TEXT
        )
    ''')
    
    # Create chunks table with metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_text TEXT,
            chunk_index INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    ''')
    
    # Create FTS5 virtual table for full-text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
            chunk_text,
            content=chunks,
            content_rowid=chunk_id
        )
    ''')
    
    conn.commit()
    return conn

db_conn = init_database()

# --- 1.2 Populate Database and Generate Embeddings ---

def populate_database_and_index():
    """Populate database with documents and build FAISS index."""
    cursor = db_conn.cursor()
    
    # Clear existing data
    cursor.execute('DELETE FROM documents')
    cursor.execute('DELETE FROM chunks')
    cursor.execute('DELETE FROM doc_chunks_fts')
    
    all_chunks = []
    all_vectors = []
    chunk_to_doc_map = []  # Maps chunk index to doc_id
    
    for doc_data in sample_docs_with_metadata:
        metadata = doc_data["metadata"]
        
        # Insert document metadata
        cursor.execute('''
            INSERT INTO documents (doc_id, title, author, year, keywords)
            VALUES (?, ?, ?, ?, ?)
        ''', (metadata["doc_id"], metadata["title"], metadata["author"], 
              metadata["year"], metadata["keywords"]))
        
        # Create Document objects and split into chunks
        lc_doc = Document(page_content=doc_data["text"])
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([lc_doc])
        
        # Store chunks and generate embeddings
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            
            # Insert into chunks table
            cursor.execute('''
                INSERT INTO chunks (doc_id, chunk_text, chunk_index)
                VALUES (?, ?, ?)
            ''', (metadata["doc_id"], chunk_text, idx))
            
            chunk_id = cursor.lastrowid
            
            # Insert into FTS5 table
            cursor.execute('''
                INSERT INTO doc_chunks_fts (rowid, chunk_text)
                VALUES (?, ?)
            ''', (chunk_id, chunk_text))
            
            all_chunks.append(chunk_text)
            chunk_to_doc_map.append(metadata["doc_id"])
    
    db_conn.commit()
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    all_vectors = np.array(embedding_model.embed_documents(all_chunks), dtype='float32')
    
    # Initialize FAISS index
    print(f"Initializing FAISS Index with dim {EMBEDDING_DIM}...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(all_vectors)
    print(f"FAISS Index ready with {index.ntotal} vectors.")
    
    return index, all_chunks, chunk_to_doc_map

faiss_index, chunks_text, chunk_to_doc_map = populate_database_and_index()

# --- 2. SEARCH FUNCTIONS ---

def fts5_search(query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
    """Perform FTS5 keyword search and return results with BM25 scores."""
    cursor = db_conn.cursor()
    
    # FTS5 query with BM25 ranking
    cursor.execute('''
        SELECT chunks.chunk_id, chunks.chunk_text, chunks.doc_id, bm25(doc_chunks_fts) as score
        FROM doc_chunks_fts
        JOIN chunks ON doc_chunks_fts.rowid = chunks.chunk_id
        WHERE doc_chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
    ''', (query, top_k))
    
    results = []
    for row in cursor.fetchall():
        chunk_id, chunk_text, doc_id, score = row
        # BM25 scores are negative (lower is better), convert to positive for normalization
        results.append((chunk_id, chunk_text, doc_id, abs(score)))
    
    return results

def faiss_search(query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
    """Perform FAISS semantic search."""
    query_vector = np.array([embedding_model.embed_query(query)], dtype='float32')
    distances, indices = faiss_index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk_text = chunks_text[idx]
        doc_id = chunk_to_doc_map[idx]
        distance = float(distances[0][i])
        results.append((idx, chunk_text, doc_id, distance))
    
    return results

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores or len(scores) == 1:
        return [1.0] * len(scores)
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]

def reciprocal_rank_fusion(
    faiss_results: List[Tuple], 
    fts_results: List[Tuple], 
    k: int = 60
) -> List[Tuple[str, int, float]]:
    """Merge results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    
    # Process FAISS results
    for rank, (idx, chunk_text, doc_id, _) in enumerate(faiss_results):
        key = (chunk_text, doc_id)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
    
    # Process FTS5 results
    for rank, (chunk_id, chunk_text, doc_id, _) in enumerate(fts_results):
        key = (chunk_text, doc_id)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
    
    # Sort by RRF score (descending)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(chunk_text, doc_id, score) for (chunk_text, doc_id), score in sorted_results]

def weighted_hybrid_search(
    query: str, 
    top_k: int = 5, 
    faiss_weight: float = 0.5,
    fts_weight: float = 0.5
) -> List[SearchResultItem]:
    """Perform hybrid search using weighted score combination."""
    # Get results from both search methods
    faiss_results = faiss_search(query, top_k)
    fts_results = fts5_search(query, top_k)
    
    # Normalize scores
    faiss_scores = normalize_scores([r[3] for r in faiss_results])
    fts_scores = normalize_scores([r[3] for r in fts_results])
    
    # Combine results with weighted scores
    combined = {}
    
    for i, (idx, chunk_text, doc_id, _) in enumerate(faiss_results):
        key = (chunk_text, doc_id)
        # For FAISS, lower distance is better, so invert normalized score
        combined[key] = combined.get(key, 0) + faiss_weight * (1 - faiss_scores[i])
    
    for i, (chunk_id, chunk_text, doc_id, _) in enumerate(fts_results):
        key = (chunk_text, doc_id)
        combined[key] = combined.get(key, 0) + fts_weight * fts_scores[i]
    
    # Sort by combined score
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Get document metadata and create result objects
    cursor = db_conn.cursor()
    results = []
    
    for (chunk_text, doc_id), score in sorted_results:
        cursor.execute('SELECT title FROM documents WHERE doc_id = ?', (doc_id,))
        title = cursor.fetchone()[0]
        
        results.append(SearchResultItem(
            chunk_text=chunk_text,
            doc_id=doc_id,
            title=title,
            score=score,
            source='hybrid'
        ))
    
    return results

# --- 3. FASTAPI APPLICATION ---

app = FastAPI(title="Hybrid Vector Search API with FAISS and FTS5")

class SearchResponse(BaseModel):
    query: str
    method: str
    results: List[dict]

@app.get("/search/faiss", response_model=SearchResponse)
async def search_faiss(q: str, k: int = 3):
    """Semantic search using FAISS only."""
    results = faiss_search(q, k)
    
    cursor = db_conn.cursor()
    formatted_results = []
    
    for idx, chunk_text, doc_id, score in results:
        cursor.execute('SELECT title FROM documents WHERE doc_id = ?', (doc_id,))
        title = cursor.fetchone()[0]
        
        formatted_results.append({
            "text": chunk_text,
            "doc_id": doc_id,
            "title": title,
            "score": float(score)
        })
    
    return {"query": q, "method": "faiss", "results": formatted_results}

@app.get("/search/fts5", response_model=SearchResponse)
async def search_fts(q: str, k: int = 3):
    """Keyword search using SQLite FTS5."""
    results = fts5_search(q, k)
    
    cursor = db_conn.cursor()
    formatted_results = []
    
    for chunk_id, chunk_text, doc_id, score in results:
        cursor.execute('SELECT title FROM documents WHERE doc_id = ?', (doc_id,))
        title = cursor.fetchone()[0]
        
        formatted_results.append({
            "text": chunk_text,
            "doc_id": doc_id,
            "title": title,
            "score": float(score)
        })
    
    return {"query": q, "method": "fts5", "results": formatted_results}

@app.get("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(q: str, k: int = 3, method: str = "weighted", 
                       faiss_weight: float = 0.5):
    """
    Hybrid search combining FAISS and FTS5.
    
    method: 'weighted' (default) or 'rrf' (reciprocal rank fusion)
    faiss_weight: Weight for FAISS results (0-1), only used with 'weighted' method
    """
    if method == "rrf":
        faiss_results = faiss_search(q, k * 2)
        fts_results = fts5_search(q, k * 2)
        merged = reciprocal_rank_fusion(faiss_results, fts_results)[:k]
        
        cursor = db_conn.cursor()
        formatted_results = []
        
        for chunk_text, doc_id, score in merged:
            cursor.execute('SELECT title FROM documents WHERE doc_id = ?', (doc_id,))
            title = cursor.fetchone()[0]
            
            formatted_results.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "title": title,
                "score": float(score)
            })
        
        return {"query": q, "method": "hybrid-rrf", "results": formatted_results}
    
    else:  # weighted
        fts_weight = 1 - faiss_weight
        results = weighted_hybrid_search(q, k, faiss_weight, fts_weight)
        
        formatted_results = [{
            "text": r.chunk_text,
            "doc_id": r.doc_id,
            "title": r.title,
            "score": r.score
        } for r in results]
        
        return {"query": q, "method": "hybrid-weighted", "results": formatted_results}

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Hybrid RAG Search API",
        "endpoints": {
            "/search/faiss": "Semantic search with FAISS",
            "/search/fts5": "Keyword search with FTS5",
            "/search/hybrid": "Hybrid search (weighted or RRF)"
        }
    }

# To run this:
# uvicorn main:app --reload
#
# Example queries:
# http://127.0.0.1:8000/search/faiss?q=machine%20learning&k=3
# http://127.0.0.1:8000/search/fts5?q=Python%20Pandas&k=3
# http://127.0.0.1:8000/search/hybrid?q=Flask%20Docker&k=3&method=weighted&faiss_weight=0.7
# http://127.0.0.1:8000/search/hybrid?q=data%20pipelines&k=3&method=rrf