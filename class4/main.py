
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = ...  # e.g., model.encode([q])[0]
    # Perform FAISS search
    k = 3
    distances, indices = faiss_index.search(np.array([query_vector]), k)
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return {"query": q, "results": results}
