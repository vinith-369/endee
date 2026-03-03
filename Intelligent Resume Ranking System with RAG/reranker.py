from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

print("Loading Cross-Encoder Model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Reranks a given list of retrieved chunks using a Cross-Encoder to refine
    relevance order over hybrid scores.
    """
    if not chunks:
        return []

    pairs = [[query, chunk["text"]] for chunk in chunks]
    
    scores = reranker_model.predict(pairs)
    
    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)
        
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return chunks[:top_k]
