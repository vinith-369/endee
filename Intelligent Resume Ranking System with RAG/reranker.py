from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

print("Loading Cross-Encoder Model...")
# A popular, fast and performant cross-encoder for passage ranking
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Reranks a given list of retrieved chunks using a Cross-Encoder to refine
    relevance order over hybrid scores.
    """
    if not chunks:
        return []

    # The cross encoder takes pairs of (query, document)
    pairs = [[query, chunk["text"]] for chunk in chunks]
    
    # Compute the relevance scores
    scores = reranker_model.predict(pairs)
    
    # Bundle scores with chunks
    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)
        
    # Sort descending by the new rerank_score
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Return the top_k refined chunks
    return chunks[:top_k]
