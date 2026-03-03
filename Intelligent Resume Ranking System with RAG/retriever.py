import math
from typing import List, Dict, Any
from vector_store import search_similar_chunks
from rank_bm25 import BM25Okapi

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalization of an array of scores to a [0, 1] range."""
    if not scores: return []
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score: return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

def lexical_search_bm25(query: str, chunks: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Performs deterministic BM25 search over a given corpus of document chunks.
    This acts as the keyword-specific arm of the Hybrid search.
    """
    if not chunks:
        return []
        
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Bundle scores with chunks
    scored_chunks = []
    for i, s in enumerate(scores):
        if s > 0:
            chunk_copy = chunks[i].copy()
            chunk_copy["bm25_score"] = float(s)
            scored_chunks.append(chunk_copy)
            
    # Sort descending
    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_chunks[:top_k]

def hybrid_retrieve(query_text: str, all_memory_chunks: List[Dict[str, Any]], top_k: int = 20, collection_name: str = "resume_chunks") -> List[Dict[str, Any]]:
    """
    Executes standard hybrid retrieval integrating Endee vector similarity
    and BM25 sparse retrieval.
    """
    # 1. Semantic Search (Endee API)
    semantic_results = search_similar_chunks(query_text, top_k=top_k, collection_name=collection_name)
    
    semantic_dict = {}
    for res in semantic_results:
        text = res["metadata"].get("text", "")
        resume_id = res["metadata"].get("resume_id", "")
        uid = f"{resume_id}::{text[:30]}"
        
        semantic_dict[uid] = {
            "text": text,
            "section": res["metadata"].get("section", ""),
            "resume_id": resume_id,
            "semantic_score": res.get("distance", res.get("score", 0.0)) # Depending on distance metric used
        }

    # 2. Lexical Search (BM25)
    lexical_results = lexical_search_bm25(query_text, all_memory_chunks, top_k=top_k)
    
    lexical_dict = {}
    for res in lexical_results:
        text = res["text"]
        resume_id = res["resume_id"]
        uid = f"{resume_id}::{text[:30]}"
        
        lexical_dict[uid] = {
            "text": text,
            "section": res.get("section", ""),
            "resume_id": resume_id,
            "bm25_score": res["bm25_score"]
        }
        
    # 3. Reciprocal Rank Fusion / Score Normalization and merging
    merged_uids = set(semantic_dict.keys()).union(set(lexical_dict.keys()))
    
    raw_sem = [semantic_dict[k]["semantic_score"] for k in semantic_dict.keys()]
    raw_lex = [lexical_dict[k]["bm25_score"] for k in lexical_dict.keys()]
    
    norm_sem = normalize_scores(raw_sem)
    norm_lex = normalize_scores(raw_lex)
    
    for k, norm in zip(semantic_dict.keys(), norm_sem):
        semantic_dict[k]["norm_semantic"] = norm
    for k, norm in zip(lexical_dict.keys(), norm_lex):
        lexical_dict[k]["norm_bm25"] = norm
        
    final_results = []
    
    for uid in merged_uids:
        sem_data = semantic_dict.get(uid, {"norm_semantic": 0.0, "resume_id": "", "text": "", "section": ""})
        lex_data = lexical_dict.get(uid, {"norm_bm25": 0.0, "resume_id": "", "text": "", "section": ""})
        
        combined_score = (sem_data["norm_semantic"] * 0.7) + (lex_data["norm_bm25"] * 0.3)
        
        text = sem_data["text"] if sem_data["text"] else lex_data["text"]
        resume_id = sem_data["resume_id"] if sem_data["resume_id"] else lex_data["resume_id"]
        section = sem_data["section"] if sem_data["section"] else lex_data["section"]
        
        final_results.append({
            "text": text,
            "resume_id": resume_id,
            "section": section,
            "hybrid_score": combined_score,
            "vector_score": sem_data["norm_semantic"],
            "bm25_score": lex_data["norm_bm25"]
        })
        
    final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return final_results[:top_k]
