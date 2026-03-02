import requests
import json
import uuid
import math
import msgpack
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# We use bge-large-en per requirement as the dense vector model
# It outputs an embedding dimension of 1024
print("Loading Embedding Model...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

# ==========================================
# ENDEE DB CLIENT WRAPPER  
# REST API (http://localhost:8080)
# ==========================================

ENDEE_URL = "http://localhost:8080/api/v1"
DIMENSION = 1024 # Matches BAAI/bge-large-en-v1.5

# Track which collection indexes have been ensured this session
_ensured_indexes = set()

def init_endee(collection_name: str = "resume_chunks"):
    """Checks if index exists, and creates it if not."""
    try:
        res = requests.get(f"{ENDEE_URL}/index/list")
        if res.status_code == 200:
            indexes = res.json().get("indexes", [])
            print("Existing Indexes:", indexes)
            if collection_name not in indexes:
                print(f"Creating Endee Index: {collection_name}")
                create_payload = {
                    "index_name": collection_name,
                    "dim": DIMENSION,
                    "space_type": "cosine" # Cosine similarity for BGE models
                }
                res = requests.post(f"{ENDEE_URL}/index/create", json=create_payload)
                print("Create Response:", res.text)
    except Exception as e:
        print(f"Failed to connect to Endee Vector Database at {ENDEE_URL}. Ensure it is running.")
        print(str(e))
    _ensured_indexes.add(collection_name)

def encode_text(text: str) -> List[float]:
    """Uses sentence-transformers to encode a text chunk"""
    return embedder.encode(text).tolist()

def upsert_chunks(resume_id: str, formatted_chunks: List[Dict[str, Any]], collection_name: str = "resume_chunks"):
    """
    Upserts multiple chunk vectors into Endee.
    `formatted_chunks` is a list of {"text": str, "section": str}
    """
    vectors = []
    
    for chunk in formatted_chunks:
        text = chunk["text"]
        section = chunk.get("section", "general")
        
        # We append the section to the text to provide better context
        contextual_text = f"Section: {section}\n\n{text}"
        embedding = encode_text(contextual_text)
        
        chunk_id = str(uuid.uuid4())
        
        vectors.append({
            "id": chunk_id,
            "vector": embedding,
            "meta": json.dumps({
                "resume_id": resume_id,
                "section": section,
                "text": text
            })
        })
        
    # Ensure the collection index exists before inserting
    if collection_name not in _ensured_indexes:
        init_endee(collection_name)
    
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(f"{ENDEE_URL}/index/{collection_name}/vector/insert", json=vectors, headers=headers)
        if res.status_code != 200:
            print(f"Batch insert failed ({res.status_code}): {res.text}")
            # Try one-by-one as fallback
            print("Attempting individual vector inserts...")
            for v in vectors:
                try:
                    r2 = requests.post(f"{ENDEE_URL}/index/{collection_name}/vector/insert", json=[v], headers=headers)
                    if r2.status_code != 200:
                        print(f"  Single insert failed for {v['id']}: {r2.text}")
                except Exception as e2:
                    print(f"  Single insert error: {e2}")
        else:
            print(f"Inserted {len(vectors)} vectors into '{collection_name}'")
    except Exception as e:
         print(f"Failed to insert vectors: {e}")


def search_similar_chunks(query_text: str, top_k: int = 15, collection_name: str = "resume_chunks") -> List[Dict[str, Any]]:
    """
    Searches Endee DB for the most semantically relevant chunks.
    Retrieves more chunks than `top_k` initially assuming cross-encoder reranking will filter them.
    """
    query_vector = encode_text(query_text)
    
    payload = {
        "k": top_k,
        "vector": query_vector
    }
    
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(f"{ENDEE_URL}/index/{collection_name}/search", json=payload, headers=headers)
        if res.status_code == 200:
            results_raw = msgpack.unpackb(res.content, raw=False)
            
            formatted_results = []
            if results_raw and len(results_raw) > 0:
                # results_raw can be nested; find the actual list of result tuples
                items = results_raw[0] if isinstance(results_raw[0], list) else results_raw
                
                for item in items:
                    # Each item should be a tuple/list of (similarity, chunk_id, metadata)
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                        
                    similarity = item[0]
                    chunk_id = item[1]
                    meta_raw = item[2]
                    
                    if isinstance(meta_raw, list):
                        meta_str = bytes(meta_raw).decode('utf-8', errors='ignore')
                    elif isinstance(meta_raw, bytes) or isinstance(meta_raw, bytearray):
                        meta_str = meta_raw.decode('utf-8', errors='ignore')
                    else:
                        meta_str = str(meta_raw)
                        
                    meta_dict = {}
                    if meta_str:
                        try:
                            meta_dict = json.loads(meta_str)
                        except:
                            pass
                    
                    # Endee Cosine Similarity might output slight bound errors (like 1.000000001) or NaNs
                    try:
                        sim_float = float(similarity)
                        if math.isnan(sim_float) or math.isinf(sim_float):
                            sim_float = 0.0
                    except:
                        sim_float = 0.0
                    
                    formatted_results.append({
                        "id": chunk_id,
                        "score": sim_float,
                        "distance": max(0.0, 1.0 - sim_float),
                        "metadata": meta_dict
                    })
            return formatted_results
        else:
            print(f"Search failed: {res.text}")
            return []
    except Exception as e:
         print(f"Search request failed: {e}")
         return []

# Initialize index on module load
init_endee()
