import uuid
import os
import json
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from parser import process_resume
from extractor import ingest_resume_document, extract_jd_metadata
from retriever import hybrid_retrieve
from reranker import rerank_chunks
from evaluator import evaluate_candidate
from models import JobDescriptionRequest, EvaluateResponse, JDExtraction, CollectionResponse

app = FastAPI(title="Resume Shortlisting AI - Endee Vector DB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

ALL_CHUNKS: List[Dict[str, Any]] = []
CANDIDATE_METADATA: Dict[str, Any] = {}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_collection(collection_name: str):
    global ALL_CHUNKS, CANDIDATE_METADATA
    path = os.path.join(DATA_DIR, f"{collection_name}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            ALL_CHUNKS = data.get("chunks", [])
            raw_meta = data.get("metadata", {})
            from models import CandidateExtraction
            CANDIDATE_METADATA = {}
            for k, v in raw_meta.items():
                if isinstance(v, dict):
                    CANDIDATE_METADATA[k] = CandidateExtraction(**v)
                else:
                    CANDIDATE_METADATA[k] = v
    else:
        ALL_CHUNKS = []
        CANDIDATE_METADATA = {}

def save_collection(collection_name: str):
    path = os.path.join(DATA_DIR, f"{collection_name}.json")
    serializable_meta = {}
    for k, v in CANDIDATE_METADATA.items():
        if hasattr(v, 'model_dump'):
            serializable_meta[k] = v.model_dump()
        else:
            serializable_meta[k] = v
    with open(path, "w") as f:
        json.dump({
            "chunks": ALL_CHUNKS,
            "metadata": serializable_meta
        }, f)

class UploadResponse(BaseModel):
    message: str
    processed_count: int

@app.post("/api/v1/upload-resumes", response_model=UploadResponse)
async def upload_resumes(
    collection_name: str = Form("resume_chunks"),
    files: List[UploadFile] = File(...)
):
    """
    Ingests multiple resumes (PDF/DOCX), strips PII, chunks by section, 
    extracts structured LLM metadata, and inserts semantic vectors into Endee.
    Deduplicates by filename — re-uploading the same file replaces the old entry.
    """
    global ALL_CHUNKS
    
    load_collection(collection_name)
    
    count = 0
    for file in files:
        if not file.filename.endswith((".pdf", ".docx")):
            continue
            
        file_bytes = await file.read()
        safe_name = file.filename.replace(" ", "_")
        
        old_ids = [cid for cid in CANDIDATE_METADATA if cid.endswith("_" + safe_name)]
        for old_id in old_ids:
            del CANDIDATE_METADATA[old_id]
            ALL_CHUNKS = [c for c in ALL_CHUNKS if c["resume_id"] != old_id]
            print(f"Replaced existing entry: {old_id}")
        
        candidate_sections = process_resume(file_bytes, file.filename)
        candidate_id = str(uuid.uuid4())[:8] + "_" + safe_name
        
        metadata_obj = ingest_resume_document(candidate_id, candidate_sections, collection_name=collection_name)
        
        CANDIDATE_METADATA[candidate_id] = metadata_obj
        
        for sec, text in candidate_sections.items():
            if sec in ["general", "skills"]:
                continue
            if len(text.strip()) > 10:
                ALL_CHUNKS.append({
                    "resume_id": candidate_id,
                    "section": sec,
                    "text": text
                })
        
        count += 1
        
    save_collection(collection_name)
    return UploadResponse(message="Upload and processing complete.", processed_count=count)


class EvaluationReportResponse(BaseModel):
    job_description: str
    rankings: List[EvaluateResponse]

@app.post("/api/v1/evaluate-job", response_model=EvaluationReportResponse)
async def evaluate_job(req: JobDescriptionRequest):
    """
    Takes a Job Description, runs hybrid retrieval (Endee + BM25) across all resumes, 
    reranks via Cross-Encoder, and calls the LLM Evaluator for the final verdict.
    """
    load_collection(req.collection_name)
    
    if not ALL_CHUNKS:
        raise HTTPException(status_code=400, detail="No resumes have been ingested in this collection yet.")
        
    jd = req.job_description
    top_k = req.top_k
    
    # Cap top_k to actual number of unique candidates
    num_candidates = len(CANDIDATE_METADATA)
    if num_candidates > 0:
        top_k = min(top_k, num_candidates)
    
    # Extract JD requirements dynamically via LLM
    jd_extraction = extract_jd_metadata(jd)
    
    # We fetch a larger initial pool to ensure the reranker has good candidates to sort
    initial_pool_size = max(top_k * 3, 10)
    
    # 1. Hybrid Retrieval (Endee Semantic + BM25)
    retrieved_chunks = hybrid_retrieve(jd, ALL_CHUNKS, top_k=initial_pool_size, collection_name=req.collection_name)
    
    # 2. Cross-Encoder Reranking
    # We re-order the combined initial pool to find the absolute most relevant chunks to the JD
    reranked_chunks = rerank_chunks(jd, retrieved_chunks, top_k=initial_pool_size)
    
    # The hybrid search retrieves *chunks* not *candidates*. We need to group evidence back to candidates.
    candidate_evidence = {}
    for chunk in reranked_chunks:
        c_id = chunk["resume_id"]
        if c_id not in candidate_evidence:
            candidate_evidence[c_id] = []
        candidate_evidence[c_id].append(chunk)
        
    # We only process candidates that had at least one piece of strong evidence bubble up
    valid_candidates = list(candidate_evidence.keys())
    
    final_evaluations = []
    
    # 3. Final Scoring & LLM Agent Evaluation
    for c_id in valid_candidates:
        evidence = candidate_evidence[c_id]
        
        # Take the top 3 best evidence chunks per candidate to avoid token limits
        top_c_evidence = sorted(evidence, key=lambda x: x.get("rerank_score", 0), reverse=True)[:3]
        
        metadata = CANDIDATE_METADATA.get(c_id)
        
        if not metadata:
             continue 
             
        eval_resp = evaluate_candidate(
             candidate_id=c_id,
             jd_text=jd,
             jd_skills=jd_extraction.required_skills,
             evidence_chunks=top_c_evidence,
             metadata=metadata,
             required_years_exp=jd_extraction.required_years_exp
        )
        
        final_evaluations.append(eval_resp)
        
    final_evaluations.sort(key=lambda x: x.overall_score, reverse=True)
    
    return EvaluationReportResponse(
        job_description=jd[:200] + "...", 
        rankings=final_evaluations[:top_k]
    )

@app.get("/api/v1/collections", response_model=CollectionResponse)
def get_collections():
    collections = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".json"):
                collections.append(f.replace(".json", ""))
    if not collections:
        collections = ["resume_chunks"]
    return CollectionResponse(collections=collections)

@app.delete("/api/v1/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Deletes a collection's local JSON data and its Endee vector index."""
    path = os.path.join(DATA_DIR, f"{collection_name}.json")
    deleted = False
    if os.path.exists(path):
        os.remove(path)
        deleted = True
    try:
        requests.post(f"http://localhost:8080/api/v1/index/{collection_name}/delete")
    except:
        pass
    if deleted:
        return {"message": f"Collection '{collection_name}' deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Resume Shortlist Engine Active"}

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
