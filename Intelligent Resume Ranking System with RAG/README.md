# Resume Shortlisting System (RAG)

An industry-grade, explainable AI hiring assistant that evaluates candidate resumes against Job Descriptions using Hybrid Retrieval-Augmented Generation (RAG) with the **Endee Vector Database**.

## Project Overview

Traditional ATS systems rely on brittle keyword matching, rejecting candidates who don't use exact terminology. This system deeply understands candidate profiles through semantic chunking, structured LLM extraction, and evidence-based grading — producing transparent, explainable scores with zero LLM hallucination in the final output.

## Architecture

```
Resume PDFs ──► Parser ──► LLM Extraction ──► Endee Vector DB
                  │              │                    │
                  │         skills, exp          embeddings
                  │              │                    │
                  ▼              ▼                    ▼
             JD Text ──► LLM Extraction ──► Hybrid Retrieval ──► Cross-Encoder ──► Score
                         (skills, exp)     (Endee + BM25)       (Reranking)       (30/50/20)
```

### Pipeline Components

| Module | File | Purpose |
|---|---|---|
| **PDF/DOCX Parser** | `parser.py` | Parses resumes, strips PII (emails, phones, URLs), segments by section headers (Education, Experience, Projects, Skills) |
| **LLM Extractor** | `extractor.py` | Uses **Ollama** (`qwen2.5:7b`) to extract structured JSON — skills list, experience entries, and total years — from both resumes and JDs |
| **Vector Store** | `vector_store.py` | Encodes chunks with `BAAI/bge-large-en-v1.5` (1024-dim), stores in **Endee** with cosine similarity |
| **Hybrid Retriever** | `retriever.py` | Combines Endee semantic search + BM25 keyword matching with min-max score normalization |
| **Cross-Encoder Reranker** | `reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` — reranks merged chunks by JD relevance |
| **Scoring Engine** | `evaluator.py` | Computes final score from 3 components (see below) |
| **API Server** | `main.py` | FastAPI with upload, evaluate, collections management endpoints |
| **Frontend** | `static/` | Monochrome black & white UI with drag-drop upload, collection selector, and score breakdown cards |

## Scoring Formula

```
Final Score = (Semantic × 30%) + (Skills × 50%) + (Experience × 20%)
```

### Component Breakdown

**Semantic Score (30%)** — Cross-encoder logits averaged over top 3 evidence chunks, normalized linearly from `[-10, +10]` to `[0, 1]`.

**Skills Score (50%)** — Substring matching of JD-required skills against candidate's extracted skills. `matched / total_required`.

**Experience Score (20%)** — `candidate_years / required_years`, capped at 1.1× for over-experience.

### Dynamic Weighting

If the JD has no specific skills or experience requirements, weights automatically redistribute:

| JD Has Skills? | JD Has Exp? | Semantic | Skills | Experience |
|---|---|---|---|---|
| ✅ | ✅ | 30% | 50% | 20% |
| ✅ | ❌ | 50% | 50% | 0% |
| ❌ | ✅ | 80% | 0% | 20% |
| ❌ | ❌ | 100% | 0% | 0% |

## Persistent Collections

Resumes are organized into named **Collections** (e.g., `backend_engineers`, `ml_interns`). Each collection:
- Has its own Endee vector index
- Stores metadata to `data/{collection_name}.json` on disk
- Survives server restarts — no need to re-ingest
- Can be selected from a dropdown in the UI when evaluating

**API Endpoints:**
- `GET /api/v1/collections` — List available collections
- `DELETE /api/v1/collections/{name}` — Delete a collection

## How Endee Is Used

**Endee (nD) Vector Database** is the local semantic storage layer:
- `vector_store.py` connects via HTTP at `http://localhost:8080/api/v1`
- Dynamically creates indexes per collection (`1024`-dim, cosine similarity)
- On evaluation, performs `/vector/search` to pull the most relevant candidate chunks
- Results are merged with BM25 lexical results before cross-encoder reranking

---

## Setup & Execution

### Prerequisites
- **Python 3.10+**
- **Ollama** — `ollama run qwen2.5:7b`
- **Endee Database** — Running locally (`./run.sh` in the Endee repository)

### Installation

```bash
cd resume-shortlister
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pymupdf python-docx sentence-transformers rank-bm25 pydantic requests python-multipart msgpack
```

### Running

```bash
uvicorn main:app --reload --port 9000
```

Models (`bge-large-en-v1.5` + `ms-marco-MiniLM-L-6-v2`) download on first boot (~1.4GB to `~/.cache/huggingface`).

### Usage

Open `http://127.0.0.1:9000` in your browser for the full UI, or use the API directly:

**Upload Resumes:**
```bash
curl -X POST http://127.0.0.1:9000/api/v1/upload-resumes \
  -F 'collection_name=my_candidates' \
  -F 'files=@resume1.pdf' \
  -F 'files=@resume2.pdf'
```

**Evaluate JD:**
```bash
curl -X POST http://127.0.0.1:9000/api/v1/evaluate-job \
  -H 'Content-Type: application/json' \
  -d '{
    "job_description": "Senior Python Developer with FastAPI, 3+ years experience",
    "collection_name": "my_candidates",
    "top_k": 3
  }'
```

**List Collections:**
```bash
curl http://127.0.0.1:9000/api/v1/collections
```

**Delete Collection:**
```bash
curl -X DELETE http://127.0.0.1:9000/api/v1/collections/my_candidates
```
