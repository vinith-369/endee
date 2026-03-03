import json
import math
import requests
from typing import List, Dict, Any, Tuple

from models import EvaluateResponse, CandidateExtraction

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

def compute_experience_score(candidate_exp: float, required_exp: float) -> float:
    """
    Experience scoring scales proportionally up to the requirement.
    It caps advantage beyond 1.5x the requirement to prevent bias from extreme seniority.
    """
    if required_exp <= 0: return 1.0 
    
    ratio = candidate_exp / required_exp
    
    if ratio >= 1.5:
        return 1.1 
    elif ratio >= 1.0:
        return 1.0
    else:
        
        return ratio

def compute_skills_score(candidate_skills: List[str], jd_skills: List[str]) -> float:
    """
    Determines basic lexical coverage of known candidate skills residing
    in the requested JD skills. 
    """
    if not jd_skills: return 1.0 # If no specific skills requested, free points
    if not candidate_skills: return 0.0
    
    cand_lower = [s.lower() for s in candidate_skills]
    matched = 0
    for req_skill in jd_skills:
        req_lower = req_skill.lower()
        if any(req_lower in cs or cs in req_lower for cs in cand_lower):
            matched += 1
            
    return min(matched / max(len(jd_skills), 1), 1.0)


def evaluate_candidate(
    candidate_id: str, 
    jd_text: str,
    jd_skills: List[str],
    evidence_chunks: List[Dict[str, Any]], 
    metadata: CandidateExtraction,
    required_years_exp: float = 0.0
) -> EvaluateResponse:
    """
    Executes the LLM Retrieval-Augmented Evaluation stage relying STRICTLY
    on the vector search/cross-encoder evidence chunks retrieved.
    Runs locally on Ollama.
    """
    
    # 1. Base Algorithmic Score Generation
    
    avg_rerank = 0.0
    raw_avg = 0.0
    if evidence_chunks:
        scores = [c.get("rerank_score", 0) for c in evidence_chunks]
        valid_scores = [s for s in scores if isinstance(s, (int, float)) and not math.isnan(s) and not math.isinf(s)]
        if valid_scores:
            raw_avg = sum(valid_scores) / len(valid_scores)
            # Min-max normalization: ms-marco cross-encoder logits typically range [-10, +10]
            # Clamp to this range then scale linearly to [0, 1]
            clamped = max(-10.0, min(10.0, raw_avg))
            avg_rerank = (clamped + 10.0) / 20.0  
        else:
            raw_avg = 0.0
            avg_rerank = 0.5  
    
    exp_score = compute_experience_score(metadata.total_years_experience, required_years_exp)
    skills_score = compute_skills_score(metadata.skills, jd_skills)
    
    # Semantic (30%), Skills (50%), Experience (20%)
    w_sem, w_skills, w_exp = 0.3, 0.5, 0.2
    
    if required_years_exp <= 0 and not jd_skills:
        # Only Semantic matters
        w_sem, w_skills, w_exp = 1.0, 0.0, 0.0
    elif required_years_exp <= 0:
        # No exp required -> Split 50/50 Semantic & Skills
        w_sem, w_skills, w_exp = 0.5, 0.5, 0.0
    elif not jd_skills:
        # No specific skills required -> Semantic 80%, Exp 20%
        w_sem, w_skills, w_exp = 0.8, 0.0, 0.2
        
    overall_score = (avg_rerank * w_sem) + (skills_score * w_skills) + (exp_score * w_exp)
    overall_score = min(overall_score, 1.0)
    
    
    clean_score = round(overall_score, 4) if not (math.isnan(overall_score) or math.isinf(overall_score)) else 0.0
    chunk_texts = [c["text"] for c in evidence_chunks]
    
    print(f"\n--- EVALUATION FOR CANDIDATE: {candidate_id} ---")
    print(f"JD Required Exp: {required_years_exp} | Candidate Exp: {metadata.total_years_experience}")
    print(f"JD Required Skills: {jd_skills} | Candidate Skills: {metadata.skills}")
    print(f"-> Base Semantic Rerank (Logistic): {avg_rerank:.4f} (Raw: {raw_avg if evidence_chunks else 0:.4f})")
    print(f"-> Experience Component Score: {exp_score:.4f}")
    print(f"-> Skills Component Score: {skills_score:.4f}")
    print(f"-> Applied Weights: Semantic({w_sem}), Skills({w_skills}), Exp({w_exp})")
    print(f"==> FINAL COMBINED SCORE: {clean_score:.4f}\n")
    
    return EvaluateResponse(
        candidate_id=candidate_id,
        overall_score=clean_score,
        score_details={
            "semantic": round(avg_rerank, 4),
            "skills": round(skills_score, 4),
            "experience": round(exp_score, 4),
            "weights": {"semantic": w_sem, "skills": w_skills, "experience": w_exp}
        },
        evidence_chunks=chunk_texts,
    )
