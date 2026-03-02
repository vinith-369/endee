import os
import json
import requests
from pydantic import ValidationError
from models import CandidateExtraction, JDExtraction

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

def extract_structured_metadata(resume_sections: dict) -> CandidateExtraction:
    """
    Uses local Ollama (qwen2.5:7b) to extract structured data from a resume.
    Focuses on the generic skills list and parsing the experience timelines.
    """
    
    # We construct a prompt primarily containing the skills/experience sections
    context_text = ""
    for sec in ["skills", "experience", "education", "general"]:
        if sec in resume_sections:
            context_text += f"\n--- {sec.upper()} ---\n{resume_sections[sec]}\n"
            
    # Qwen schema instruction
    schema_json = CandidateExtraction.model_json_schema()
    
    prompt = f"""
    You are an expert HR data extraction system. Your task is to accurately extract 
    metadata from candidate resumes in a strict structured JSON format. Do not guess 
    information that is not present.
    
    Extract the following from the provided text:
    1. A comprehensive list of technical and soft 'skills'.
    2. Details of their professional 'experience', breaking it down by role.
    3. Their 'total_years_experience' across all relevant roles.

    RESUME TEXT:
    {context_text}
    """
        
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": f"You are a precise data extraction AI. Always output strict JSON matching exactly this schema: {json.dumps(schema_json)}"},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        result_content = response.json()["message"]["content"]
        
        # Parse text into our Pydantic schema
        parsed_data = json.loads(result_content)
        return CandidateExtraction(**parsed_data)
        
    except Exception as e:
        print(f"Ollama Extraction failed: {e}")
        # Return empty structured fallback
        return CandidateExtraction(skills=[], experience=[], total_years_experience=0)

        
def extract_jd_metadata(jd_text: str) -> JDExtraction:
    """
    Uses local Ollama to extract explicit technical/soft skills and minimum years 
    of experience required directly from the Job Description text.
    """
    schema_json = JDExtraction.model_json_schema()
    
    prompt = f"""
    You are an expert HR parser. Read the following Job Description and strictly extract:
    1. A list of explicitly required 'skills' (both technical and soft skills).
    2. The minimum required 'required_years_exp' as a float (e.g. 3.0, 5.0). If no explicit minimum years are stated, return 0.0.

    JOB DESCRIPTION:
    {jd_text}
    """
        
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": f"You are a precise data extraction AI. Always output strict JSON matching exactly this schema: {json.dumps(schema_json)}"},
                {"role": "user", "content": prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        result_content = response.json()["message"]["content"]
        parsed_data = json.loads(result_content)
        
        return JDExtraction(**parsed_data)
        
    except Exception as e:
        print(f"Ollama JD Extraction failed: {e}")
        return JDExtraction(required_skills=[], required_years_exp=0.0)

def ingest_resume_document(resume_id: str, sections: dict, collection_name: str = "resume_chunks"):
    '''
    Master ingestion function
    1. Extracts metadata
    2. Embeds sections in Endee Vector DB
    '''
    # 1. Structural Metadata
    metadata_obj = extract_structured_metadata(sections)
    
    # 2. Vectorize Context Elements
    from vector_store import upsert_chunks
    
    chunks = []
    for sec_name, sec_text in sections.items():
        if sec_name in ["general", "skills"]:
            continue
        if len(sec_text.strip()) > 10: # ensure it's not simply empty
            chunks.append({"section": sec_name, "text": sec_text})
            
    # DEBUG PRINT FOR USER
    print(f"\n{'='*50}")
    print(f"RESUME ID: {resume_id}")
    print(f"EXTRACTED METADATA:\n{metadata_obj.model_dump_json(indent=2)}")
    print(f"\nPREPARED CHUNKS FOR EMBEDDING ({len(chunks)} total):")
    for i, c in enumerate(chunks):
        print(f"  [Chunk {i+1}] Section: {c['section']}")
        preview = c['text'][:150].replace('\n', ' ')
        print(f"  Text preview: {preview}...\n")
    print(f"{'='*50}\n")
            
    # Batch upsert
    upsert_chunks(resume_id, chunks, collection_name=collection_name)
    
    return metadata_obj
