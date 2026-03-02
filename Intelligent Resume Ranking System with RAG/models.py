from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ==========================================
# 1. API Request / Response Models
# ==========================================

class JobDescriptionRequest(BaseModel):
    job_description: str = Field(..., description="The raw job description text")
    collection_name: str = Field(default="resume_chunks", description="The repository of resumes to search within")
    top_k: int = Field(default=5, description="Number of top candidates to retrieve")

class CollectionResponse(BaseModel):
    collections: List[str]

class JDExtraction(BaseModel):
    required_skills: List[str] = Field(default_factory=list, description="Core technical and soft skills required")
    required_years_exp: float = Field(default=0.0, description="Minimum years of experience actively required (0 if none)")

class EvaluateResponse(BaseModel):
    candidate_id: str = ""
    overall_score: float = 0.0
    score_details: Dict = Field(default_factory=dict)
    evidence_chunks: List[str] = Field(default_factory=list)


# ==========================================
# 2. LLM Extraction Models (Structured Output)
# ==========================================

class ExperienceEntry(BaseModel):
    role: str = Field(default="", description="Job title or role of the candidate")
    field: str = Field(default="", description="The primary industry, field, or domain of experience")
    technologies: List[str] = Field(default_factory=list, description="List of technologies, tools, or frameworks used")
    duration_years: float = Field(default=0.0, description="Duration worked in years. Example: 1.5")

class CandidateExtraction(BaseModel):
    skills: List[str] = Field(default_factory=list, description="Comprehensive list of technical and soft skills")
    experience: List[ExperienceEntry] = Field(default_factory=list, description="List of professional job experiences")
    total_years_experience: float = Field(default=0.0, description="Total cumulative years of experience across all roles")
