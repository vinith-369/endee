import fitz  # PyMuPDF
import docx
import re
from typing import Dict, List, Tuple
from io import BytesIO

# ==========================================
# 1. PII Stripping
# ==========================================

def remove_pii(text: str) -> str:
    """
    Strips common Personal Identifiable Information (Emails, Phones, URLs, Addresses)
    to reduce embedding noise and prevent evaluation bias.
    """
    # Remove Emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    # Remove phone numbers (simplified)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
    # Remove URLs / LinkedIn
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    text = re.sub(r'linkedin\.com/in/[^\s]+', '[LINKEDIN]', text)
    return text

# ==========================================
# 2. File Parsing
# ==========================================

def parse_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def parse_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

# ==========================================
# 3. Section Segmentation Logic
# ==========================================

# Common resume section headers
SECTION_HEADERS = {
    "education": [r"\beducation\b", r"\bacademic background\b", r"\bdegrees\b"],
    "experience": [r"\bexperience\b", r"\bemployment history\b", r"\bwork history\b", r"\bprofessional experience\b"],
    "projects": [r"\bprojects\b", r"\bpersonal projects\b", r"\bacademic projects\b"],
    "skills": [r"\bskills\b", r"\btechnologies\b", r"\btechnical skills\b", r"\bcore competencies\b"],
    "certificates": [r"\bcertificates\b", r"\bcertifications\b", r"\blicenses\b"],
    "about": [r"\babout me\b", r"\bsummary\b", r"\bprofile\b", r"\bobjective\b"],
    "achievements": [r"\bachievements\b", r"\bawards\b", r"\bhonors\b", r"\bpublications\b"]
}

def identify_section(line: str) -> str:
    """Matches a line against known section headers."""
    line_lower = line.strip().lower()
    if len(line_lower) > 30: # Unlikely to be a header
        return None
        
    for section, patterns in SECTION_HEADERS.items():
        for pattern in patterns:
            # We enforce exact matches or matches at the start of a short line
            if re.search(pattern, line_lower):
                return section
    return None

def segment_resume(text: str) -> Dict[str, str]:
    """
    Splits the parsed resume text into logical blocks based on headers.
    Returns a dictionary mapping section_name -> section text block.
    """
    sections = {}
    current_section = "general" # Fallback/header section (likely name/contact info)
    current_text = []
    
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        matched_section = identify_section(line)
        
        if matched_section:
            # Save the previous section
            if current_text:
                if current_section not in sections:
                    sections[current_section] = ""
                sections[current_section] += "\n".join(current_text) + "\n"
            
            # Start new section
            current_section = matched_section
            current_text = []
        else:
            current_text.append(line)
            
    # Save the final section
    if current_text:
        if current_section not in sections:
            sections[current_section] = ""
        sections[current_section] += "\n".join(current_text) + "\n"
        
    return sections

def process_resume(file_bytes: bytes, filename: str) -> Dict[str, str]:
    """Main pipeline to parse, strip PII, and segment roughly."""
    if filename.lower().endswith(".pdf"):
        raw_text = parse_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        raw_text = parse_docx(file_bytes)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
        
    clean_text = remove_pii(raw_text)
    segmented = segment_resume(clean_text)
    return segmented
