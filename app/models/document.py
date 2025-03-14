from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    """Question request model."""
    question: str
    
class QuestionResponse(BaseModel):
    """Question response model."""
    answer: str
    sources: List[str]
    
class DocumentResponse(BaseModel):
    """Document response model."""
    message: str
    document_id: str