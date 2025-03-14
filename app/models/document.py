from pydantic import BaseModel
from typing import List

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
    num_chunks: int
    num_pages: int