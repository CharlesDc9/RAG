from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from app.services.rag import RAGEngine
from io import BytesIO
import pypdf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")

app = FastAPI(title="RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine(MISTRAL_API_KEY)

class QuestionRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default"


@app.get("/")
async def root():
    return {
        "message": "Welcome to RAG API",
        "docs": "/docs",
        "endpoints": {
            "upload": "/upload",
            "ask": "/ask"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    try:
        # Check if file is PDF
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        content = await file.read()
        
        # Extract text from PDF
        pdf_file = BytesIO(content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        
        # Process the extracted text
        documents = await rag_engine.process_document(text_content)
        
        return {
            "message": "PDF processed successfully",
            "num_chunks": len(documents),
            "num_pages": len(pdf_reader.pages)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents."""
    try:
        response = await rag_engine.get_answer(
            question=request.question,
            thread_id=request.thread_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)