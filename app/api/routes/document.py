from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.document import QuestionRequest, QuestionResponse, DocumentResponse
from app.services.document import DocumentService
from app.services.rag_service import RAGService
from app.services.llm import LLMService
from io import BytesIO
import pypdf
import logging
from typing import List, Dict

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(DocumentService),
    rag_service: RAGService = Depends(RAGService)
):
    """Upload and process a document."""
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
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )
        
        # Process document
        documents = await document_service.process_document(text_content)
        
        # Add to vector store
        await rag_service.add_documents(documents)
        
        return DocumentResponse(
            message="Document processed successfully",
            num_chunks=len(documents),
            num_pages=len(pdf_reader.pages)
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/question", response_model=QuestionResponse)
async def ask_question(
    question: QuestionRequest,
    rag_service: RAGService = Depends(RAGService)
):
    """Ask a question about the uploaded documents."""
    try:
        # Get answer directly from RAG service
        result = await rag_service.get_answer(question.question)
        return result
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/documents", response_model=Dict)
async def list_documents(
    rag_service: RAGService = Depends(RAGService)
):
    """List all documents in the system."""
    try:
        documents = rag_service.engine.get_all_documents()
        return {
            "total_documents": len(documents.get('ids', [])),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}", response_model=Dict)
async def get_document_details(
    document_id: str,
    rag_service: RAGService = Depends(RAGService)
):
    """Get details about a specific document."""
    try:
        document = rag_service.engine.get_document_by_metadata({"source": document_id})
        if not document or not document.get('ids'):
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(RAGService)
):
    """Delete a specific document and all its chunks."""
    try:
        rag_service.engine.delete_documents_by_metadata({"source": document_id})
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
