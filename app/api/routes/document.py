from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.document import QuestionRequest, QuestionResponse, DocumentResponse
from app.services.document import DocumentService
from app.services.rag import RAGService
from app.services.llm import LLMService

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(DocumentService),
    rag_service: RAGService = Depends(RAGService)
):
    """Upload and process a document."""
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Process document
        documents = await document_service.process_document(text_content)
        
        # Add to vector store
        await rag_service.add_documents(documents)
        
        return DocumentResponse(
            message="Document processed successfully",
            document_id=documents[0].metadata["id"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/question", response_model=QuestionResponse)
async def ask_question(
    question: QuestionRequest,
    rag_service: RAGService = Depends(RAGService),
    llm_service: LLMService = Depends(LLMService)
):
    """Ask a question about the uploaded documents."""
    try:
        # Retrieve relevant context
        context, sources = await rag_service.retrieve_context(question.question)
        
        # Get LLM response
        answer = await llm_service.get_response(question.question, context)
        
        return QuestionResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))