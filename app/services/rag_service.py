from typing import List, Tuple, Dict
from app.services.rag import RAGEngine, State, Search
from app.core.config import get_settings
from langchain_core.documents import Document

settings = get_settings()

class RAGService:
    """Service for RAG operations using the RAGEngine."""
    
    def __init__(self):
        self.engine = RAGEngine(settings.MISTRAL_API_KEY)
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        for doc in documents:
            await self.engine.process_document(doc.page_content, doc.metadata)
    
    async def retrieve_context(self, question: str) -> Tuple[str, List[str]]:
        """Retrieve context for a question."""
        try:
            # Use the engine's functions directly
            state = {"question": question}
            query_result = self.engine.analyze_query(state)
            state.update(query_result)  # Add query to state
            context_result = self.engine.retrieve(state)
            
            # Extract context
            context_docs = context_result["context"]
            context_text = "\n\n".join(doc.page_content for doc in context_docs)
            
            # Extract sources (metadata)
            sources = []
            for doc in context_docs:
                if "source" in doc.metadata:
                    sources.append(str(doc.metadata["source"]))
                elif "section" in doc.metadata:
                    sources.append(f"Section: {doc.metadata['section']}")
            
            return context_text, sources
        except Exception as e:
            print(f"Error in retrieve_context: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    async def get_answer(self, question: str) -> Dict:
        """Get answer for a question directly."""
        try:
            result = await self.engine.get_answer(question)
            
            # Ensure the response format matches QuestionResponse model
            return {
                "answer": result["answer"],
                "sources": [str(doc) for doc in result["context"]]  # Convert to strings
            }
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": ["Error occurred during processing"]
            }