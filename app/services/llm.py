from langchain.chat_models import init_chat_model
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    """Service for LLM operations."""
    
    def __init__(self):
        self.llm = init_chat_model(
            settings.MODEL_NAME,
            model_provider="mistralai"
        )
    
    async def get_response(self, question: str, context: str) -> str:
        """Get LLM response for a question."""
        prompt = f"""You are an assistant that MUST search the provided database before answering. 
        ALWAYS use the 'retrieve' tool first to find relevant information. 
        Do not rely on your general knowledge without checking the database first. 
        If you can't find relevant information in the database, say so explicitly.
        Use three sentences maximum and keep the answer as concise as possible.
        If you can't find the answer, say so explicitly.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        return self.llm.invoke(prompt).content