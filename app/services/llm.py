from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    """Service for LLM operations."""
    
    def __init__(self):
        self.llm = init_chat_model(
            settings.MODEL_NAME,
            model_provider="mistralai"
        )
        
        # Define prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""
        self.prompt = PromptTemplate.from_template(template)
    
    async def get_response(self, question: str, context: str) -> str:
        """Get LLM response for a question."""
        messages = self.prompt.invoke({"question": question, "context": context})
        response = self.llm.invoke(messages)
        return response.content