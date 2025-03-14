from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "RAG API"
    ENVIRONMENT: str = "development"
    MISTRAL_API_KEY: str
    MODEL_NAME: str = "mistral-large-latest"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # LangSmith settings
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "default"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()