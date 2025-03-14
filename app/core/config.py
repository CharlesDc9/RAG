from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "RAG API"
    ENVIRONMENT: str
    MISTRAL_API_KEY: str
    MODEL_NAME: str
    EMBEDDINGS_MODEL: str
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()