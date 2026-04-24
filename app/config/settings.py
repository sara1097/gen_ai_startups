# app/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Centralized configuration management using pydantic-settings.
    Loads from environment variables or .env file.
    """
    
    # API Keys
    GROQ_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    HF_TOKEN: str
    
    # Models
    LLM_MODEL: str = "qwen/qwen3-32b"
    FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    SPARSE_MODEL: str = "Qdrant/bm25"
    
    # App Settings
    COLLECTION_NAME: str = "startups"
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache()
def get_settings() -> Settings:
    """
    Dependency provider for settings.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
