import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "Enterprise AI Analytics & Document Intelligence Copilot"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "chroma_db"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "analytics.db"
    
    # Retrieval Hyperparameters
    CHUNK_SIZE: int = 500  # Characters
    CHUNK_OVERLAP: int = 100  # Characters
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    TOP_K: int = 4
    MIN_RELEVANCE_SCORE: float = 0.015
    
    # Vector Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # LLM & API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

settings = Settings()
