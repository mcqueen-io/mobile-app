from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "In-Car AI Assistant"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # Development frontend
        "https://your-production-domain.com"  # Production frontend
    ]
    
    # Rate Limiting
    RATE_LIMIT_WINDOW: int = 60  # 1 minute
    MAX_REQUESTS_PER_WINDOW: int = 100
    
    # AI Security
    MAX_CONTEXT_LENGTH: int = 4096
    MAX_RESPONSE_LENGTH: int = 1024
    ENABLE_CONTENT_FILTERING: bool = True
    ENABLE_PROMPT_INJECTION_PROTECTION: bool = True
    
    # Database
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # MongoDB
    MONGODB_URI: str
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    
    # External Services
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Voice Processing
    VOICE_MODEL_PATH: str = os.getenv("VOICE_MODEL_PATH", "./models/voice_embedding")
    
    # MCP Settings
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    
    # Google Cloud Settings
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: str
    
    # Gemini Settings
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash-001"
    GEMINI_MAX_OUTPUT_TOKENS: int = 1024
    GEMINI_TEMPERATURE: float = 0.7
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 