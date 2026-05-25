# title="app/config.py"
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Multimodal Advanced RAG System"
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "storage", "uploads")
    OUTPUT_DIR: str = os.path.join(os.getcwd(), "storage", "outputs")
    
    # API Credentials
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "mock-groq-key")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "x")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "mock-tavily-key")
    
    # Model Controls
    FLORENCE_MODEL: str = "microsoft/Florence-2-base"
    YOLO_LAYOUT_MODEL: str = "lllyasviel/yolov8x-doclayout" # Or standard yolov8x-cls configured for layouts
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure physical runtime directories exist safely
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
