import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class AppSettings(BaseSettings):
    APP_NAME: str = "Enterprise Multimodal RAG Assistant"
    VERSION: str = "2.0.0"
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", os.path.join(DATA_DIR, "chroma_db"))
    CHAT_HISTORY_DIR: str = os.getenv("CHAT_HISTORY_DIR", os.path.join(DATA_DIR, "chat_history"))
    MD5_LOG_PATH: str = os.path.join(DATA_DIR, "processed_hashes.txt")
    
    # Models & Retrieval
    COLLECTION_NAME: str = "enterprise_multimodal_kb"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    HF_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    
    # Chunking & RAG
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1200))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    TOP_K: int = int(os.getenv("TOP_K_RETRIEVAL", 4))
    
    # Multimodal Captioner
    CAPTION_MODEL: str = "nlpconnect/vit-gpt2-image-captioning"

settings = AppSettings()
