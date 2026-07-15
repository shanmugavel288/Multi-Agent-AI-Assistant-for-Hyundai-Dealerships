# config/settings.py
# ─────────────────────────────────────────────────────────────
# Centralised configuration loader for Hyundai Agentic AI
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM ──────────────────────────────────────────────────
    LLM_PROVIDER: str        = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY: str        = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str          = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    GEMINI_API_KEY: str      = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str        = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    # ── Database ─────────────────────────────────────────────
    DB_HOST: str             = os.getenv("DB_HOST", "localhost")
    DB_PORT: str             = os.getenv("DB_PORT", "5432")
    DB_NAME: str             = os.getenv("DB_NAME", "hyundai_db")
    DB_USER: str             = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str         = os.getenv("DB_PASSWORD", "")

    # ── ChromaDB ─────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str  = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
    CHROMA_COLLECTION: str   = os.getenv("CHROMA_COLLECTION_NAME", "hyundai_knowledge_base")

    # ── Paths ─────────────────────────────────────────────────
    PDF_PATH: str            = os.getenv("PDF_PATH","C:/Users/shanm/Downloads/hyundai_agentic_ai/hyundai_ai/data/Hyundai_RAG_Knowledge_Base.pdf")
    EMBEDDING_MODEL: str     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()
