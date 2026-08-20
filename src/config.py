import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (local dev)
load_dotenv()

# ── GOOGLE_API_KEY: Streamlit Cloud secrets → .env → env var ─────────────────
def _get_google_api_key() -> str:
    # 1. Try Streamlit secrets (production on Streamlit Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Fall back to environment variable / .env file
    return os.getenv("GOOGLE_API_KEY", "")

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Embedding and Vector DB Configuration
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sarkari_schemes"

# Chunking Configuration
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Retrieval & Generation Configuration
TOP_K = 5
SIM_THRESHOLD = 0.35  # Minimum cosine similarity score (1 - cosine_distance) to proceed with generation

# LLM Configuration
GOOGLE_API_KEY = _get_google_api_key()
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.6-flash")
