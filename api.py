"""
api.py — FastAPI REST API for the Sarkari Scheme Navigator.

Endpoints:
    GET  /health  — Check that the vector index is up and how many chunks are loaded.
    POST /ask     — Ask a question; returns a grounded answer with source citations.
    POST /ingest  — Rebuild the ChromaDB index from PDFs in data/raw/ (admin use).

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the project root is importable regardless of where you run this from
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import TOP_K, SIM_THRESHOLD, LLM_MODEL_NAME, COLLECTION_NAME, CHROMA_DIR, RAW_DIR
from src.generator import answer
from src.vectorstore import get_chroma_client, get_embedder, populate_db
from src.ingest import load_documents
from src.chunker import chunk_documents


app = FastAPI(
    title="Sarkari Scheme Navigator API",
    description=(
        "REST API for grounded RAG over Indian Government Scheme documents. "
        "Every answer cites the exact source clause; the system abstains rather "
        "than guessing when confidence is below threshold."
    ),
    version="1.0.0",
)

# Allow any frontend (browser, Streamlit, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="Your question about any government scheme.",
        example="What is the maximum loan amount under Mahila Samridhi Yojana?",
    )
    k: Optional[int] = Field(
        default=TOP_K, ge=1, le=20,
        description="Number of context chunks to retrieve (1–20).",
    )
    threshold: Optional[float] = Field(
        default=SIM_THRESHOLD, ge=0.0, le=1.0,
        description="Minimum cosine similarity before the system abstains (0.0–1.0).",
    )


class SourceItem(BaseModel):
    source: str
    page: int
    score: float
    excerpt: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    abstained: bool
    confidence: float


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    indexed_chunks: int
    model: str
    chroma_dir: str


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    raw_dir: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Verify that the ChromaDB vector index is online and report its size.
    """
    chunk_count = 0
    try:
        col = get_chroma_client().get_collection(
            COLLECTION_NAME, embedding_function=get_embedder()
        )
        chunk_count = col.count()
    except Exception:
        pass  # Index not built yet — that's fine, just report 0

    return HealthResponse(
        status="ok",
        collection_name=COLLECTION_NAME,
        indexed_chunks=chunk_count,
        model=LLM_MODEL_NAME,
        chroma_dir=str(CHROMA_DIR),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(req: AskRequest):
    """
    Ask a question about any Indian government scheme.

    The system retrieves the most relevant document clauses, checks confidence,
    and either returns a cited answer or abstains gracefully.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Pydantic already applies the Field defaults, so req.k and req.threshold
    # are never None here — but we guard anyway for safety.
    result = answer(
        query=question,
        k=req.k if req.k is not None else TOP_K,
        threshold=req.threshold if req.threshold is not None else SIM_THRESHOLD,
    )

    return AskResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[
            SourceItem(
                source=s["source"],
                page=s["page"],
                score=s["score"],
                excerpt=s.get("excerpt"),
            )
            for s in result.get("sources", [])
        ],
        abstained=result["abstained"],
        confidence=result["confidence"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
def ingest():
    """
    Rebuild the ChromaDB vector index from all PDFs in data/raw/.

    Safe to call multiple times — drops and recreates the collection each time.
    """
    try:
        docs = load_documents(raw_dir=RAW_DIR)
        chunks = chunk_documents(docs)
        populate_db(chunks, persist_dir=CHROMA_DIR)
        return IngestResponse(
            status="ok",
            documents_loaded=len({d["source"] for d in docs}),
            chunks_created=len(chunks),
            raw_dir=str(RAW_DIR),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)